import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

# ----------------------------------------------------------------------
#  Learnable α-gate (one scalar per orientation, shared across channels)
# ----------------------------------------------------------------------
class OrientationGate(nn.Module):
    def __init__(self, field_type: enn.FieldType):
        super().__init__()
        self.field_type = field_type
        self.gsize = len(field_type.gspace.fibergroup.elements)
        self.alpha_logits = nn.Parameter(torch.zeros(self.gsize))

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        t = x.tensor
        if t.shape[1] % self.gsize: # not a regular field
            return x
        
        n_rep = t.shape[1] // self.gsize
        alpha = torch.sigmoid(self.alpha_logits)
        alpha = alpha.view(1, 1, self.gsize, 1, 1)
        
        t = t.view(t.shape[0], n_rep, self.gsize, t.shape[2], t.shape[3]) * alpha
        t = t.view_as(x.tensor)
        return enn.GeometricTensor(t, self.field_type)

# ----------------------------------------------------------------------
#  R2Conv + α-gate (drop-in replacement for enn.R2Conv)
# ----------------------------------------------------------------------
class PartialR2Conv(nn.Module):
    def __init__(self, in_type, out_type, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.gate = OrientationGate(out_type)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return self.gate(self.conv(x))

class PartialResNetBlock(torch.nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, kernel_size: int, stride: int = 1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type)
        self.conv1 = PartialR2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
        self.conv2 = PartialR2Conv(
            out_type, out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.shortcut = None
        if stride != 1 or in_type.size != out_type.size:
            self.shortcut = PartialR2Conv(
                in_type, out_type,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return out + identity

# ------------------------------------------------------------------
#  ResNet-44 (6*n + 2 layers with n = 7)
# ------------------------------------------------------------------
class P4mW_ResNet44(torch.nn.Module):
    def __init__(self, n: int = 7, num_classes: int = 43, in_channels: int = 3, group: str = "P4M", widths: list):
        super().__init__()

        # Select group based on config
        if group.upper() == "P4M":
            self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        elif group.upper() == "P4":
            self.r2_act = gspaces.Rot2dOnR2(N=4)
        else:
            raise ValueError(f"Group '{group}' not supported.")

        channels = widths
        self.in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        
        # Stem
        self.conv1_out = enn.FieldType(self.r2_act, channels[0] * [self.r2_act.regular_repr])
        self.conv1 = PartialR2Conv(
            self.in_type, self.conv1_out,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Stages
        self.stage1 = self._make_stage(self.conv1_out, channels[0], n, stride=1)
        self.stage2 = self._make_stage(self.stage1[-1].out_type, channels[1], n, stride=2)
        self.stage3 = self._make_stage(self.stage2[-1].out_type, channels[2], n, stride=2)
        
        # Head
        self.bn_final = enn.InnerBatchNorm(self.stage3[-1].out_type)
        self.relu_final = enn.ReLU(self.stage3[-1].out_type)
        self.gpool = enn.GroupPooling(self.stage3[-1].out_type)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(channels[2], num_classes)
        
        self.apply(self._init_weights)

    def _make_stage(self, in_type: enn.FieldType, width: int, num_blocks: int, stride: int):
        out_type = enn.FieldType(self.r2_act, width * [self.r2_act.regular_repr])
        layers = [PartialResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(PartialResNetBlock(out_type, out_type, 3, stride=1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu_final(self.bn_final(x))
        x = self.gpool(x)
        x = self.avgpool(x.tensor)
        x = torch.flatten(x, 1)
        return self.fc(x)