import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn

# ----------------------------------------------------------------------
#  Differentiable Gumbel-Softmax Gate (Replaces OrientationGate)
#  Implements the core idea of Pose-Selective Sparse G-CNNs.
# ----------------------------------------------------------------------
class DifferentiableMaskGate(nn.Module):
    def __init__(self, field_type: enn.FieldType, initial_temp: float = 1.0):
        super().__init__()
        self.field_type = field_type
        self.gsize = len(field_type.gspace.fibergroup.elements)
        
        # Logits for the binary mask, as described in the PDF 
        self.b_logits = nn.Parameter(torch.zeros(self.gsize))
        
        # Temperature for Gumbel-Softmax, will be annealed during training 
        self.temp = nn.Parameter(torch.tensor(initial_temp), requires_grad=False)
        
        # A flag to control whether to add Gumbel noise.
        # Turned off after annealing phase for fine-tuning.
        self.use_noise = True

    def get_mask(self):
        """
        Generates the relaxed binary mask using Gumbel-Softmax.
        """
        # During training with noise, use the Gumbel-Softmax trick 
        if self.training and self.use_noise:
            # Gumbel noise and add it to the logits
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.b_logits)))
            y = (self.b_logits + gumbel_noise) / self.temp
        else:
            # At inference or during fine-tuning, use deterministic logits 
            y = self.b_logits / self.temp
            
        # The relaxed Bernoulli variable pi_g, which lives in (0, 1) 
        return torch.sigmoid(y)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        t = x.tensor
        if t.shape[1] % self.gsize: # not a regular field
            return x
        
        n_rep = t.shape[1] // self.gsize
        
        # Get the current mask [π_g]
        mask = self.get_mask()
        mask = mask.view(1, 1, self.gsize, 1, 1) # Reshape for broadcasting
        
        # Apply the mask to the feature map 
        t = t.view(t.shape[0], n_rep, self.gsize, t.shape[2], t.shape[3]) * mask
        t = t.view_as(x.tensor)
        
        return enn.GeometricTensor(t, self.field_type)

# ----------------------------------------------------------------------
#  R2Conv + Gumbel-Softmax Gate
# ----------------------------------------------------------------------
class SparseR2Conv(nn.Module):
    def __init__(self, in_type, out_type, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.gate = DifferentiableMaskGate(out_type)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return self.gate(self.conv(x))

class SparseResNetBlock(torch.nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, kernel_size: int, stride: int = 1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type)
        self.conv1 = SparseR2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
        self.conv2 = SparseR2Conv(
            out_type, out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.shortcut = None
        if stride != 1 or in_type.size != out_type.size:
            self.shortcut = SparseR2Conv(
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
#  Pose-Selective Sparse ResNet-44
# ------------------------------------------------------------------
class PoseSelectiveSparse_ResNet44(torch.nn.Module):
    def __init__(self, n: int = 7, num_classes: int = 43, in_channels: int = 3, group: str = "P4M", widths: list = [11, 23, 45]):
        super().__init__()
        # ... (Group selection logic is unchanged) ...
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
        self.conv1 = SparseR2Conv(
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
        layers = [SparseResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(SparseResNetBlock(out_type, out_type, 3, stride=1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (Forward pass is unchanged) ...
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