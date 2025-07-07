import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

# This file defines the dense (non-pruned) baseline versions of your models
# for accurate FLOPs comparison.

# --- Baseline ResNet Block (uses standard e2cnn convolution) ---
class BaselineResNetBlock(torch.nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, kernel_size: int, stride: int = 1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        
        self.bn1 = enn.InnerBatchNorm(in_type)
        self.relu1 = enn.ReLU(in_type)
        # Use the standard, dense R2Conv layer
        self.conv1 = enn.R2Conv(
            in_type, out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu2 = enn.ReLU(out_type)
        # Use the standard, dense R2Conv layer
        self.conv2 = enn.R2Conv(
            out_type, out_type,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
        self.shortcut = None
        if stride != 1 or in_type.size != out_type.size:
            self.shortcut = enn.R2Conv(
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

# --- Baseline ResNet44 Architecture ---
class Baseline_ResNet44(torch.nn.Module):
    def __init__(self, n: int = 7, num_classes: int = 43, in_channels: int = 3, group: str = "P4M", widths: list = [11, 23, 45]):
        super().__init__()
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
        self.conv1 = enn.R2Conv(
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

    def _make_stage(self, in_type: enn.FieldType, width: int, num_blocks: int, stride: int):
        out_type = enn.FieldType(self.r2_act, width * [self.r2_act.regular_repr])
        layers = [BaselineResNetBlock(in_type, out_type, 3, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BaselineResNetBlock(out_type, out_type, 3, stride=1))
        return torch.nn.Sequential(*layers)

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


# --- Baseline P4CNN for MNIST ---
class Baseline_P4CNN_MNIST(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(Baseline_P4CNN_MNIST, self).__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        channels = 10
        c_out = enn.FieldType(self.r2_act, channels * [self.r2_act.regular_repr])
        
        self.block1 = nn.Sequential(
            enn.R2Conv(self.in_type, c_out, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out)
        )
        self.block2 = nn.Sequential(
            enn.R2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out),
            enn.PointwiseMaxPool(c_out, kernel_size=2, stride=2)
        )
        self.block3 = enn.R2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block4 = enn.R2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block5 = enn.R2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block6 = enn.R2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn_relu = nn.Sequential(enn.InnerBatchNorm(c_out), enn.ReLU(c_out))
        self.block7 = enn.R2Conv(c_out, c_out, kernel_size=4, padding=0, bias=False)
        self.gpool = enn.GroupPooling(c_out)
        self.fc = nn.Linear(channels * 11 * 11, num_classes)

    def forward(self, x: torch.Tensor):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.bn_relu(self.block3(x))
        x = self.bn_relu(self.block4(x))
        x = self.bn_relu(self.block5(x))
        x = self.bn_relu(self.block6(x))
        x = self.bn_relu(self.block7(x))
        x = self.gpool(x)
        x = x.tensor.flatten(start_dim=1)
        x = self.fc(x)
        return x
