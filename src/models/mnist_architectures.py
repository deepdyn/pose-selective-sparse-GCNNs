import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

# Import the sparse convolution block from your existing models
# This assumes the file is in the same directory
from .pose_gcnn import SparseR2Conv

class PoseSelective_P4CNN_MNIST(nn.Module):
    """
    Implements the 7-layer P4CNN architecture from Cohen and Welling (2016)
    for the MNIST/Rot-MNIST datasets, but with the Pose-Selective-Sparse-GCNNs
    logic integrated.

    Reference: Section 8.1 of "Group Equivariant Convolutional Networks"
    (arXiv:1602.07576v3)
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(PoseSelective_P4CNN_MNIST, self).__init__()

        # The group of 90-degree rotations (p4)
        self.r2_act = gspaces.Rot2dOnR2(N=4)

        # The input images are trivial fields (no orientation)
        self.in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # The paper uses 20 channels in the baseline, and divides by sqrt(4)=2 for the P4CNN
        # to keep the number of parameters roughly constant. So we use 10 channels.
        channels = 10
        
        # Layer 1: Lifts the image to a p4 feature map
        # Input: 1 channel (image) -> Output: 10 channels (p4 regular representation)
        c_out = enn.FieldType(self.r2_act, channels * [self.r2_act.regular_repr])
        self.block1 = nn.Sequential(
            SparseR2Conv(self.in_type, c_out, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out)
        )

        # Layer 2: p4 -> p4 convolution
        self.block2 = nn.Sequential(
            SparseR2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out),
            # [cite_start]Max pooling after layer 2, as specified in the paper [cite: 3341]
            enn.PointwiseMaxPool(c_out, kernel_size=2, stride=2)
        )
        # The feature map size is now 14x14

        # Layers 3, 4, 5, 6: Four more p4 -> p4 convolutional layers
        self.block3 = SparseR2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block4 = SparseR2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block5 = SparseR2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.block6 = SparseR2Conv(c_out, c_out, kernel_size=3, padding=1, bias=False)
        
        # Intermediate batch norm and ReLU layers
        self.bn_relu = nn.Sequential(
            enn.InnerBatchNorm(c_out),
            enn.ReLU(c_out)
        )

        # [cite_start]Layer 7: Final convolutional layer, using a 4x4 kernel [cite: 3341]
        self.block7 = SparseR2Conv(c_out, c_out, kernel_size=4, padding=0, bias=False)
        # The feature map size is now 11x11

        # Final classification head
        # [cite_start]Pool over the group dimension (rotations) to achieve rotation invariance [cite: 3345]
        self.gpool = enn.GroupPooling(c_out)
        self.fc = nn.Linear(channels * 11 * 11, num_classes)

    def forward(self, x: torch.Tensor):
        # Wrap the input tensor
        x = enn.GeometricTensor(x, self.in_type)

        # Pass through the network
        x = self.block1(x)
        x = self.block2(x)
        x = self.bn_relu(self.block3(x))
        x = self.bn_relu(self.block4(x))
        x = self.bn_relu(self.block5(x))
        x = self.bn_relu(self.block6(x))
        x = self.bn_relu(self.block7(x))

        # Invariant pooling and classification
        x = self.gpool(x)
        x = x.tensor.flatten(start_dim=1)
        x = self.fc(x)

        return x
