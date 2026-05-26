import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantCNN(nn.Module):
    """
    C8-Equivariant Convolutional Neural Network (Discrete SO(2)-equivariant approximation).
    Designed for 2D rotation-robust image classification on ModelNet10 multi-view render datasets.
    
    This model utilizes a C8 cyclic group representation (8 discrete 45-degree rotations).
    Intermediate features are mapped to regular representations of C8 to preserve full equivariance,
    which are then pooled over the group to yield invariant descriptors before classification.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Define C8 rotation group (cyclic group of order 8)
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # Define the input field type as an attribute to avoid dynamic recreation during forward passes
        # Input features are 3 channels (RGB) which correspond to trivial representations
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*3)
        
        # Intermediate layers use regular representations. Each regular representation of C8 has size 8.
        # Effective intermediate channels correspond to: number of regular fibers * 8
        type1 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr]*4)   # 32 channels
        type2 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr]*8)   # 64 channels
        type3 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr]*16)  # 128 channels
        type4 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr]*32)  # 256 channels
        
        # Block 1: 3 -> 32 channels
        # Using PointwiseAvgPool to preserve smooth equivariant transitions and avoid orientation instability/aliasing of MaxPool
        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, type1, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type1),
            enn.ReLU(type1, inplace=True),
            enn.PointwiseAvgPool(type1, kernel_size=2, stride=2)
        )
        
        # Block 2: 32 -> 64 channels
        self.block2 = enn.SequentialModule(
            enn.R2Conv(type1, type2, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type2),
            enn.ReLU(type2, inplace=True),
            enn.PointwiseAvgPool(type2, kernel_size=2, stride=2)
        )
        
        # Block 3: 64 -> 128 channels
        self.block3 = enn.SequentialModule(
            enn.R2Conv(type2, type3, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type3),
            enn.ReLU(type3, inplace=True),
            enn.PointwiseAvgPool(type3, kernel_size=2, stride=2)
        )
        
        # Block 4: 128 -> 256 channels
        self.block4 = enn.SequentialModule(
            enn.R2Conv(type3, type4, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type4),
            enn.ReLU(type4, inplace=True),
            enn.PointwiseAvgPool(type4, kernel_size=2, stride=2)
        )
        
        # Group pooling: pools over the 8 cyclic group orientations, yielding 32 trivial invariant fibers.
        self.gpool = enn.GroupPooling(type4)
        
        # Global Average Spatial Pooling (1x1) to preserve full mathematical equivariance
        # and significantly reduce parameter footprint.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Invariant Classifier (32 channels -> 10 classes)
        self.classifier = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # Convert standard PyTorch tensor to GeometricTensor in the trivial input type
        x_geo = enn.GeometricTensor(x, self.in_type)
        
        # Run C8-equivariant convolutional blocks
        x_geo = self.block1(x_geo)
        x_geo = self.block2(x_geo)
        x_geo = self.block3(x_geo)
        x_geo = self.block4(x_geo)
        
        # Pool over the rotation group to obtain invariant representation
        x_geo = self.gpool(x_geo)
        
        # Extract standard PyTorch tensor
        x_tensor = x_geo.tensor
        
        # Global spatial average pooling and lightweight linear classification
        x_tensor = self.pool(x_tensor).flatten(1)
        out = self.classifier(x_tensor)
        return out