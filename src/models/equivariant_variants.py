import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantCNNVariant(nn.Module):
    """
    Parameterized C_N-Equivariant CNN for architecture exploration studies.
    
    Supports variable:
      - Group order N (e.g. 4, 8, 16 for C4, C8, C16 cyclic groups)
      - Channel scaling via fiber counts per block
      - Optional deep classifier head to study classifier bottleneck effects
      - Optional GroupPooling ablation (skip_gpool) to study invariance effects
    
    Predefined configurations:
      Eq-Small:      N=8,  fibers=(4, 8, 16, 32)   — matches original EquivariantCNN
      Eq-Medium:     N=8,  fibers=(8, 16, 32, 64)
      Eq-Large:      N=8,  fibers=(16, 32, 64, 128)
      Eq-Large-Deep: N=8,  fibers=(16, 32, 64, 128), deep_classifier=True
      C4 variant:    N=4,  fibers=(4, 8, 16, 32)
      C16 variant:   N=16, fibers=(4, 8, 16, 32)
      No-GPool:      N=8,  fibers=(4, 8, 16, 32),   skip_gpool=True
    """
    def __init__(self, num_classes=10, N=8, fibers=(4, 8, 16, 32), 
                 deep_classifier=False, skip_gpool=False):
        super().__init__()
        
        self.N = N
        self.fibers = fibers
        self.deep_classifier = deep_classifier
        self.skip_gpool = skip_gpool
        
        # Define C_N rotation group (cyclic group of order N)
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        
        # Input: 3 RGB channels as trivial representations
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * 3)
        
        # Intermediate field types using regular representations
        # Each regular repr of C_N has dimension N, so effective channels = fibers[i] * N
        type1 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * fibers[0])
        type2 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * fibers[1])
        type3 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * fibers[2])
        type4 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * fibers[3])
        
        # Block 1
        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, type1, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type1),
            enn.ReLU(type1, inplace=True),
            enn.PointwiseAvgPool(type1, kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = enn.SequentialModule(
            enn.R2Conv(type1, type2, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type2),
            enn.ReLU(type2, inplace=True),
            enn.PointwiseAvgPool(type2, kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = enn.SequentialModule(
            enn.R2Conv(type2, type3, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type3),
            enn.ReLU(type3, inplace=True),
            enn.PointwiseAvgPool(type3, kernel_size=2, stride=2)
        )
        
        # Block 4
        self.block4 = enn.SequentialModule(
            enn.R2Conv(type3, type4, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(type4),
            enn.ReLU(type4, inplace=True),
            enn.PointwiseAvgPool(type4, kernel_size=2, stride=2)
        )
        
        if skip_gpool:
            # Ablation: skip GroupPooling, use raw equivariant features
            # This deliberately breaks rotation invariance to test whether
            # enforced invariance discards discriminative orientation information
            self.gpool = None
            classifier_in = fibers[3] * N  # Full equivariant channels (e.g. 32*8=256)
        else:
            # Group pooling: pools over N orientations, yielding fibers[-1] invariant channels
            self.gpool = enn.GroupPooling(type4)
            classifier_in = fibers[3]  # Invariant channels after group pooling
        
        # Number of channels entering the classifier
        self.classifier_in = classifier_in
        
        # Global spatial average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        if deep_classifier:
            # Deeper classifier to test whether the bottleneck is in the classifier
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )
        else:
            # Minimal classifier (same as original EquivariantCNN)
            self.classifier = nn.Linear(classifier_in, num_classes)
    
    def forward(self, x):
        # Convert to GeometricTensor
        x_geo = enn.GeometricTensor(x, self.in_type)
        
        # Equivariant feature extraction
        x_geo = self.block1(x_geo)
        x_geo = self.block2(x_geo)
        x_geo = self.block3(x_geo)
        x_geo = self.block4(x_geo)
        
        if self.gpool is not None:
            # Pool over group to obtain invariant representation
            x_geo = self.gpool(x_geo)
        
        # Extract tensor, spatial pool, classify
        x_tensor = x_geo.tensor
        x_tensor = self.pool(x_tensor).flatten(1)
        out = self.classifier(x_tensor)
        return out
    
    def get_feature_info(self, x):
        """
        Diagnostic method: returns intermediate tensor shapes for bottleneck analysis.
        Traces a single input through the network and records dimensions at each stage.
        """
        info = {}
        x_geo = enn.GeometricTensor(x, self.in_type)
        info['input'] = {'shape': tuple(x.shape), 'channels': x.shape[1]}
        
        x_geo = self.block1(x_geo)
        info['after_block1'] = {
            'shape': tuple(x_geo.tensor.shape),
            'equivariant_channels': x_geo.tensor.shape[1],
            'fibers': self.fibers[0],
            'spatial': tuple(x_geo.tensor.shape[2:])
        }
        
        x_geo = self.block2(x_geo)
        info['after_block2'] = {
            'shape': tuple(x_geo.tensor.shape),
            'equivariant_channels': x_geo.tensor.shape[1],
            'fibers': self.fibers[1],
            'spatial': tuple(x_geo.tensor.shape[2:])
        }
        
        x_geo = self.block3(x_geo)
        info['after_block3'] = {
            'shape': tuple(x_geo.tensor.shape),
            'equivariant_channels': x_geo.tensor.shape[1],
            'fibers': self.fibers[2],
            'spatial': tuple(x_geo.tensor.shape[2:])
        }
        
        x_geo = self.block4(x_geo)
        info['before_gpool'] = {
            'shape': tuple(x_geo.tensor.shape),
            'equivariant_channels': x_geo.tensor.shape[1],
            'fibers': self.fibers[3],
            'spatial': tuple(x_geo.tensor.shape[2:])
        }
        
        if self.gpool is not None:
            x_geo = self.gpool(x_geo)
            info['after_gpool'] = {
                'shape': tuple(x_geo.tensor.shape),
                'invariant_channels': x_geo.tensor.shape[1],
                'spatial': tuple(x_geo.tensor.shape[2:])
            }
        else:
            info['after_gpool'] = {
                'shape': tuple(x_geo.tensor.shape),
                'invariant_channels': x_geo.tensor.shape[1],
                'spatial': tuple(x_geo.tensor.shape[2:]),
                'note': 'GroupPooling SKIPPED — raw equivariant features'
            }
        
        x_tensor = self.pool(x_geo.tensor).flatten(1)
        info['after_spatial_pool'] = {
            'shape': tuple(x_tensor.shape),
            'features': x_tensor.shape[1]
        }
        
        return info
