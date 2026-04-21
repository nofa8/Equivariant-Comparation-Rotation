from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=8)  # C8 group

        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*3)

        self.block1 = enn.R2Conv(in_type, ...)

        # continue building equivariant blocks...

        self.gpool = enn.GroupPooling(...)

        self.fc = torch.nn.Linear(..., num_classes)

    def forward(self, x):
        # wrap tensor into GeometricTensor
        pass