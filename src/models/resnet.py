import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18(num_classes, pretrained=True, freeze=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model