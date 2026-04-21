from torchvision.models import resnet18

def get_resnet18(num_classes, pretrained=True, freeze=False):
    model = resnet18(pretrained=pretrained)

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