from src.models.model_s import ModelS

def build_model(name, num_classes=10, freeze=False, pretrained=True):
    if name == "model_s":
        return ModelS(num_classes)
    elif name == "model_t":
        from src.models.resnet import get_resnet18
        return get_resnet18(num_classes=num_classes, pretrained=pretrained, freeze=freeze)
    elif name == "model_eq":
        from src.models.equivariant import EquivariantCNN
        return EquivariantCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {name}")