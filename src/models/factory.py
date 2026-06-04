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
    elif name == "model_eq_medium":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=8, fibers=(8, 16, 32, 64))
    elif name == "model_eq_large":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=8, fibers=(16, 32, 64, 128))
    elif name == "model_eq_large_deep":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=8, fibers=(16, 32, 64, 128), deep_classifier=True)
    elif name == "model_eq_c4":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=4, fibers=(4, 8, 16, 32))
    elif name == "model_eq_c16":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=16, fibers=(4, 8, 16, 32))
    elif name == "model_eq_no_gpool":
        from src.models.equivariant_variants import EquivariantCNNVariant
        return EquivariantCNNVariant(num_classes=num_classes, N=8, fibers=(4, 8, 16, 32), skip_gpool=True)
    else:
        raise ValueError(f"Unknown model {name}")