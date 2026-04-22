from src.models.model_s import ModelS

# later extend with ModelT, ModelEq

def build_model(name, num_classes=10):
    if name == "model_s":
        return ModelS(num_classes)
    else:
        raise ValueError(f"Unknown model {name}")