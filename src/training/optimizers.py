import torch.optim as optim

def build_optimizer(model, config):
    lr = config.get("lr", 1e-3)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config["optimizer"] == "sgd":
        sgd_lr = config.get("lr", 0.01)
        opt = optim.SGD(trainable_params, lr=sgd_lr, momentum=0.9, weight_decay=1e-4)
    else:
        opt = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    return opt