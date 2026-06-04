import torch.optim as optim

def build_optimizer(model, config):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt_type = config.get("optimizer", "adamw").lower()

    if opt_type == "sgd":
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        nesterov = config.get("nesterov", True)
        weight_decay = config.get("weight_decay", 1e-4)
        
        opt = optim.SGD(
            trainable_params, 
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            weight_decay=weight_decay
        )
    elif opt_type == "adam":
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        
        opt = optim.Adam(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )
    else:  # default to AdamW
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 0.01)
        
        opt = optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )

    return opt