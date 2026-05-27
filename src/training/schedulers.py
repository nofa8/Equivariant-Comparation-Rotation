import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(optimizer, config):
    """
    Factory to build a learning rate scheduler based on config parameters.
    """
    scheduler_type = config.get("scheduler")
    if not scheduler_type:
        return None

    scheduler_type = scheduler_type.lower()
    epochs = config.get("epochs", 30)

    if scheduler_type == "cosine":
        # Decay to a small fraction of the initial learning rate
        eta_min = config.get("scheduler_eta_min", 1e-6)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        
    elif scheduler_type == "step":
        step_size = config.get("scheduler_step_size", epochs // 3)
        gamma = config.get("scheduler_gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
