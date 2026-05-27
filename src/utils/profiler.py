import torch
from torchinfo import summary

def profile_model_efficiency(model, input_size=(1, 3, 224, 224)):
    """
    Profiles a PyTorch model's parameter count and FLOPs (estimated via Multi-Adds / MACs) using torchinfo.
    Returns a dict with total parameters, trainable parameters, and MACs (FLOPs).
    """
    try:
        model_stats = summary(model, input_size=input_size, verbose=0)
        total_params = model_stats.total_params
        trainable_params = model_stats.trainable_params
        macs = model_stats.total_mult_adds
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": macs,  # MACs are commonly used as an estimate for FLOPs
        }
    except Exception as e:
        print(f"⚠️ Profiler warning: {e}")
        # Fallback to analytical counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": -1,
        }
