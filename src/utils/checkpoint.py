import torch
from pathlib import Path

def save_checkpoint(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model