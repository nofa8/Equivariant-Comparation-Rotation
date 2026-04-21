from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def get_writer(run_name: str):
    log_dir = Path("outputs/logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)