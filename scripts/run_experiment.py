import argparse
import yaml
import torch
import os 
from pathlib import Path

from src.models.factory import build_model
from src.datasets.factory import build_dataloaders
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizers import build_optimizer
from src.utils.seed import set_seed
from src.utils.logger import get_writer


BASE_CONFIG = {
    "data_root": "data/raw/ModelNet10_views",
    "splits_file": "data/processed/splits.json",
    "batch_size": 64,
    "epochs": 30,
}


def load_experiments():
    with open("configs/experiments.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiments = load_experiments()

    if args.exp not in experiments:
        raise ValueError(f"Experiment {args.exp} not found")

    exp_cfg = experiments[args.exp]

    print(f"\n🚀 Running Experiment: {args.exp}")
    print(exp_cfg)

    config = {**BASE_CONFIG, **exp_cfg}

    # -------------------------------
    # Data
    # -------------------------------
    train_loader, val_loader = build_dataloaders(
        config, augment=config["augmentation"]
    )

    # -------------------------------
    freeze = config.get("freeze", False)
    pretrained = config.get("pretrained", True)
    model = build_model(config["model"], freeze=freeze, pretrained=pretrained)

    # Logging parameter counts for scientific documentation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameter Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Frozen Parameters:    {total_params - trainable_params:,}\n")

    # -------------------------------
    # Training components
    # -------------------------------
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config["loss"])

    # Set up experiment directory
    exp_dir = Path("outputs") / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration used for the experiment
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(exp_dir / "logs")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        writer,
        checkpoint_dir=exp_dir / "checkpoints",
    )

    # -------------------------------
    # Train
    # -------------------------------
    best_acc = trainer.fit(config["epochs"])

    print(f"✅ Finished {args.exp} | Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()