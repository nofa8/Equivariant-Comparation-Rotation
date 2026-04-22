import argparse
import yaml
import torch

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
    # Model
    # -------------------------------
    model = build_model(config["model"])

    # -------------------------------
    # Training components
    # -------------------------------
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config["loss"])

    writer = get_writer(args.exp)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        writer,
    )

    # -------------------------------
    # Train
    # -------------------------------
    best_acc = trainer.fit(config["epochs"])

    print(f"✅ Finished {args.exp} | Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()