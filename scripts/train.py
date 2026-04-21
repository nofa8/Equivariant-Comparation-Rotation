import torch
from torch.utils.data import DataLoader, random_split

from src.models.model_s import ModelS
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizers import build_optimizer
from src.utils.seed import set_seed
from src.utils.logger import get_writer

# TEMP: replace later with ModelNet dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = ModelS(num_classes=10)

    config = {"optimizer": "adamw", "loss": "ce"}

    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config["loss"])

    writer = get_writer("baseline")

    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, loss_fn, device, writer
    )

    trainer.fit(epochs=5)


if __name__ == "__main__":
    main()