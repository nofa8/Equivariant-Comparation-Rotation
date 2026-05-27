from torch.utils.data import DataLoader
from src.datasets.modelnet_dataset import ModelNetDataset
from src.datasets.transforms import build_transforms


TRAIN_ANGLES = [0, 30, 60, 90, 120, 150]


def build_dataloaders(config, augment):
    normalize = config.get("normalize", False)
    train_transform = build_transforms(augment=augment, normalize=normalize)
    val_transform = build_transforms(augment=False, normalize=normalize)

    train_set = ModelNetDataset(
        root=config["data_root"],
        split="train",
        splits_file=config["splits_file"],
        allowed_angles=TRAIN_ANGLES,
        transform=train_transform,
    )

    val_set = ModelNetDataset(
        root=config["data_root"],
        split="val",
        splits_file=config["splits_file"],
        allowed_angles=TRAIN_ANGLES,
        transform=val_transform,
    )

    num_workers = config.get("num_workers", 4)

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader