import re
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from src.datasets.splits import generate_splits, save_splits, load_splits

ANGLE_REGEX = re.compile(r"\.(\d+)\.png")

def extract_angle(filename: str) -> int:
    """Extract angle from 'airplane_0627.10.png' -> 10"""
    match = ANGLE_REGEX.search(filename)
    if match is None:
        raise ValueError(f"Could not extract angle from {filename}")
    return int(match.group(1))


def split_by_model_id(model_dirs: List[Path], ratios, seed=42):
    rng = np.random.default_rng(seed)
    model_dirs = model_dirs.copy()
    rng.shuffle(model_dirs)

    n = len(model_dirs)
    t1 = int(ratios[0] * n)
    t2 = int((ratios[0] + ratios[1]) * n)

    return model_dirs[:t1], model_dirs[t1:t2], model_dirs[t2:]


class ModelNetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        split_ratios=(0.61, 0.19, 0.20),
        seed: int = 42,
        allowed_angles: List[int] = [],
        transform=None,
        splits_file: str = "data/processed/splits.json",
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        if allowed_angles is not None:
            self.allowed_angles = [a // 30 if a >= 12 else a for a in allowed_angles]
        else:
            self.allowed_angles = None
        self.splits_file: str = splits_file
        self.class_to_idx = self._build_class_index()
        self.samples = self._build_samples(split_ratios, seed)

    # --------------------------------------------------

    def _build_class_index(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        return {cls: i for i, cls in enumerate(classes)}

    def _get_all_model_dirs(self):
        """
        Traverse BOTH train/ and test/ folders and merge them.
        """
        all_models = []

        for cls_name, cls_idx in self.class_to_idx.items():
            class_dir = self.root / cls_name

            for split_folder in ["train", "test"]:
                split_path = class_dir / split_folder

                if not split_path.exists():
                    continue

                for model_dir in split_path.iterdir():
                    if model_dir.is_dir():
                        all_models.append((model_dir, cls_idx))

        return all_models

    def _split_models(self, split_ratios, seed):
        splits_path = Path(self.splits_file)

        all_models = self._get_all_model_dirs()

        # Convert to relative paths
        model_dirs = [m[0] for m in all_models]
        model_to_class = {m[0]: m[1] for m in all_models}

        relative_paths = [
            str(m.relative_to(self.root)) for m in model_dirs
        ]

        # ----------------------------------------
        # Case 1: splits already exist → load
        # ----------------------------------------
        if splits_path.exists():
            split_dict = load_splits(splits_path)

        # ----------------------------------------
        # Case 2: create splits → save
        # ----------------------------------------
        else:
            split_dict = generate_splits(relative_paths, split_ratios, seed)
            save_splits(split_dict, splits_path)

        # ----------------------------------------
        # Convert back to full paths
        # ----------------------------------------
        def resolve(split_list):
            return [
                (self.root / rel_path, model_to_class[self.root / rel_path])
                for rel_path in split_list
            ]

        return {
            "train": resolve(split_dict["train"]),
            "val": resolve(split_dict["val"]),
            "test": resolve(split_dict["test"]),
        }

    def _build_samples(self, split_ratios, seed):
        split_map = self._split_models(split_ratios, seed)
        selected_models = split_map[self.split]

        samples = []

        for model_dir, class_idx in selected_models:
            for img_path in model_dir.glob("*.png"):
                angle = extract_angle(img_path.name)

                if self.allowed_angles is not None:
                    if angle not in self.allowed_angles:
                        continue

                samples.append((img_path, class_idx, angle))

        return samples

    # --------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    # --------------------------------------------------
    # Debug helpers
    # --------------------------------------------------

    def get_model_ids(self):
        return set([str(p.parent) for p, _, _ in self.samples])

    def get_angle_distribution(self):
        from collections import Counter
        return Counter([angle for _, _, angle in self.samples])