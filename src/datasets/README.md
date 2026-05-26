# Datasets

Dataset loading and preprocessing for ModelNet10 multi-view renders.

## Modules

### `modelnet_dataset.py` — ModelNetDataset

Custom PyTorch `Dataset` for loading multi-view PNG renders of 3D ModelNet objects. Each 3D model is rendered from 12 viewpoints (0°, 30°, ..., 330°). The dataset supports:
- **Model-level train/val/test splitting** (prevents data leakage across views of the same object).
- **Angle filtering** via `allowed_angles` (e.g., train on 6 views, test on all 12).
- **Automatic split persistence** to JSON for reproducibility.

### `splits.py` — Split Generation

Handles deterministic splitting of model directories into train/val/test partitions using numpy's `default_rng` with a fixed seed. Splits are saved to and loaded from `data/processed/splits.json`.

### `transforms.py` — Data Augmentation

Provides `build_transforms(img_size, augment)`:
- **Base**: Resize to 224×224 + ToTensor.
- **Augmented**: Adds RandomHorizontalFlip, RandomRotation(30°), and ColorJitter.

### `factory.py` — DataLoader Factory

`build_dataloaders(config, augment)` constructs train and validation DataLoaders with the canonical training angles (0°, 30°, 60°, 90°, 120°, 150°).

### `dummy_dataset.py`

Synthetic dataset of random tensors for pipeline testing without real data.
