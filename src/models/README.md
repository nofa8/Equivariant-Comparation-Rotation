# Model Architectures

This directory contains the three model families used in the comparative study.

## Models

### `model_s.py` — ModelS (Custom CNN)

A 4-block VGG-style convolutional neural network trained from scratch. Each block follows the pattern: `Conv2d(3×3) → BatchNorm2d → ReLU → MaxPool2d(2×2)`. Channel progression: 3→32→64→128→256. The classifier uses `AdaptiveAvgPool2d(4,4)` followed by a fully-connected head (4096→512→10) with 50% dropout.

Used in experiments: **S-1**, **S-3**.

### `resnet.py` — ResNet-18 (Transfer Learning)

A torchvision ResNet-18 backbone with a custom classification head (512→ReLU→Dropout(0.4)→10). Supports two modes:
- **Feature extraction** (`freeze=True`): All backbone weights are frozen; only the new head is trained.
- **Fine-tuning** (`freeze=False`): The entire model is trained end-to-end with a lower learning rate.

Used in experiments: **T-FE-1**, **T-FE-2**, **T-FT-1**, **T-FT-2**.

### `equivariant.py` — EquivariantCNN (C8-Equivariant CNN)

A rotation-equivariant CNN built with [e2cnn](https://github.com/QUVA-Lab/e2cnn). Uses the cyclic group C8 (8 discrete 45° rotations) as the symmetry group. Key design choices:
- **Regular representations** for all intermediate layers (preserves full group information).
- **PointwiseAvgPool** instead of MaxPool (MaxPool is not equivariant).
- **InnerBatchNorm** instead of standard BatchNorm (normalizes within each fiber).
- **GroupPooling** at the end to yield rotation-invariant features before classification.
- **No bias** in R2Conv layers (bias can break equivariance for non-trivial representations).

Used in experiments: **Eq-1**, **Eq-2**.

## Factory

`factory.py` provides `build_model(name, ...)` which maps experiment config strings (`model_s`, `model_t`, `model_eq`) to instantiated models.

## Notes

- `cnn.py` contains `SimpleCNN`, an early prototype identical to `ModelS`. It is unused and kept for reference.
