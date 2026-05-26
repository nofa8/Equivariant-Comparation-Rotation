# Rotation Robustness in 3D Object Classification

A comparative study of three deep learning approaches to rotation-robust image classification, evaluated on multi-view renders of the ModelNet10 dataset.

## Research Question

> *How do different deep learning strategies — training from scratch, transfer learning, and equivariant architectures — compare in terms of rotation robustness for 3D object classification from 2D multi-view renders?*

## Models Under Study

| Model | Type | Key Idea |
|-------|------|----------|
| **Model S** (Custom CNN) | Trained from scratch | 4-block VGG-style CNN (Conv→BN→ReLU→MaxPool) |
| **Model T** (ResNet-18) | Transfer learning | ImageNet-pretrained ResNet-18 with custom classifier head |
| **Model Eq** (C8-Equivariant CNN) | Equivariant architecture | [e2cnn](https://github.com/QUVA-Lab/e2cnn)-based CNN with C8 cyclic group symmetry |

## Experiment Matrix

| Experiment | Model | Strategy | Augmentation | LR |
|-----------|-------|----------|-------------|-----|
| S-1 | Model S | From scratch | ❌ | 1e-3 |
| S-3 | Model S | From scratch | ✅ | 1e-3 |
| T-FE-1 | ResNet-18 | Feature extraction (frozen backbone) | ❌ | 1e-3 |
| T-FE-2 | ResNet-18 | Feature extraction (frozen backbone) | ✅ | 1e-3 |
| T-FT-1 | ResNet-18 | Fine-tuning (full model) | ❌ | 1e-4 |
| T-FT-2 | ResNet-18 | Fine-tuning (full model) | ✅ | 1e-4 |
| Eq-1 | Equivariant CNN | C8-equivariant layers | ❌ | 1e-3 |
| Eq-2 | Equivariant CNN | C8-equivariant layers | ✅ | 1e-3 |

## Key Results

| Model | Overall Accuracy | Macro F1 | Rotation AUC |
|-------|-----------------|----------|-------------|
| S-1 | 0.4620 | 0.3650 | 0.4277 |
| S-3 | 0.8846 | 0.8349 | 0.8831 |
| T-FE-1 | 0.8673 | 0.8208 | 0.8661 |
| T-FE-2 | 0.8325 | 0.7705 | 0.8326 |
| T-FT-1 | 0.9128 | 0.8841 | 0.9109 |
| **T-FT-2** | **0.9258** | **0.8939** | **0.9253** |
| Eq-1 | 0.7954 | 0.7455 | 0.7941 |
| Eq-2 | 0.7907 | 0.7126 | 0.7905 |

**Rotation AUC** is a normalized area-under-the-curve metric computed over per-angle accuracy across all 12 viewpoints (0°–330°). A score of 1.0 means perfect accuracy at every angle; a low score indicates view-dependent performance.

## Project Structure

```
DL_Project/
├── configs/                    # Experiment configurations (YAML)
│   └── experiments.yaml
├── data/
│   ├── raw/ModelNet10_views/   # Multi-view PNG renders (gitignored)
│   └── processed/splits.json  # Train/val/test splits (model-level)
├── src/                        # Core library
│   ├── models/                 # Model architectures
│   ├── datasets/               # Dataset loading, transforms, splits
│   ├── training/               # Trainer, loss functions, optimizers
│   ├── evaluation/             # Classification metrics, confusion matrix, rotation analysis
│   └── utils/                  # Checkpointing, logging, seed management
├── scripts/                    # Runnable experiment scripts
│   ├── run_experiment.py       # Main training entry point
│   ├── evaluate_checkpoint.py  # Full evaluation pipeline
│   ├── generate_comparison.py  # Cross-experiment comparative analysis
│   ├── verify_equivariance.py  # Mathematical equivariance verification
│   ├── check_data.py           # Data leakage verification
│   └── train.py                # Quick prototyping script (CIFAR-10)
├── notebooks/                  # Jupyter notebooks for interactive runs
├── outputs/                    # Experiment outputs (checkpoints, logs, figures)
├── main.py                     # Scaffolding entry point (dev only)
└── pyproject.toml              # Project dependencies (managed by uv)
```

## Quick Start

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended, not required)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd DL_Project

# Install dependencies
uv sync
```

### Running an Experiment

```bash
# Train a single experiment
PYTHONPATH=. uv run python scripts/run_experiment.py --exp S-1

# Evaluate a trained checkpoint
PYTHONPATH=. uv run python scripts/evaluate_checkpoint.py --ckpt outputs/S-1/checkpoints/best.pt

# Generate comparative analysis
PYTHONPATH=. uv run python scripts/generate_comparison.py --exps S-1,S-3,T-FT-1,T-FT-2,Eq-1,Eq-2

# Verify equivariance properties of the C8 model
PYTHONPATH=. uv run python scripts/verify_equivariance.py
```

### Running All Experiments

Use the Jupyter notebook for a complete sequential run of all 8 experiments with inline visualizations:

```bash
uv run jupyter notebook notebooks/run_all_experiments.ipynb
```

## Evaluation Protocol

1. **Training views**: 6 canonical angles (0°, 30°, 60°, 90°, 120°, 150°)
2. **Test views**: All 12 angles (0°–330°), including 6 unseen viewpoints (180°–330°)
3. **Split strategy**: Model-level (3D object) splitting to prevent data leakage between views
4. **Metrics**: Overall accuracy, macro-averaged F1/precision/recall, confusion matrix, rotation sensitivity curve, and rotation AUC

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Core deep learning framework |
| `e2cnn` | Equivariant neural network layers |
| `scikit-learn` | Classification metrics |
| `matplotlib`, `seaborn` | Visualization |
| `tensorboard` | Training monitoring |
| `tqdm` | Progress bars |
| `pyyaml` | Experiment configuration |

## Reproducibility

All experiments use:
- **Seed**: 42 (Python `random`, NumPy, PyTorch, CUDA)
- **Deterministic mode**: `torch.backends.cudnn.deterministic = True`
- **Config archival**: Each experiment saves its full config to `outputs/<exp>/config.yaml`
