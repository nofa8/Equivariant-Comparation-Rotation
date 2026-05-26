# Experiment Configuration

## `experiments.yaml`

Defines all 8 experiments in the study. Each entry specifies:

| Key | Description | Values |
|-----|-------------|--------|
| `model` | Model architecture key | `model_s`, `model_t`, `model_eq` |
| `optimizer` | Optimizer type | `adamw`, `sgd` |
| `loss` | Loss function | `ce`, `label_smoothing` |
| `augmentation` | Whether to apply data augmentation | `true`, `false` |
| `freeze` | Freeze backbone weights (ResNet only) | `true`, `false` |
| `pretrained` | Use ImageNet-pretrained weights (ResNet only) | `true`, `false` |
| `lr` | Learning rate override (default: 1e-3) | float |

Shared defaults (set in `scripts/run_experiment.py`):
- `data_root`: `data/raw/ModelNet10_views`
- `splits_file`: `data/processed/splits.json`
- `batch_size`: 64
- `epochs`: 30
