# Scripts

Executable entry points for the experiment pipeline.

## Training

### `run_experiment.py`

Main training script. Reads `configs/experiments.yaml`, builds the model/data/optimizer from the config, trains for 30 epochs, and saves the best checkpoint.

```bash
PYTHONPATH=. uv run python scripts/run_experiment.py --exp S-1
```

### `train.py`

Quick prototyping script using CIFAR-10 instead of ModelNet. Useful for testing the training pipeline without the full dataset.

## Evaluation

### `evaluate_checkpoint.py`

Full evaluation pipeline for a trained checkpoint. Produces:
- Classification metrics (accuracy, F1, precision, recall).
- Confusion matrix plot.
- Rotation sensitivity curve and AUC.
- JSON results file.

```bash
PYTHONPATH=. uv run python scripts/evaluate_checkpoint.py --ckpt outputs/S-1/checkpoints/best.pt
```

### `generate_comparison.py`

Cross-experiment comparative analysis. Reads evaluation results from multiple experiments and generates:
- Combined rotation sensitivity plot.
- CSV summary table.

```bash
PYTHONPATH=. uv run python scripts/generate_comparison.py --exps S-1,S-3,T-FT-1,T-FT-2,Eq-1,Eq-2
```

## Verification

### `verify_equivariance.py`

Mathematical verification that the C8-equivariant model satisfies the equivariance property: `f(g·x) = g·f(x)` for all group elements `g ∈ C8`. Tests grid-aligned rotations (exact) and interpolated rotations (approximate).

### `check_data.py`

Verifies that train/test splits have zero model-level overlap (no data leakage).

## Notebook Support

### `update_notebook.py`

Programmatically updates specific cells in the Jupyter notebook to match the latest experiment list.
