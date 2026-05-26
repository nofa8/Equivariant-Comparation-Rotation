# Utilities

Shared utility modules for reproducibility, logging, and model persistence.

## Modules

### `seed.py` — Reproducibility

`set_seed(seed)` sets seeds across all random sources (Python `random`, NumPy, PyTorch CPU/CUDA) and enables deterministic cuDNN behavior.

### `logger.py` — TensorBoard Logging

`get_writer(run_name)` creates a TensorBoard `SummaryWriter` under `outputs/logs/<run_name>/`.

### `checkpoint.py` — Model Checkpointing

- `save_checkpoint(model, path)` — Saves raw model `state_dict`.
- `load_checkpoint(model, path, device)` — Loads either a raw `state_dict` or a rich checkpoint dict containing `model_state_dict`, `epoch`, and `best_val_acc`.
