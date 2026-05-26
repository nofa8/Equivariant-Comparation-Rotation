# Training

Training loop, loss functions, and optimizer configuration.

## Modules

### `trainer.py` — Trainer

Core training loop with:
- Per-epoch train/validation passes with accuracy and loss tracking.
- TensorBoard logging (loss and accuracy curves).
- Best-model checkpointing based on validation accuracy.
- Progress bars via `tqdm`.

### `losses.py` — Loss Functions

`build_loss(name)` maps config strings to PyTorch loss functions:
- `"ce"` → `CrossEntropyLoss`
- `"label_smoothing"` → `CrossEntropyLoss(label_smoothing=0.1)`

### `optimizers.py` — Optimizer Configuration

`build_optimizer(model, config)` constructs optimizers from config:
- `"sgd"` → SGD with momentum=0.9 and weight_decay=1e-4.
- Default → AdamW with weight_decay=0.01.
- Automatically filters to only `requires_grad=True` parameters (relevant for frozen ResNet experiments).
