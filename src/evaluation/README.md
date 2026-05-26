# Evaluation

Post-training evaluation metrics and visualization.

## Modules

### `classification.py` — Standard Classification Metrics

`evaluate_classification(model, loader, device, loss_fn)` computes:
- Per-class and overall accuracy.
- Macro-averaged precision, recall, and F1 (via scikit-learn's `classification_report`).
- Mean loss over the dataset.

### `confusion.py` — Confusion Matrix

`plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path)` generates a publication-quality confusion matrix heatmap using seaborn and saves it at 300 DPI.

### `rotation.py` — Rotation Sensitivity Analysis

The core experiment evaluation module:

- `evaluate_rotation_sensitivity(model, config, transform, device)` — Tests the model's accuracy on each of the 12 viewpoints independently (0°, 30°, ..., 330°) and computes the **Rotation AUC** metric using the trapezoidal rule, normalized to [0, 1].
- `plot_and_save_rotation_curve(rotation_accuracies, auc, model_name, save_path)` — Plots the rotation sensitivity curve with training domain highlighting.

**Rotation AUC** measures how flat the accuracy curve is across all viewpoints. A score near 1.0 indicates rotation robustness; a score that drops sharply beyond the training domain (0°–150°) indicates view dependence.
