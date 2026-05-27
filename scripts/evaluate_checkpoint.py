import argparse
import json
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.models.factory import build_model
from src.datasets.modelnet_dataset import ModelNetDataset
from src.datasets.transforms import build_transforms
from src.training.losses import build_loss
from src.utils.checkpoint import load_checkpoint

# Decoupled Evaluation package imports
from src.evaluation import (
    evaluate_classification,
    plot_and_save_confusion_matrix,
    evaluate_rotation_sensitivity,
    plot_and_save_rotation_curve,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the best.pt checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # Resolve paths
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # The experiment directory is the parent of the checkpoints directory
    # e.g., outputs/S-1/checkpoints/best.pt -> outputs/S-1
    exp_dir = ckpt_path.parent.parent
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found at: {config_path}")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"📖 Loaded experiment config from: {config_path}")

    # Setup directories
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1. Load Checkpoint Metadata & Weights
    # -------------------------------
    print("🏗️ Building model...")
    num_classes = config.get("num_classes", 10)
    model = build_model(config["model"], num_classes=num_classes)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"📈 Loaded Rich Checkpoint Metadata | Best Val Accuracy: {ckpt.get('best_val_acc', 0.0):.4f} | Saved at Epoch: {ckpt.get('epoch', 0)}")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("💾 Loaded Legacy Checkpoint (Raw State Dict).")
        model.load_state_dict(ckpt)
        
    model = model.to(device)
    model.eval()
    print("✅ Model loaded and weights restored successfully!")

    # -------------------------------
    # 2. Build Dataset & DataLoader
    # -------------------------------
    val_transform = build_transforms(img_size=224, augment=False, normalize=config.get("normalize", False))
    
    # We use all angles [0..11] representing 0° to 330° for full evaluation
    full_eval_angles = list(range(12))

    dataset = ModelNetDataset(
        root=config["data_root"],
        split=args.split,
        splits_file=config["splits_file"],
        allowed_angles=full_eval_angles,
        transform=val_transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"📋 Loaded '{args.split}' split with {len(dataset)} images (all 12 views included).")

    # -------------------------------
    # 3. Standard Split Evaluation
    # -------------------------------
    loss_fn = build_loss(config["loss"])
    print("🕵️ Running forward pass...")
    report, mean_loss, all_preds, all_labels = evaluate_classification(
        model, loader, device, loss_fn=loss_fn
    )

    overall_acc = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    macro_precision = report.get("macro avg", {}).get("precision", 0.0)
    macro_recall = report.get("macro avg", {}).get("recall", 0.0)

    print(f"\n📊 --- Overall Split [{args.split.upper()}] Metrics ---")
    print(f"Loss:      {mean_loss:.4f}" if mean_loss is not None else "Loss:      N/A")
    print(f"Accuracy:  {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}\n")

    # Generate and Save Confusion Matrix Plot
    class_names = sorted(list(dataset.class_to_idx.keys()))
    cm_plot_path = figures_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        save_path=cm_plot_path,
        title=f"Confusion Matrix ({config['model'].upper()} - {args.split.upper()} split)",
    )
    print(f"🖼️ Saved Confusion Matrix plot to: {cm_plot_path}")

    # -------------------------------
    # 4. Rotation Sensitivity Curve & AUC
    # -------------------------------
    rotation_accs, rotation_auc = evaluate_rotation_sensitivity(
        model=model,
        config=config,
        transform=val_transform,
        device=device,
        split=args.split,
        batch_size=args.batch_size,
    )

    # Plot and save rotation curve
    rot_plot_path = figures_dir / "rotation_sensitivity.png"
    plot_and_save_rotation_curve(
        rotation_accuracies=rotation_accs,
        auc=rotation_auc,
        model_name=config["model"],
        save_path=rot_plot_path,
    )
    print(f"🖼️ Saved Rotation Sensitivity plot to: {rot_plot_path}")

    # -------------------------------
    # 5. Save Results to JSON
    # -------------------------------
    results = {
        "experiment": args.ckpt,
        "split": args.split,
        "overall_loss": mean_loss,
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "rotation_auc": rotation_auc,
        "rotation_accuracies": rotation_accs,
    }
    
    results_path = exp_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Saved raw evaluation results JSON to: {results_path}\n")

if __name__ == "__main__":
    main()
