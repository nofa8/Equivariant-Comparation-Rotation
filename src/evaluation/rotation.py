import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets.modelnet_dataset import ModelNetDataset

def evaluate_rotation_sensitivity(model, config, transform, device, split="test", batch_size=64):
    """
    Evaluates the model's accuracy on each view angle individually and calculates the Rotation AUC score.
    
    Returns:
        rotation_accuracies (dict): Mapping from angle in degrees (e.g. 0, 30, ...) to accuracy.
        auc (float): Single-number robustness score (Rotation AUC, normalized between 0 and 1).
    """
    model.eval()
    rotation_accuracies = {}
    
    print("\n🔄 Running Rotation Sensitivity Analysis...")
    for view_idx in range(12):
        angle_deg = view_idx * 30
        
        # Build single-view dataset
        single_view_dataset = ModelNetDataset(
            root=config["data_root"],
            split=split,
            splits_file=config["splits_file"],
            allowed_angles=[view_idx],
            transform=transform,
        )
        
        if len(single_view_dataset) == 0:
            print(f"⚠️ No samples found for view {view_idx} ({angle_deg}°). Skipping.")
            continue

        single_view_loader = DataLoader(
            single_view_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in single_view_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total if total > 0 else 0.0
        rotation_accuracies[angle_deg] = acc
        print(f"   Angle {angle_deg:3d}° (View {view_idx:2d}): Accuracy = {acc:.4f} ({correct}/{total})")

    # Compute Normalized Rotation AUC using Trapezoidal rule
    angles = sorted(list(rotation_accuracies.keys()))
    accuracies = [rotation_accuracies[a] for a in angles]
    
    if len(angles) > 1 and (angles[-1] - angles[0]) > 0:
        if hasattr(np, "trapezoid"):
            auc = np.trapezoid(accuracies, angles) / (angles[-1] - angles[0])
        else:
            auc = np.trapz(accuracies, angles) / (angles[-1] - angles[0])
    else:
        auc = 0.0
        
    print(f"📊 Calculated Rotation Robustness AUC: {auc:.4f}\n")
    return rotation_accuracies, auc

def plot_and_save_rotation_curve(rotation_accuracies, auc, model_name, save_path):
    """
    Plots the rotation sensitivity curve and saves it.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    angles = sorted(list(rotation_accuracies.keys()))
    accuracies = [rotation_accuracies[a] for a in angles]

    plt.figure(figsize=(9, 5))
    plt.plot(
        angles, 
        accuracies, 
        marker="o", 
        linewidth=2.5, 
        color="#1f77b4", 
        label=f"Model Accuracy (AUC: {auc:.4f})"
    )
    
    # Highlight training domain if relevant (e.g. 0° to 150°)
    plt.axvspan(0, 150, color="green", alpha=0.1, label="Training Rotation Domain (0°-150°)")
    
    plt.title(f"Rotation Sensitivity Curve ({model_name.upper()})")
    plt.xlabel("Rotation Angle (Degrees)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.xticks(angles)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
