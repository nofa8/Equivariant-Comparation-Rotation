import argparse
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate comparative analysis for trained models.")
    parser.add_argument("--exps", type=str, default="S-1,S-3", help="Comma-separated experiment names to compare.")
    args = parser.parse_args()

    exp_names = [e.strip() for e in args.exps.split(",")]
    print(f"📊 Generating comparison for experiments: {exp_names}")

    comp_dir = Path("outputs/comparisons")
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to hold metrics for plotting and CSV export
    all_data = {}

    for exp in exp_names:
        exp_dir = Path("outputs") / exp
        results_path = exp_dir / "evaluation_results.json"
        
        if not results_path.exists():
            print(f"⚠️ Warning: No evaluation results found for {exp} at {results_path}. Skipping.")
            continue

        with open(results_path, "r") as f:
            data = json.load(f)
            
        all_data[exp] = data
        print(f"   Loaded results for {exp}")

    if not all_data:
        print("❌ Error: No valid experiment results were loaded.")
        return

    # -------------------------------
    # 1. Generate Combined Comparison Plot
    # -------------------------------
    plt.figure(figsize=(10, 6))
    
    # Modern, Sleek color palette (custom HSL/RGB Hex values)
    colors = {
        "S-1": "#d62728",    # Vibrant Red for fragile baseline
        "S-3": "#2ca02c",    # Vibrant Green for augmented baseline
        "T-FE-1": "#1f77b4", # Steel Blue
        "T-FE-2": "#9467bd", # Muted Purple
        "T-FT-1": "#ff7f0e", # Orange
        "T-FT-2": "#8c564b", # Brown
        "ModelEq": "#008080", # Teal
        "Eq-1": "#008080",    # Teal
        "Eq-2": "#20b2aa"     # Light Sea Green
    }
    
    # Fallback list of scientific colors if name not matched
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    summary_rows = []

    for idx, (exp, data) in enumerate(all_data.items()):
        rot_accs = data["rotation_accuracies"]
        # Convert keys from str to int if necessary
        angles = sorted([int(k) for k in rot_accs.keys()])
        accuracies = [rot_accs[str(a)] for a in angles]

        # Compute standard degradation metric: (Acc_0 - Acc_180) / Acc_0
        acc_0 = rot_accs.get("0", rot_accs.get(0, 0.0))
        acc_180 = rot_accs.get("180", rot_accs.get(180, 0.0))
        
        if acc_0 > 0:
            degradation = (acc_0 - acc_180) / acc_0
        else:
            degradation = 0.0

        auc = data.get("rotation_auc", 0.0)
        overall_acc = data.get("overall_accuracy", 0.0)
        macro_f1 = data.get("macro_f1", 0.0)

        # Record summary row
        summary_rows.append({
            "model": exp,
            "acc": f"{overall_acc:.4f}",
            "f1": f"{macro_f1:.4f}",
            "auc": f"{auc:.4f}",
            "deg_180_drop": f"{degradation:.4f}"
        })

        color = colors.get(exp, fallback_colors[idx % len(fallback_colors)])
        
        # Legend Label formatting
        label = f"{exp} (AUC: {auc:.2f}, Deg: {degradation:.1%})"
        
        plt.plot(
            angles, 
            accuracies, 
            marker="o", 
            linewidth=2.5, 
            color=color, 
            label=label
        )

    # Highlight training domain if it corresponds to standard angles [0..150]
    plt.axvspan(0, 150, color="green", alpha=0.07, label="Training Domain (0°-150°)")

    plt.title("Comparative Viewpoint Rotation Sensitivity", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Rotation Angle (Degrees)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.xticks(angles)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, loc="lower left")
    plt.tight_layout()

    plot_path = comp_dir / "rotation_curve_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved comparative rotation plot to: {plot_path}")

    # -------------------------------
    # 2. Save Comparative CSV Summary Table
    # -------------------------------
    csv_path = comp_dir / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "acc", "f1", "auc", "deg_180_drop"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"💾 Saved statistical results summary CSV to: {csv_path}")

    # -------------------------------
    # 3. Print Beautiful Summary Terminal Table
    # -------------------------------
    print("\n📈 ==================== COMPARATIVE RESULTS ====================")
    print(f"{'Model':<10} | {'Overall Acc':<11} | {'Macro F1':<10} | {'Rotation AUC':<12} | {'180° Drop (D)':<12}")
    print("-" * 68)
    for row in summary_rows:
        print(f"{row['model']:<10} | {row['acc']:<11} | {row['f1']:<10} | {row['auc']:<12} | {float(row['deg_180_drop']):.1%}")
    print("================================================================\n")

if __name__ == "__main__":
    main()
