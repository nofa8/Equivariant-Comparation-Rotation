import json
from pathlib import Path

def main():
    nb_path = Path("notebooks/run_all_experiments.ipynb")
    print(f"Loading notebook from {nb_path}...")
    with open(nb_path, "r") as f:
        nb = json.load(f)
        
    # Cell 9: Update evaluation loop
    nb["cells"][9]["source"] = [
        "# Scientific evaluation of all 8 competed experiments across 12 viewpoints\n",
        "for exp in ['S-1', 'S-3', 'T-FE-1', 'T-FE-2', 'T-FT-1', 'T-FT-2', 'Eq-1', 'Eq-2']:\n",
        "    evaluate_experiment(exp, split='val')\n"
    ]
    
    # Cell 11: Update comparative dashboard list
    nb["cells"][11]["source"] = [
        "# Define the experiments you want to include in your comparative analysis\n",
        "active_experiments = 'S-1,S-3,T-FE-1,T-FE-2,T-FT-1,T-FT-2,Eq-1,Eq-2'\n",
        "\n",
        "print(f\"📊 Aggregating curves and metrics for: {active_experiments}...\")\n",
        "cmd = f\"PYTHONPATH=. uv run python scripts/generate_comparison.py --exps {active_experiments}\"\n",
        "os.system(cmd)\n",
        "\n",
        "# Load and display statistical summary table\n",
        "summary_df = pd.read_csv(\"outputs/comparisons/results_summary.csv\")\n",
        "summary_df['deg_180_drop'] = summary_df['deg_180_drop'].apply(lambda x: f\"{x:.1%}\")\n",
        "\n",
        "print(\"\\n🏆 STATISTICAL ANALYSIS TABLE:\")\n",
        "display(summary_df)\n"
    ]
    
    # Cell 15: Show confusion matrix of the equivariant model
    nb["cells"][15]["source"] = [
        "def show_confusion_matrix(exp_name):\n",
        "    path = f\"outputs/{exp_name}/figures/confusion_matrix.png\"\n",
        "    if os.path.exists(path):\n",
        "        print(f\"\\n🖼️ Confusion Matrix for {exp_name}:\")\n",
        "        display(Image(filename=path, width=600))\n",
        "    else:\n",
        "        print(f\"⚠️ Confusion Matrix plot not found at {path}\")\n",
        "\n",
        "show_confusion_matrix(\"Eq-1\")\n"
    ]
    
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with all 8 models!")

if __name__ == "__main__":
    main()
