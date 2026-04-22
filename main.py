from scripts.run_experiment import main as run_experiment_main


def main():
    print("🚀 Starting the DL project!")

    # Create dummy files
    import os

    os.makedirs("data/raw/ModelNet10_views", exist_ok=True)
    with open("data/processed/splits.json", "w") as f:
        f.write('{"train": [], "val": [], "test": []}')

    # Create a dummy experiment file
    os.makedirs("configs", exist_ok=True)
    with open("configs/experiments.yaml", "w") as f:
        f.write("S-1:\n")
        f.write("  model: model_s\n")
        f.write("  optimizer: adamw\n")
        f.write("  loss: ce\n")
        f.write("  augmentation: false\n")

    # Run the experiment
    run_experiment_main()


if __name__ == "__main__":
    main()
