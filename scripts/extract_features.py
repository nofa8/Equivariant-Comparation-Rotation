import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from src.models.factory import build_model
from src.datasets.modelnet_dataset import ModelNetDataset
from src.datasets.transforms import build_transforms

def main():
    parser = argparse.ArgumentParser(description="Extract latent features from a trained ResNet-18 model.")
    parser.add_argument("--ckpt", type=str, default="outputs/T-FT-2/checkpoints/best.pt", help="Path to ResNet-18 checkpoint.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to extract features for.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # Resolve paths
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    exp_dir = ckpt_path.parent.parent
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found at: {config_path}")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"📖 Loaded experiment config from: {config_path}")

    # 1. Load Checkpoint & Weights
    print("🏗️ Building ResNet-18 model...")
    num_classes = config.get("num_classes", 10)
    model = build_model(config["model"], num_classes=num_classes)
    
    # Check if checkpoint contains rich metadata
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    # 2. Modify Model to Extract Latent Features
    # Since model.fc is our sequential classification head,
    # setting model.fc to Identity extracts the 512-dimensional output of the avgpool layer.
    original_head = model.fc
    model.fc = nn.Identity()
    print("✂️ Replaced classification head with Identity layer to extract 512D backbone features.")

    # 3. Build Dataset & DataLoader
    val_transform = build_transforms(img_size=224, augment=False, normalize=config.get("normalize", False))
    
    # We use all 12 views representing 0° to 330° for full test set feature extraction
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

    print(f"📋 Loaded '{args.split}' split with {len(dataset)} images (all 12 views).")

    # 4. Forward Pass & Feature Accumulation
    features = []
    labels = []
    
    print("🕵️ Extracting features...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = model(x)
            features.append(feat.cpu())
            labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"📈 Extracted feature tensor shape: {features.shape}")
    print(f"🏷️ Label tensor shape: {labels.shape}")

    # 5. Save Features
    out_dir = exp_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    features_path = out_dir / f"test_features_512d.pt"
    labels_path = out_dir / f"test_labels.pt"
    
    torch.save(features, features_path)
    torch.save(labels, labels_path)
    print(f"💾 Saved computed features to: {features_path}")
    print(f"💾 Saved computed labels to: {labels_path}")

    # Let's save as numpy as well to be fully robust
    np_features_path = out_dir / f"test_features_512d.npy"
    np.save(np_features_path, features.numpy())
    print(f"💾 Saved computed features to numpy format: {np_features_path}\n")

if __name__ == "__main__":
    main()
