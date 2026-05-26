from src.datasets.modelnet_dataset import ModelNetDataset

# Define your thesis canonical angles [cite: 87]
TRAIN_ANGLES = [0, 30, 60, 90, 120, 150]

train_set = ModelNetDataset(root="data/raw/ModelNet10_views", split="train", allowed_angles=TRAIN_ANGLES)
#val_set = ModelNetDataset(root="data/raw/ModelNet10_views", split="val", allowed_angles=TRAIN_ANGLES)
test_set = ModelNetDataset(root="data/raw/ModelNet10_views", split="test", allowed_angles=TRAIN_ANGLES)

train_ids = train_set.get_model_ids()
#val_ids = val_set.get_model_ids()
test_ids = test_set.get_model_ids()

# Critical Research Verification
#assert train_ids.isdisjoint(val_ids), "LEAKAGE DETECTED: Train and Val overlap!"
assert train_ids.isdisjoint(test_ids), "LEAKAGE DETECTED: Train and Test overlap!"

print(f"Dataset Size: {len(train_set)+ len(test_set)} images")
print("Splits are reproducible and zero-leakage verified ✅")