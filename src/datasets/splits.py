import json
from pathlib import Path
import numpy as np


# This function is used to split the model directories into train, val, and test sets
# It takes the model directories and the ratios of the splits as input
# It returns a dictionary with the train, val, and test sets
def generate_splits(model_dirs, ratios, seed):
    # A random number generator is created with the given seed
    rng = np.random.default_rng(seed)
    model_dirs = list(model_dirs)
    rng.shuffle(model_dirs)

    n = len(model_dirs)
    t1 = int(ratios[0] * n)
    t2 = int((ratios[0] + ratios[1]) * n)

    return {
        "train": model_dirs[:t1],
        "val": model_dirs[t1:t2],
        "test": model_dirs[t2:],
    }


def save_splits(split_dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(split_dict, f, indent=2)


def load_splits(path: Path):
    with open(path, "r") as f:
        return json.load(f)