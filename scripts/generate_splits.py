"""
Generate deterministic train/val/test splits.

Run once after data download to create config/splits.json.
Seed 42 ensures reproducibility across all team members.
"""

import json
import random
from pathlib import Path


def generate_splits(
    data_dir: str = "data/processed",
    output_path: str = "config/splits.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    data_path = Path(data_dir)
    real_dir = data_path / "real"
    fake_dir = data_path / "fake"

    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {real_dir} and {fake_dir} to exist. "
            f"Run scripts/setup_data.py first."
        )

    real_files = sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpg"))
    fake_files = sorted(fake_dir.glob("*.png")) + sorted(fake_dir.glob("*.jpg"))

    n_real = len(real_files)
    n_fake = len(fake_files)

    print(f"Found {n_real} real, {n_fake} fake")

    random.seed(seed)

    real_indices = list(range(n_real))
    fake_indices = list(range(n_fake))
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    def split_indices(indices, train_r, val_r):
        n = len(indices)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return {
            "train": sorted(indices[:n_train]),
            "val": sorted(indices[n_train:n_train + n_val]),
            "test": sorted(indices[n_train + n_val:]),
        }

    real_splits = split_indices(real_indices, train_ratio, val_ratio)
    fake_splits = split_indices(fake_indices, train_ratio, val_ratio)

    splits = {
        "train": {
            "real_indices": real_splits["train"],
            "fake_indices": fake_splits["train"],
        },
        "val": {
            "real_indices": real_splits["val"],
            "fake_indices": fake_splits["val"],
        },
        "test": {
            "real_indices": real_splits["test"],
            "fake_indices": fake_splits["test"],
        },
        "metadata": {
            "seed": seed,
            "n_real": n_real,
            "n_fake": n_fake,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved to {output_path}")
    for name in ["train", "val", "test"]:
        nr = len(splits[name]["real_indices"])
        nf = len(splits[name]["fake_indices"])
        print(f"  {name}: {nr} real, {nf} fake")


if __name__ == "__main__":
    generate_splits()
