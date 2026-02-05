"""
Shared dataset class. Single source of truth for data loading.

FROZEN AFTER DAY 1. Do not modify without notifying the other team.

Returns (image, spectrum, label) tuples so all three classifiers
can train from the same DataLoader.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.fft import compute_spectrum


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """Binary classification dataset: real (0) vs. fake (1).

    Each sample returns:
        image:    (3, H, W) float32 tensor in [0, 1]
        spectrum: (1, H, W) or (3, H, W) log-magnitude power spectrum
        label:    int, 0 = real, 1 = fake
    """

    def __init__(
        self,
        real_paths: list[str],
        fake_paths: list[str],
        img_size: int = 256,
        per_channel_spectrum: bool = False,
        precomputed_spectra_dir: Optional[str] = None,
    ):
        """
        Args:
            real_paths: List of file paths to real images.
            fake_paths: List of file paths to fake images.
            img_size: Resize target (square).
            per_channel_spectrum: If True, spectrum has 3 channels (R, G, B).
            precomputed_spectra_dir: If set, load cached spectra from this dir
                                     instead of computing on the fly.
        """
        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        self.per_channel = per_channel_spectrum
        self.precomputed_dir = precomputed_spectra_dir

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # -> [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]

        # Load and transform image
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        # Spectrum: load from cache or compute
        if self.precomputed_dir is not None:
            spec_path = os.path.join(
                self.precomputed_dir, f"{Path(path).stem}.pt"
            )
            if os.path.exists(spec_path):
                spectrum = torch.load(spec_path, weights_only=True)
            else:
                spectrum = compute_spectrum(image, per_channel=self.per_channel)
        else:
            spectrum = compute_spectrum(image, per_channel=self.per_channel)

        return image, spectrum, label


# ---------------------------------------------------------------------------
# Split management
# ---------------------------------------------------------------------------

def load_splits(splits_path: str) -> dict:
    """Load train/val/test split indices from JSON."""
    with open(splits_path, "r") as f:
        return json.load(f)


def get_paths_for_split(
    all_real_paths: list[str],
    all_fake_paths: list[str],
    splits: dict,
    split_name: str,
) -> tuple[list[str], list[str]]:
    """Select real and fake paths for a given split.

    The split indices are applied independently to real and fake lists.
    Both lists must be sorted deterministically before calling this.

    Args:
        all_real_paths: Sorted list of all real image paths.
        all_fake_paths: Sorted list of all fake image paths.
        splits: Dict with keys 'train', 'val', 'test', each mapping to
                a dict with 'real_indices' and 'fake_indices'.
        split_name: One of 'train', 'val', 'test'.

    Returns:
        (real_paths, fake_paths) for the requested split.
    """
    split = splits[split_name]
    real = [all_real_paths[i] for i in split["real_indices"]]
    fake = [all_fake_paths[i] for i in split["fake_indices"]]
    return real, fake


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    real_paths: list[str],
    fake_paths: list[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    img_size: int = 256,
    per_channel_spectrum: bool = False,
    precomputed_spectra_dir: Optional[str] = None,
) -> DataLoader:
    """Convenience wrapper to build a DataLoader."""
    dataset = DeepfakeDataset(
        real_paths=real_paths,
        fake_paths=fake_paths,
        img_size=img_size,
        per_channel_spectrum=per_channel_spectrum,
        precomputed_spectra_dir=precomputed_spectra_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
