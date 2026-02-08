"""
Cross-generator evaluation: test all three classifiers on Stable Diffusion fakes
that were never seen during training.

Real images: test split from data/processed/real/ (via splits.json)
Fake images: ALL images from data/cross_generator/{sd15, sd21, sdxl}/

Usage:
    PYTHONPATH=. python scripts/eval_cross_generator.py
    PYTHONPATH=. python scripts/eval_cross_generator.py --sd-version sd15
    PYTHONPATH=. python scripts/eval_cross_generator.py --sd-version all
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fft import compute_spectrum
from src.dataset import load_splits, get_paths_for_split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrossGeneratorDataset(Dataset):
    """
    Dataset for cross-generator evaluation.
    Real: test-split images from data/processed/real/
    Fake: all images from a given SD version directory.
    Returns (image, spectrum, label).
    """

    def __init__(self, real_paths: list, fake_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Real samples from test split
        self.samples = [(p, 0) for p in real_paths]

        # Fake samples: all images in fake_dir
        fake_path = Path(fake_dir)
        fake_files = sorted(
            [str(f) for f in fake_path.glob("*.*")
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        )
        self.samples += [(p, 1) for p in fake_files]

        self.n_real = len(real_paths)
        self.n_fake = len(fake_files)
        print(f"  Dataset: {self.n_real} real (test split), {self.n_fake} fake ({fake_dir})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        spectrum = compute_spectrum(image, per_channel=False)
        return image, spectrum, label


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, input_type="pixel"):
    """
    Run inference and compute metrics.

    Args:
        model: trained classifier
        dataloader: CrossGeneratorDataset loader
        device: torch device
        input_type: "pixel", "spectrum", or "dual"

    Returns:
        dict with accuracy, precision, recall, f1, confusion matrix counts
    """
    model.eval()
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, spectra, labels in tqdm(dataloader, desc="    Evaluating", leave=False):
            images = images.to(device)
            spectra = spectra.to(device)
            labels = labels.to(device)

            if input_type == "pixel":
                outputs = model(images)
            elif input_type == "spectrum":
                outputs = model(spectra)
            elif input_type == "dual":
                outputs = model(images, spectra)
            else:
                raise ValueError(f"Unknown input_type: {input_type}")

            _, predicted = torch.max(outputs, 1)

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------

def load_classifier(name, checkpoint_path, device):
    """
    Load a classifier from checkpoint.
    Adjust imports here to match your actual classifier module.
    """
    # Try importing from src.classifier first, fall back to src.models
    try:
        from src.classifier import PixelClassifier, SpectrumClassifier, DualClassifier
    except ImportError:
        from src.models import PixelClassifier, SpectrumClassifier, DualClassifier

    if name == "pixel":
        model = PixelClassifier()
    elif name == "spectrum":
        model = SpectrumClassifier()
    elif name == "dual":
        model = DualClassifier()
    else:
        raise ValueError(f"Unknown classifier: {name}")

    state = torch.load(checkpoint_path, map_location=device)
    # Handle both bare state_dict and wrapped checkpoint formats
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLASSIFIERS = {
    "pixel": {
        "checkpoint": "checkpoints/pixel_classifier.pt",
        "input_type": "pixel",
    },
    "spectrum": {
        "checkpoint": "checkpoints/spectrum_classifier.pt",
        "input_type": "spectrum",
    },
    "dual": {
        "checkpoint": "checkpoints/dual_classifier.pt",
        "input_type": "dual",
    },
}

SD_VERSIONS = ["sd15", "sd21", "sdxl"]


def main():
    parser = argparse.ArgumentParser(description="Cross-generator evaluation on SD fakes")
    parser.add_argument("--real-dir", default="data/processed/real")
    parser.add_argument("--cross-gen-dir", default="data/cross_generator")
    parser.add_argument("--splits-file", default="config/splits.json")
    parser.add_argument("--sd-version", default="all",
                        help="Which SD version to test: sd15, sd21, sdxl, or all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/cross_generator_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Get real test-split paths
    real_paths = sorted([str(p) for p in Path(args.real_dir).glob("*.png")]
                        + [str(p) for p in Path(args.real_dir).glob("*.jpg")])
    splits = load_splits(args.splits_file)
    real_test_paths, _ = get_paths_for_split(real_paths, [], splits, "test")
    print(f"Real test images: {len(real_test_paths)}")

    # Determine which SD versions to evaluate
    versions = SD_VERSIONS if args.sd_version == "all" else [args.sd_version]

    all_results = {}

    for sd_ver in versions:
        fake_dir = os.path.join(args.cross_gen_dir, sd_ver)
        if not os.path.exists(fake_dir):
            print(f"\nSKIP {sd_ver}: {fake_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"SD Version: {sd_ver}")
        print(f"{'='*60}")

        dataset = CrossGeneratorDataset(real_test_paths, fake_dir)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        all_results[sd_ver] = {}

        for clf_name, cfg in CLASSIFIERS.items():
            ckpt = cfg["checkpoint"]
            if not os.path.exists(ckpt):
                print(f"\n  SKIP {clf_name}: {ckpt} not found")
                continue

            print(f"\n  Classifier: {clf_name}")
            model = load_classifier(clf_name, ckpt, device)
            metrics = evaluate(model, dataloader, device, cfg["input_type"])
            all_results[sd_ver][clf_name] = metrics

            print(f"    Accuracy:  {metrics['accuracy']}")
            print(f"    Precision: {metrics['precision']}")
            print(f"    Recall:    {metrics['recall']}  (fake detection rate)")
            print(f"    F1:        {metrics['f1']}")
            print(f"    TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Detection Rate (Recall) on SD Fakes")
    print(f"{'='*60}")
    print(f"{'SD Version':<12}", end="")
    for clf_name in CLASSIFIERS:
        print(f"{clf_name:<14}", end="")
    print()
    print("-" * 54)
    for sd_ver in versions:
        if sd_ver not in all_results:
            continue
        print(f"{sd_ver:<12}", end="")
        for clf_name in CLASSIFIERS:
            if clf_name in all_results[sd_ver]:
                r = all_results[sd_ver][clf_name]["recall"]
                print(f"{r:<14.4f}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
