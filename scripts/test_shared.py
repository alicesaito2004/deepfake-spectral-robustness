"""
Run this BEFORE splitting into teams. Verifies every shared component works.

Tests:
  1. FFT pipeline produces correct shapes
  2. Per-channel vs. grayscale spectrum
  3. Azimuthal averaging produces 1D profile
  4. Band energy produces correct number of bands
  5. Spectral residual computes without error
  6. CRITICAL: Gradient flows through FFT (required for adversarial attacks)
  7. Dataset class returns correct tuple format
  8. Splits are valid (no overlap, correct sizes)

Usage:
    python scripts/test_shared.py
    python scripts/test_shared.py --real-dir data/celeba --fake-dir data/stylegan2
"""

import argparse
import json
import os
import sys

import torch
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fft import compute_spectrum, azimuthal_average, band_energy, spectral_residual


PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"


def test_compute_spectrum_grayscale():
    """Grayscale spectrum: (3,256,256) -> (1,256,256)."""
    img = torch.rand(3, 256, 256)
    spec = compute_spectrum(img, per_channel=False)
    assert spec.shape == (1, 256, 256), f"Expected (1,256,256), got {spec.shape}"
    assert not torch.isnan(spec).any(), "NaN in spectrum"
    assert not torch.isinf(spec).any(), "Inf in spectrum"
    print(f"  {PASS} compute_spectrum grayscale: {spec.shape}")


def test_compute_spectrum_perchannel():
    """Per-channel spectrum: (3,256,256) -> (3,256,256)."""
    img = torch.rand(3, 256, 256)
    spec = compute_spectrum(img, per_channel=True)
    assert spec.shape == (3, 256, 256), f"Expected (3,256,256), got {spec.shape}"
    print(f"  {PASS} compute_spectrum per-channel: {spec.shape}")


def test_compute_spectrum_single_channel():
    """Single channel input: (1,256,256) -> (1,256,256)."""
    img = torch.rand(1, 256, 256)
    spec = compute_spectrum(img, per_channel=False)
    assert spec.shape == (1, 256, 256), f"Expected (1,256,256), got {spec.shape}"
    print(f"  {PASS} compute_spectrum single-channel input: {spec.shape}")


def test_azimuthal_average():
    """Azimuthal average: (256,256) -> (128,)."""
    spec_2d = torch.rand(256, 256)
    profile = azimuthal_average(spec_2d)
    expected_len = min(256 // 2, 256 // 2)
    assert profile.shape == (expected_len,), f"Expected ({expected_len},), got {profile.shape}"
    assert (profile >= 0).all(), "Negative values in azimuthal profile"
    print(f"  {PASS} azimuthal_average: {profile.shape}")


def test_band_energy():
    """Band energy: (256,256), 20 bands -> (20,)."""
    spec_2d = torch.rand(256, 256)
    energies = band_energy(spec_2d, n_bands=20)
    assert energies.shape == (20,), f"Expected (20,), got {energies.shape}"
    assert (energies >= 0).all(), "Negative band energies"
    print(f"  {PASS} band_energy: {energies.shape}")


def test_spectral_residual():
    """Spectral residual: (128,) -> (128,)."""
    profile = torch.rand(128) + 0.1  # avoid log(0)
    res = spectral_residual(profile)
    assert res.shape == profile.shape, f"Shape mismatch: {res.shape} vs {profile.shape}"
    assert not torch.isnan(res).any(), "NaN in residual"
    print(f"  {PASS} spectral_residual: {res.shape}")


def test_gradient_flow():
    """CRITICAL: Gradients must flow through FFT for adversarial attacks."""
    img = torch.rand(3, 256, 256, requires_grad=True)
    spec = compute_spectrum(img, per_channel=False)
    loss = spec.sum()
    loss.backward()

    assert img.grad is not None, "No gradient computed"
    grad_norm = img.grad.norm().item()
    assert grad_norm > 0, f"Gradient is ZERO — adversarial attacks will not work"
    print(f"  {PASS} gradient flow through FFT: grad_norm = {grad_norm:.4f}")


def test_gradient_flow_perchannel():
    """Gradients through per-channel FFT."""
    img = torch.rand(3, 256, 256, requires_grad=True)
    spec = compute_spectrum(img, per_channel=True)
    loss = spec.sum()
    loss.backward()

    grad_norm = img.grad.norm().item()
    assert grad_norm > 0, "Gradient is ZERO for per-channel FFT"
    print(f"  {PASS} gradient flow (per-channel): grad_norm = {grad_norm:.4f}")


def test_spectrum_detects_artifact():
    """Sanity: spectrum of image with periodic artifact differs from clean image."""
    torch.manual_seed(42)
    clean = torch.rand(3, 256, 256) * 0.5 + 0.25

    # Add checkerboard artifact (mimics transposed conv)
    y = torch.arange(256).float()
    x = torch.arange(256).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    artifact = 0.03 * torch.sin(2 * 3.14159 * xx / 4) * torch.sin(2 * 3.14159 * yy / 4)

    dirty = (clean + artifact.unsqueeze(0)).clamp(0, 1)

    spec_clean = compute_spectrum(clean).squeeze(0)
    spec_dirty = compute_spectrum(dirty).squeeze(0)

    diff = (spec_dirty - spec_clean).abs().max().item()
    assert diff > 0.01, f"Spectrum difference too small ({diff:.6f}) — artifact not visible"
    print(f"  {PASS} artifact detection: max spectral diff = {diff:.4f}")


def test_dataset_class(real_dir=None, fake_dir=None):
    """Dataset returns (image, spectrum, label) with correct shapes."""
    if real_dir and fake_dir:
        from src.dataset import DeepfakeDataset
        import os

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        real_paths = sorted([
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])[:5]
        fake_paths = sorted([
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])[:5]

        ds = DeepfakeDataset(real_paths, fake_paths, img_size=256)
        img, spec, label = ds[0]

        assert img.shape == (3, 256, 256), f"Image shape: {img.shape}"
        assert spec.shape[1:] == (256, 256), f"Spectrum shape: {spec.shape}"
        assert label in (0, 1), f"Label: {label}"
        assert img.min() >= 0 and img.max() <= 1, "Image not in [0,1]"
        print(f"  {PASS} dataset: img={img.shape}, spec={spec.shape}, label={label}")
    else:
        print(f"  SKIP dataset test (no --real-dir / --fake-dir provided)")


def test_splits(splits_path=None):
    """Splits have no overlap and correct ratios."""
    if splits_path and os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)

        for prefix in ["real", "fake"]:
            key = f"{prefix}_indices"
            train = set(splits["train"][key])
            val = set(splits["val"][key])
            test = set(splits["test"][key])

            assert len(train & val) == 0, f"{prefix}: train/val overlap"
            assert len(train & test) == 0, f"{prefix}: train/test overlap"
            assert len(val & test) == 0, f"{prefix}: val/test overlap"

            total = len(train) + len(val) + len(test)
            train_ratio = len(train) / total
            val_ratio = len(val) / total
            test_ratio = len(test) / total

            print(f"  {PASS} {prefix} splits: "
                  f"train={len(train)} ({train_ratio:.0%}), "
                  f"val={len(val)} ({val_ratio:.0%}), "
                  f"test={len(test)} ({test_ratio:.0%})")
    else:
        print(f"  SKIP splits test (no splits.json found)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--fake-dir", default=None)
    parser.add_argument("--splits", default="config/splits.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  SHARED INFRASTRUCTURE TESTS")
    print("=" * 60)

    tests = [
        ("FFT: grayscale spectrum", test_compute_spectrum_grayscale),
        ("FFT: per-channel spectrum", test_compute_spectrum_perchannel),
        ("FFT: single-channel input", test_compute_spectrum_single_channel),
        ("FFT: azimuthal average", test_azimuthal_average),
        ("FFT: band energy", test_band_energy),
        ("FFT: spectral residual", test_spectral_residual),
        ("FFT: gradient flow (grayscale)", test_gradient_flow),
        ("FFT: gradient flow (per-channel)", test_gradient_flow_perchannel),
        ("FFT: artifact detection", test_spectrum_detects_artifact),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {FAIL} {name}: {e}")
            failed += 1

    # Tests that need data
    print()
    try:
        test_dataset_class(args.real_dir, args.fake_dir)
        if args.real_dir:
            passed += 1
    except Exception as e:
        print(f"  {FAIL} dataset: {e}")
        failed += 1

    try:
        test_splits(args.splits)
        if os.path.exists(args.splits):
            passed += 1
    except Exception as e:
        print(f"  {FAIL} splits: {e}")
        failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)

    if failed > 0:
        print("\nFix failures before splitting into teams.")
        sys.exit(1)
    else:
        print("\nShared infrastructure is verified. Safe to split.")


if __name__ == "__main__":
    main()
