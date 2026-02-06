# Deepfake Spectral Robustness

Investigating whether input representation affects adversarial robustness of deepfake detectors.

## Research Question

Does transforming images to the frequency domain (via FFT) provide implicit robustness against adversarial attacks? We train three CNN classifiers with identical architectures but different inputs and measure which is hardest to attack.

## Classifiers

| Model | Input | Description |
|-------|-------|-------------|
| Pixel Classifier | RGB image (3, 256, 256) | Baseline, operates on raw pixels |
| Spectrum Classifier | FFT magnitude (1, 256, 256) | Operates on frequency domain |
| Dual-Branch Classifier | Both | Fuses pixel and spectrum features |

All three share the same 4-block CNN backbone (32→64→128→256 channels) with global average pooling and linear head.

## Project Structure

```
├── config/
│   └── splits.json              # Train/val/test indices (seed 42)
├── scripts/
│   ├── setup_data.py            # Process raw images to 256x256
│   ├── setup_cross_generator.py # Process SD images for generalization test
│   ├── generate_splits.py       # Create deterministic splits
│   ├── sanity_check.py          # Spectral analysis plots
│   ├── compare_channels.py      # Grayscale vs RGB comparison
│   └── eval_adversarial.py      # Adversarial robustness evaluation
├── src/
│   ├── fft.py                   # Differentiable FFT operations
│   ├── dataset.py               # Dataset class returning (image, spectrum, label)
│   ├── classifier.py            # Pixel, Spectrum, and Dual-Branch classifiers
│   └── attacks.py               # FGSM and PGD implementations
├── tests/
│   ├── test_attacks.py          # FGSM unit tests
│   └── test_gradient_flow.py    # Verify gradients flow through FFT
├── checkpoints/                 # Trained model weights
├── outputs/                     # Plots and results
└── data/
    ├── processed/               # Training data (5000 real + 4630 fake)
    └── cross_generator/         # Stable Diffusion test set
```

## Setup

```bash
git clone https://github.com/alicesaito2004/deepfake-spectral-robustness.git
cd deepfake-spectral-robustness
pip install -r requirements.txt
```

## Data Preparation

1. Download the Human Faces Dataset and extract to `data/raw/`:
   ```
   data/raw/
       real/
       fake/
   ```

2. Process and generate splits:
   ```bash
   python scripts/setup_data.py
   python scripts/generate_splits.py
   ```

3. Verify pipeline:
   ```bash
   PYTHONPATH=. python scripts/sanity_check.py
   ```

## Training

```bash
PYTHONPATH=. python scripts/train.py
```

Checkpoints saved to `checkpoints/`.

## Adversarial Evaluation

```bash
PYTHONPATH=. python scripts/eval_adversarial.py \
    --pixel-ckpt checkpoints/pixel_classifier.pt \
    --spectrum-ckpt checkpoints/spectrum_classifier.pt \
    --dual-ckpt checkpoints/dual_classifier.pt
```

Runs PGD-20 attacks at ε = 1/255, 2/255, 4/255, 8/255, 16/255.

## Key Findings

### Spectral Analysis
- Fake images retain more high-frequency energy than real images
- Grayscale spectrum (1 channel) slightly more discriminative than per-channel RGB
- Clear spectral differences visible in azimuthal average and band energy plots

### Adversarial Robustness
*Results pending classifier training*

## Team

- Sean & Alice: FFT pipeline, adversarial attacks, robustness evaluation
- Yuantong & Meiqi: Classifier architectures, training, interpretability

## References

- Durall et al. (2020) "Unmasking DeepFakes with simple Features"
- Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Recognition"
- Dzanic et al. (2020) "Fourier Spectrum Discrepancies in Deep Network Generated Images"
