# Representation-Dependent Adversarial Robustness of Deepfake Detectors

Does the choice of input representation — pixel, frequency, or fused — determine a deepfake detector's robustness to adversarial attack?

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-forensics.git
cd deepfake-forensics
pip install -r requirements.txt
```

### 2. Download data

```bash
python scripts/setup_data.py --data-root data/
```

This downloads CelebA (real faces) and StyleGAN2 (generated faces), resizes everything to 256×256. If auto-download fails, see the script output for manual download instructions.

### 3. Generate splits

```bash
python scripts/generate_splits.py \
    --real-dir data/celeba \
    --fake-dir data/stylegan2 \
    --output config/splits.json
```

### 4. Sanity check

```bash
python scripts/sanity_check.py \
    --real-dir data/celeba \
    --fake-dir data/stylegan2 \
    --output-dir outputs/sanity_check
```

Review the plots in `outputs/sanity_check/`. If real vs. fake spectra are not visibly different, debug before proceeding.

### 5. Precompute spectra (optional, speeds up training)

```bash
python scripts/precompute_spectra.py \
    --real-dir data/celeba \
    --fake-dir data/stylegan2 \
    --output-dir data/spectra
```

### Colab

If working in Google Colab, run `notebooks/colab_setup.py` at the start of each session. Store data on Google Drive under `deepfake-data/`.

## Project Structure

```
deepfake-forensics/
├── config/
│   └── splits.json              # Train/val/test indices (generated)
├── data/                        # Images + cached spectra (not in repo)
├── src/
│   ├── __init__.py
│   ├── fft.py                   # Shared FFT function (FROZEN)
│   ├── dataset.py               # Shared dataset class (FROZEN)
│   ├── classifier.py            # Three classifier architectures
│   ├── train_classifier.py      # Training script
│   ├── eval.py                  # Evaluation metrics
│   ├── gradcam.py               # Grad-CAM interpretability
│   └── attacks.py               # FGSM, PGD
├── scripts/
│   ├── setup_data.py            # Download and preprocess data
│   ├── generate_splits.py       # Create train/val/test splits
│   ├── precompute_spectra.py    # Cache spectra to disk
│   └── sanity_check.py          # Visual verification
├── notebooks/
│   └── colab_setup.py           # Colab session initializer
├── app/
│   └── gradio_app.py            # Interactive demo
├── outputs/                     # Plots, figures, results
├── requirements.txt
├── .gitignore
└── README.md
```

## Shared Files (Frozen After Day 1)

These files are the contract between teams. Do not modify without notifying everyone:

- `config/splits.json` — Deterministic data split indices
- `src/fft.py` — FFT computation (fully differentiable, torch ops only)
- `src/dataset.py` — Dataset class returning (image, spectrum, label)

## Team Structure

- **Team 1 (Sean + Alice):** FFT pipeline, spectral analysis, adversarial attacks
- **Team 2 (Yuantong + Meiqi):** Classifiers, training, Grad-CAM, evaluation, demo
