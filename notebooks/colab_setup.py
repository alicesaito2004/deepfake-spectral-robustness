# Colab Setup
#
# Run this cell at the start of every Colab session.
# It clones the repo, installs dependencies, and mounts Drive.

# ── 1. Clone repo ────────────────────────────────────────────────────────
# Replace with your actual repo URL
REPO_URL = "https://github.com/YOUR_USERNAME/deepfake-forensics.git"

!git clone {REPO_URL} 2>/dev/null || (cd deepfake-forensics && git pull --rebase origin main)
%cd deepfake-forensics

# ── 2. Install dependencies ──────────────────────────────────────────────
!pip install -r requirements.txt -q

# ── 3. Mount Google Drive ────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 4. Symlink data from Drive ───────────────────────────────────────────
# Assumes data is stored in Drive under deepfake-data/
import os

DRIVE_DATA = "/content/drive/MyDrive/deepfake-data"
LOCAL_DATA = "data"

os.makedirs(DRIVE_DATA, exist_ok=True)

if not os.path.exists(LOCAL_DATA):
    os.symlink(DRIVE_DATA, LOCAL_DATA)

# Verify
for subdir in ["celeba", "stylegan2"]:
    path = os.path.join(LOCAL_DATA, subdir)
    if os.path.exists(path):
        n = len(os.listdir(path))
        print(f"  {subdir}: {n} images")
    else:
        print(f"  {subdir}: NOT FOUND — run scripts/setup_data.py")

# ── 5. Verify imports ───────────────────────────────────────────────────
import torch
from src.fft import compute_spectrum, azimuthal_average, band_energy
from src.dataset import DeepfakeDataset, load_splits, make_dataloader

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\nSetup complete.")
