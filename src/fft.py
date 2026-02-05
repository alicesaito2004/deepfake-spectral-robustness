"""
FFT utilities for deepfake spectral analysis.

All operations use torch.fft and are fully differentiable.
"""

import torch


def compute_spectrum(
    image: torch.Tensor,
    per_channel: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute centered 2D log-magnitude spectrum from an image.

    Args:
        image: (B, C, H, W) or (C, H, W) float tensor in [0, 1].
        per_channel: If True, compute spectrum per RGB channel (output C=3).
                     If False, convert to grayscale first (output C=1).
        eps: Small constant for numerical stability.

    Returns:
        Spectrum tensor of shape (B, C, H, W) or (C, H, W).
        C=1 if per_channel=False, C=3 if per_channel=True.
        DC component centered.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if per_channel:
        input_tensor = image
    else:
        if image.shape[1] == 3:
            weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            input_tensor = (image * weights).sum(dim=1, keepdim=True)
        elif image.shape[1] == 1:
            input_tensor = image
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}")

    freq = torch.fft.fft2(input_tensor, norm="ortho")
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    magnitude = torch.abs(freq)
    spectrum = torch.log(magnitude + eps)

    if squeeze_output:
        spectrum = spectrum.squeeze(0)

    return spectrum


def azimuthal_average(spectrum: torch.Tensor) -> torch.Tensor:
    """Compute azimuthally (radially) averaged power spectrum.

    Args:
        spectrum: (H, W) or (1, H, W) centered log-magnitude spectrum.

    Returns:
        1D tensor of length max_radius with average value at each radius.
    """
    if spectrum.dim() == 3:
        spectrum = spectrum.squeeze(0)

    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    max_radius = min(cy, cx)

    y = torch.arange(H, device=spectrum.device, dtype=spectrum.dtype) - cy
    x = torch.arange(W, device=spectrum.device, dtype=spectrum.dtype) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_map = torch.sqrt(xx ** 2 + yy ** 2)

    profile = torch.zeros(max_radius, device=spectrum.device, dtype=spectrum.dtype)
    for r in range(max_radius):
        mask = (radius_map >= r) & (radius_map < r + 1)
        if mask.sum() > 0:
            profile[r] = spectrum[mask].mean()

    return profile


def band_energy(
    spectrum: torch.Tensor,
    n_bands: int = 20,
) -> torch.Tensor:
    """Extract energy in concentric frequency bands.

    Args:
        spectrum: (H, W) or (1, H, W) centered spectrum.
        n_bands: Number of frequency bands.

    Returns:
        1D tensor of length n_bands with mean energy per band.
    """
    if spectrum.dim() == 3:
        spectrum = spectrum.squeeze(0)

    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    max_radius = float(min(cy, cx))

    y = torch.arange(H, device=spectrum.device, dtype=spectrum.dtype) - cy
    x = torch.arange(W, device=spectrum.device, dtype=spectrum.dtype) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_map = torch.sqrt(xx ** 2 + yy ** 2)

    band_width = max_radius / n_bands
    energies = torch.zeros(n_bands, device=spectrum.device, dtype=spectrum.dtype)

    for i in range(n_bands):
        r_inner = i * band_width
        r_outer = (i + 1) * band_width
        mask = (radius_map >= r_inner) & (radius_map < r_outer)
        if mask.sum() > 0:
            energies[i] = spectrum[mask].mean()

    return energies


def spectral_residual(profile: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute residual from ideal 1/f^2 natural image model.

    Args:
        profile: 1D tensor from azimuthal_average.
        eps: Small constant for numerical stability.

    Returns:
        1D tensor of same length, residual from fitted 1/f^2 model.
    """
    n = len(profile)
    freqs = torch.arange(1, n + 1, device=profile.device, dtype=profile.dtype)
    log_freqs = torch.log(freqs)

    valid = profile > -float("inf")
    if valid.sum() < 2:
        return torch.zeros_like(profile)

    x = log_freqs[valid]
    y = profile[valid]

    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum() + eps
    intercept = y_mean - slope * x_mean

    fitted = slope * log_freqs + intercept
    residual = profile - fitted

    return residual
