"""
Adversarial attack implementations.

All attacks are differentiable and work with the spectral pipeline.
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """Fast Gradient Sign Method attack.

    Args:
        model: Classifier model that takes images and returns logits.
        images: Input images (B, C, H, W) in [0, 1].
        labels: Ground truth labels (B,).
        epsilon: Perturbation magnitude in L-infinity norm.
        criterion: Loss function. Defaults to CrossEntropyLoss.

    Returns:
        Adversarial images (B, C, H, W) clamped to [0, 1].
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images = images.clone().detach().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    grad_sign = images.grad.sign()
    perturbed = images + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    criterion: nn.Module = None,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent attack.

    Args:
        model: Classifier model that takes images and returns logits.
        images: Input images (B, C, H, W) in [0, 1].
        labels: Ground truth labels (B,).
        epsilon: Maximum perturbation in L-infinity norm.
        alpha: Step size per iteration.
        num_steps: Number of PGD iterations.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        random_start: If True, start from random point within epsilon ball.

    Returns:
        Adversarial images (B, C, H, W) clamped to [0, 1].
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    perturbed = images.clone().detach()

    if random_start:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

    for _ in range(num_steps):
        perturbed.requires_grad_(True)

        outputs = model(perturbed)
        loss = criterion(outputs, labels)
        loss.backward()

        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad_sign

        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0.0, 1.0)

    return perturbed.detach()


def fgsm_attack_spectrum(
    model: nn.Module,
    images: torch.Tensor,
    spectra: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    criterion: nn.Module = None,
    attack_target: str = "image",
) -> tuple[torch.Tensor, torch.Tensor]:
    """FGSM attack for models that take both image and spectrum inputs.

    Args:
        model: Classifier that takes (images, spectra) and returns logits.
        images: Input images (B, C, H, W) in [0, 1].
        spectra: Input spectra (B, 1, H, W).
        labels: Ground truth labels (B,).
        epsilon: Perturbation magnitude.
        criterion: Loss function.
        attack_target: "image", "spectrum", or "both".

    Returns:
        Tuple of (perturbed_images, perturbed_spectra).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images = images.clone().detach()
    spectra = spectra.clone().detach()

    if attack_target in ("image", "both"):
        images.requires_grad_(True)
    if attack_target in ("spectrum", "both"):
        spectra.requires_grad_(True)

    outputs = model(images, spectra)
    loss = criterion(outputs, labels)
    loss.backward()

    perturbed_images = images
    perturbed_spectra = spectra

    if attack_target in ("image", "both") and images.grad is not None:
        grad_sign = images.grad.sign()
        perturbed_images = images + epsilon * grad_sign
        perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)

    if attack_target in ("spectrum", "both") and spectra.grad is not None:
        grad_sign = spectra.grad.sign()
        perturbed_spectra = spectra + epsilon * grad_sign

    return perturbed_images.detach(), perturbed_spectra.detach()


def fgsm_through_fft(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    compute_spectrum_fn,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """FGSM attack on spectrum classifier with gradients through FFT.

    Perturbs images in pixel space, recomputes spectrum, attacks spectrum classifier.
    Gradients flow: loss -> spectrum -> FFT -> pixels.

    Args:
        model: Spectrum classifier that takes spectra and returns logits.
        images: Input images (B, C, H, W) in [0, 1].
        labels: Ground truth labels (B,).
        epsilon: Perturbation magnitude.
        compute_spectrum_fn: Function to compute spectrum from images.
        criterion: Loss function.

    Returns:
        Adversarial images (B, C, H, W) clamped to [0, 1].
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images = images.clone().detach().requires_grad_(True)

    spectra = compute_spectrum_fn(images)
    outputs = model(spectra)
    loss = criterion(outputs, labels)
    loss.backward()

    grad_sign = images.grad.sign()
    perturbed = images + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed.detach()
