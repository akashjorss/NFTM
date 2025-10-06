from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


Reduction = Literal["mean", "none"]


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr(pred: torch.Tensor, target: torch.Tensor, reduction: Reduction = "mean") -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target, reduction="none")
    dims = tuple(range(1, mse.ndim))
    mse = mse.mean(dim=dims)
    psnr_vals = 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))
    if reduction == "none":
        return psnr_vals
    return psnr_vals.mean()


def _gaussian_window(kernel_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    window = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return window


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: Reduction = "mean",
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    channel = pred.size(1)
    window = _gaussian_window(kernel_size, sigma, channel, pred.device, pred.dtype)
    padding = kernel_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channel)
    mu_target = F.conv2d(target, window, padding=padding, groups=channel)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / denominator

    dims = tuple(range(1, ssim_map.ndim))
    ssim_vals = ssim_map.mean(dim=dims)
    if reduction == "none":
        return ssim_vals
    return ssim_vals.mean()


_lpips_cache: dict[tuple[torch.device, str], torch.nn.Module] = {}


def lpips_dist(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    reduction: Reduction = "mean",
) -> torch.Tensor:
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    if pred.shape != target.shape:
        raise ValueError("Input tensors to lpips_dist must have the same shape")

    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - dependency error surfaced to caller
        raise ImportError("lpips package is required for lpips_dist") from exc

    key = (pred.device, net)
    model = _lpips_cache.get(key)
    if model is None:
        model = lpips.LPIPS(net=net).to(pred.device)
        model.eval()
        _lpips_cache[key] = model

    with torch.no_grad():
        values = model(pred, target)

    values = values.view(values.size(0), -1).mean(dim=1)
    if reduction == "none":
        return values
    return values.mean()
