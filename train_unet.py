#!/usr/bin/env python3
"""Train a Tiny U-Net for CIFAR-10 masked inpainting."""

import argparse
import json
import os
import time
import copy
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.utils as vutils

from unet_model import TinyUNet
from metrics import psnr as psnr_metric, ssim as ssim_metric, lpips_dist, param_count
from image_inpainting import (
    make_transforms,
    random_mask,
    corrupt_images,
    clamp_known,
    tv_l1,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny U-Net on CIFAR-10 inpainting")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="out_unet")
    parser.add_argument("--target_params", type=int, default=46375)
    parser.add_argument("--base", type=int, default=10, help="Base channel width for TinyUNet")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--eval_subset", type=int, default=2000,
                        help="Number of test images used for per-epoch validation (0 to disable)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.3)
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def instantiate_unet(base_width: int) -> nn.Module:
    """Instantiate TinyUNet handling minor signature variations."""
    options = [
        {"in_channels": 4, "out_channels": 3, "base": base_width},
        {"in_channels": 4, "out_channels": 3, "base_channels": base_width},
        {"in_channels": 4, "out_channels": 3, "width": base_width},
        {"in_ch": 4, "out_ch": 3, "base": base_width},
        {"in_ch": 4, "out_ch": 3, "base_channels": base_width},
        {"in_ch": 4, "out_ch": 3, "width": base_width},
    ]
    for kwargs in options:
        try:
            return TinyUNet(**kwargs)
        except TypeError:
            continue
    # Fallback to positional instantiation attempts
    try:
        return TinyUNet(4, 3, base_width)
    except TypeError:
        pass
    try:
        return TinyUNet(4, 3)
    except TypeError as exc:
        raise RuntimeError("Unable to instantiate TinyUNet with provided arguments") from exc


def prepare_dataloaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    transform = make_transforms()
    train_set = tv.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_expand = mask.expand_as(pred)
    diff = (pred - target) * mask_expand
    denom = mask_expand.flatten(1).sum(dim=1).clamp_min(1e-6)
    mse = diff.pow(2).flatten(1).sum(dim=1) / denom
    psnr_vals = 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))
    return psnr_vals


def masked_metric(metric_fn, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_expand = mask.expand_as(pred)
    masked_pred = (pred * mask_expand).contiguous()
    masked_target = (target * mask_expand).contiguous()
    values = metric_fn(masked_pred, masked_target, reduction="none")
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values, device=pred.device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tv_weight: float,
    noise_std: float,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        M = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
        I0 = corrupt_images(imgs, M, noise_std=noise_std)
        X = torch.cat([I0, M], dim=1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        pred = clamp_known(pred, imgs, M)

        l1_loss = F.l1_loss(pred, imgs)
        tv_loss = tv_l1(pred)
        loss = l1_loss + tv_weight * tv_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int,
    noise_std: float,
) -> Optional[float]:
    if max_images <= 0:
        return None

    model.eval()
    psnr_vals = []
    seen = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            M = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
            I0 = corrupt_images(imgs, M, noise_std=noise_std)
            X = torch.cat([I0, M], dim=1)
            pred = model(X)
            pred = clamp_known(pred, imgs, M)
            batch_psnr = psnr_metric(pred, imgs, reduction="none")
            remaining = max_images - seen
            if remaining <= 0:
                break
            batch_vals = batch_psnr.detach().cpu().tolist()
            psnr_vals.extend(batch_vals[:remaining])
            seen += min(len(batch_vals), remaining)
            if seen >= max_images:
                break
    if not psnr_vals:
        return None
    return sum(psnr_vals) / len(psnr_vals)


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    agg = {
        "psnr_all": 0.0,
        "psnr_miss": 0.0,
        "ssim_all": 0.0,
        "ssim_miss": 0.0,
        "lpips_all": 0.0,
        "lpips_miss": 0.0,
    }
    total_images = 0
    sample_tensors: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            M = random_mask(imgs, p_missing=(0.25, 0.5), block_prob=0.5, min_blocks=1, max_blocks=3)
            I0 = corrupt_images(imgs, M, noise_std=noise_std)
            X = torch.cat([I0, M], dim=1)
            pred = model(X)
            pred = clamp_known(pred, imgs, M)

            miss = 1.0 - M

            batch_metrics = {
                "psnr_all": psnr_metric(pred, imgs, reduction="none"),
                "psnr_miss": masked_psnr(pred, imgs, miss),
                "ssim_all": ssim_metric(pred, imgs, reduction="none"),
                "ssim_miss": masked_metric(ssim_metric, pred, imgs, miss),
                "lpips_all": lpips_dist(pred.contiguous(), imgs.contiguous(), reduction="none"),
                "lpips_miss": masked_metric(lpips_dist, pred, imgs, miss),
            }

            batch_size = imgs.size(0)
            for key, value in batch_metrics.items():
                tensor_val = value.detach()
                if tensor_val.ndim == 0:
                    agg[key] += float(tensor_val.cpu()) * batch_size
                else:
                    agg[key] += float(tensor_val.sum().cpu())
            total_images += batch_size

            if not sample_tensors:
                sample_tensors = {
                    "gt": imgs.detach().cpu(),
                    "input": I0.detach().cpu(),
                    "mask": M.detach().cpu(),
                    "pred": pred.detach().cpu(),
                }

    if total_images == 0:
        raise RuntimeError("Empty loader provided for evaluation")

    for key in agg:
        agg[key] /= total_images
    return agg, sample_tensors


def save_train_log(path: str, logs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def plot_psnr_curve(psnr_values, save_path: str) -> None:
    if not psnr_values:
        return
    epochs = list(range(1, len(psnr_values) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, psnr_values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR per Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_samples(sample_tensors: Dict[str, torch.Tensor], save_path: str) -> None:
    if not sample_tensors:
        return
    gt = sample_tensors["gt"]
    inp = sample_tensors["input"]
    mask = sample_tensors["mask"]
    pred = sample_tensors["pred"]

    num_rows = min(6, gt.size(0))
    panels = []
    for i in range(num_rows):
        gt_img = gt[i]
        inp_img = inp[i]
        mask_img = mask[i].expand_as(gt_img)
        pred_img = pred[i]

        panels.extend([gt_img, inp_img, mask_img * 2 - 1, pred_img])

    grid = vutils.make_grid(panels, nrow=4, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, save_path)


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_dataloaders(args, device)

    model = instantiate_unet(args.base)
    model.to(device)
    params = param_count(model)
    print(f"Model parameters: {params}")
    if args.target_params > 0:
        lower = args.target_params * 0.95
        upper = args.target_params * 1.05
        if not (lower <= params <= upper):
            print(
                f"[Warning] Parameter count {params} outside ±5% of target {args.target_params}."
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_history = []
    logs = []
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.tv_weight,
            args.noise_std,
            args.grad_clip,
        )
        elapsed = time.time() - start

        val_psnr = evaluate_subset(model, test_loader, device, args.eval_subset, args.noise_std)
        if val_psnr is not None:
            val_history.append(val_psnr)
            if val_psnr > best_metric:
                best_metric = val_psnr
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

        logs.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "psnr_val": val_psnr,
                "time_sec": elapsed,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f}"
            + (f", val_psnr: {val_psnr:.2f}" if val_psnr is not None else "")
            + f", time: {elapsed:.1f}s"
        )

    if best_metric == -float("inf"):
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    metrics, sample_tensors = compute_metrics(model, test_loader, device, args.noise_std)
    metrics["params"] = int(params)
    metrics["seed"] = args.seed

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    ckpt = {
        "model": best_state,
        "args": vars(args),
        "params": int(params),
        "best_metric": float(metrics["psnr_all"]),
    }
    torch.save(ckpt, os.path.join(args.save_dir, "ckpt.pt"))

    save_train_log(os.path.join(args.save_dir, "train_log.json"), logs)
    plot_psnr_curve(val_history, os.path.join(args.save_dir, "psnr_curve.png"))
    save_samples(sample_tensors, os.path.join(args.save_dir, "pred_samples.png"))


if __name__ == "__main__":
    main()
