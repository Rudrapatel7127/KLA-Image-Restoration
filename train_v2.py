"""
PS01 SR-UNet Training v2 (experimental, self-contained)
========================================================
Rebuild/improve training for the PS01 image-restoration pipeline without
touching the original ``src/training/train.py``.

Differences vs the original recipe:
  * CLI-driven hyper-parameters (--epochs, --lr, --loss, --aug, --seed, ...).
  * Deterministic train/validation split IDENTICAL to the original
    (seed 42, 5% = 160 images) so validation PSNR/SSIM are directly
    comparable with ``checkpoints/best.pth`` (epoch 24: 25.8306 / 0.7223).
  * Optional warm start from an existing checkpoint (e.g. best.pth) for
    fine-tuning past the observed plateau.
  * Richer paired augmentation: flips + 90-degree rotations + light
    additive Gaussian / multiplicative speckle noise applied to the noisy
    input only (GT stays clean). Geometry transforms are always paired.
  * Configurable loss: L1+SSIM (default, matches original alpha=0.84),
    Charbonnier+SSIM, plain L1, plain Charbonnier.
  * LPIPS added to validation (every --val-lpips-every epochs) on GPU.

Run (CUDA venv): .venv-cuda/Scripts/python.exe -B -m train_v2 --ckpt-dir checkpoints/finetune_v1
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from src import config
from src.data.dataset import WaferDataset
from src.metrics import SSIM, mse, psnr
from src.models.sr_unet import SRUNet
from src.models.sr_unet_v2 import SRUNetV2
from src.models.two_stage import TwoStageModel
from src.models.residual_noise import ResidualNoiseSR
from src.models.noise_conditioned import NoiseConditionedSRUNet

MODEL_REGISTRY = {
    "srunet": SRUNet,
    "srunet_v2": SRUNetV2,
    "two_stage": TwoStageModel,
    "residual_noise": ResidualNoiseSR,
    "noise_conditioned": NoiseConditionedSRUNet,
}

# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class Charbonnier(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps**2))


class CharbonnierSSIM(nn.Module):
    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha
        self.charb = Charbonnier()
        self.ssim_metric = SSIM()

    def forward(self, pred, target):
        return self.alpha * self.charb(pred, target) + (1.0 - self.alpha) * (1.0 - self.ssim_metric(pred, target))


class NoiseWeightedCharbonnierSSIM(nn.Module):
    """Charbonnier + SSIM with a noise-aware weight on the reconstruction term.

    Dataset measurements show multiplicative noise: residual std grows almost
    linearly with intensity (std ~ 0.15 * intensity). Weighting the residual by
    ~1/sqrt(intensity) (variance-stabilizing for Poisson-like noise) spends
    capacity where the noise is strongest without smoothing bright detail.
    """

    def __init__(self, alpha: float = 0.84, noise_weight: float = 1.0, k: float = 0.05, max_w: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.noise_weight = noise_weight
        self.charb = Charbonnier()
        self.ssim_metric = SSIM()
        self.k = k
        self.max_w = max_w

    def forward(self, pred, target):
        diff = pred - target
        w = (1.0 / torch.sqrt(target.detach() + self.k)).clamp(max=self.max_w)
        charb_w = torch.mean(torch.sqrt(diff * diff + 1e-6) * w)
        return self.alpha * charb_w + (1.0 - self.alpha) * (1.0 - self.ssim_metric(pred, target))


class L1SSIM(nn.Module):
    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim_metric = SSIM()

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + (1.0 - self.alpha) * (1.0 - self.ssim_metric(pred, target))


def build_criterion(name: str, alpha: float, noise_weight: float = 0.0) -> nn.Module:
    name = name.lower()
    if name in ("l1ssim", "l1_ssim", "l1+ssim"):
        return L1SSIM(alpha)
    if name in ("charbonnier_ssim", "charb_ssim"):
        if noise_weight > 0:
            return NoiseWeightedCharbonnierSSIM(alpha, noise_weight)
        return CharbonnierSSIM(alpha)
    if name == "l1":
        return nn.L1Loss()
    if name in ("charbonnier", "charb"):
        return Charbonnier()
    raise ValueError(f"Unknown loss: {name}")


# --------------------------------------------------------------------------- #
# Paired augmentation (applied inside the dataset to keep pairs in sync)
# --------------------------------------------------------------------------- #
class PairedAug:
    """Geometric + intensity-safe noise augmentation. Geometry is applied to
    both noisy and GT identically; noise is applied to the noisy input only."""

    def __init__(self, mode: str, noise_scale: float, bright_max: float = 2.2, noise_proportional: bool = False):
        self.mode = mode
        self.noise_scale = noise_scale
        self.bright_max = bright_max
        self.noise_proportional = noise_proportional

    def __call__(self, noisy, gt):
        if self.mode == "none":
            return noisy, gt

        k = torch.randint(0, 4, (1,)).item()
        if k:
            noisy = torch.rot90(noisy, k, dims=[1, 2])
            gt = torch.rot90(gt, k, dims=[1, 2])

        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if torch.rand(1).item() < 0.5:
            noisy = torch.flip(noisy, dims=[1])
            gt = torch.flip(gt, dims=[1])

        if "noise" in self.mode:
            # Degradation-aware augmentation on the noisy input ONLY (the GT
            # stays the clean reference in [0, 1]).
            if self.noise_scale > 0:
                jitter = 0.5 + torch.rand(1).item()
                if self.noise_proportional:
                    # Measured degradation: residual std ~ 0.15 * intensity.
                    sigma = self.noise_scale * noisy.abs() + 0.005
                    noisy = noisy + torch.randn_like(noisy) * (sigma * jitter)
                else:
                    gauss = torch.randn_like(noisy) * (self.noise_scale * torch.rand(1).item())
                    speckle = 1.0 + (torch.rand(1).item() - 0.5) * 0.2 * (self.noise_scale * 10)
                    noisy = noisy * speckle + gauss
            if self.bright_max > 1.0:
                # Brightness augmentation: train inputs range up to ~1.7 but
                # test inputs reach ~2.16; teach the model the full domain.
                b = 0.9 + (self.bright_max - 0.9) * torch.rand(1).item()
                noisy = noisy * b
        return noisy, gt


# --------------------------------------------------------------------------- #
# Dataset wrapper (deterministic, mirrors original file ordering)
# --------------------------------------------------------------------------- #
class PairedWaferDataset(Dataset):
    def __init__(self, root, aug_mode: str, noise_scale: float, bright_max: float = 2.2, noise_proportional: bool = False):
        base = WaferDataset(root, augment=False)  # reuse ordering + validation
        self.files = base.files
        self.gt_dir = base.gt_dir
        self.noisy_dir = base.noisy_dir
        self.aug = PairedAug(aug_mode, noise_scale, bright_max, noise_proportional)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        gt_path = self.files[idx]
        noisy_path = self.noisy_dir / gt_path.name
        gt = torch.from_numpy(np.load(gt_path).astype(np.float32)).unsqueeze(0)
        noisy = torch.from_numpy(np.load(noisy_path).astype(np.float32)).unsqueeze(0)
        return self.aug(noisy, gt)


def build_split(root, val_split, seed):
    """Deterministic split, IDENTICAL to original train.py (seed 42)."""
    ds = PairedWaferDataset(root, aug_mode="none", noise_scale=0.0)
    val_size = int(len(ds) * val_split)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=generator).tolist()
    train_idx, val_idx = indices[: len(ds) - val_size], indices[len(ds) - val_size :]
    return train_idx, val_idx


def build_aug_dataset(root, aug_mode, noise_scale, bright_max, noise_proportional=False):
    ds = PairedWaferDataset(root, aug_mode=aug_mode, noise_scale=noise_scale, bright_max=bright_max, noise_proportional=noise_proportional)
    return ds


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_v2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="PS01 SR-UNet training v2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model", type=str, default="srunet", choices=sorted(MODEL_REGISTRY), help="architecture: srunet (default) or srunet_v2 (stronger SR head)")
    ap.add_argument("--warm-start", type=str, default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--warm-strict", type=str, default="true", choices=["true", "false"], help="strict=True load for warm start; 'false' transfers shared weights only (e.g. new arch head)")
    ap.add_argument("--loss", type=str, default="l1ssim")
    ap.add_argument("--alpha", type=float, default=0.84)
    ap.add_argument("--aug", type=str, default="flip_rot_noise", choices=["none", "flip_rot", "flip_rot_noise"])
    ap.add_argument("--noise-scale", type=float, default=0.015)
    ap.add_argument("--noise-proportional", action="store_true", help="augment with per-pixel sigma = noise_scale * |x| + 0.005 (matches measured multiplicative noise)")
    ap.add_argument("--noise-weight", type=float, default=0.0, help=">0 enables variance-stabilizing residual weighting in Charbonnier+SSIM (k=0.05, max w=4)")
    ap.add_argument("--bright-max", type=float, default=2.2, help="max brightness factor for input augmentation (test inputs reach ~2.16)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/finetune_v1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--val-lpips-every", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="resume from latest.pth in ckpt-dir")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path("logs") / f"train_v2_{ckpt_dir.name}.log")

    print("=" * 60)
    print("PS01 SR-UNet Training v2")
    print("=" * 60)
    logger.info("PyTorch %s | device %s", torch.__version__, device)
    logger.info("args: %s", vars(args))

    try:
        set_seed(args.seed)
        train_idx, val_idx = build_split(config.TRAIN_ROOT, args.val_split, args.seed)
        train_ds = build_aug_dataset(config.TRAIN_ROOT, args.aug, args.noise_scale, args.bright_max, args.noise_proportional)
        train_ds_aug = Subset(train_ds, train_idx)
        val_ds = Subset(build_aug_dataset(config.TRAIN_ROOT, "none", 0.0, 1.0), val_idx)
        logger.info("Train %d | Val %d (deterministic split seed=%d)", len(train_idx), len(val_idx), args.seed)

        common = dict(batch_size=args.batch, num_workers=args.num_workers, pin_memory=False)
        train_loader = DataLoader(train_ds_aug, shuffle=True, drop_last=True, **common)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)

        model = MODEL_REGISTRY[args.model]().to(device)
        logger.info("Parameters: %.2fM", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

        start_epoch = 0
        best_psnr = -1.0
        optimizer_state = None
        scheduler_state = None
        if args.warm_start:
            w = torch.load(args.warm_start, map_location=device, weights_only=False)
            sd = w["model_state_dict"] if "model_state_dict" in w else w
            if args.warm_strict == "false":
                model_sd = model.state_dict()
                n_before = len(sd)
                sd = {
                    k: v for k, v in sd.items()
                    if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
                }
                logger.info(
                    "Transferred %d/%d tensors (shape-matched only, strict=False)",
                    len(sd), n_before,
                )
            model.load_state_dict(sd, strict=False)
            logger.info("Warm start from %s (epoch %s, strict=%s)", args.warm_start, w.get("epoch", "?"), args.warm_strict)
        if args.resume:
            rp = ckpt_dir / "latest.pth"
            if rp.exists():
                w = torch.load(rp, map_location=device, weights_only=False)
                model.load_state_dict(w["model_state_dict"])
                start_epoch = int(w["epoch"])
                best_psnr = float(w.get("best_psnr", -1.0))
                optimizer_state = w.get("optimizer_state_dict")
                scheduler_state = w.get("scheduler_state_dict")
                logger.info("Resumed from %s at epoch %d (best %.3f)", rp, start_epoch, best_psnr)

        criterion = build_criterion(args.loss, args.alpha, args.noise_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        lpips_fn = None
        if args.val_lpips_every > 0:
            try:
                import lpips
                lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
            except Exception as e:
                logger.warning("LPIPS unavailable: %s", e)

        t0 = time.time()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch", leave=True, dynamic_ncols=True)
            for noisy, gt in pbar:
                noisy, gt = noisy.to(device), gt.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    pred = model(noisy)
                    loss = criterion(pred, gt)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                running += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})
            pbar.close()
            avg_loss = running / max(len(train_loader), 1)
            scheduler.step()

            model.eval()
            vpsnr, vssim, vmse = 0.0, 0.0, 0.0
            ssim_m = SSIM().to(device)
            with torch.no_grad():
                for noisy, gt in val_loader:
                    noisy, gt = noisy.to(device), gt.to(device)
                    with autocast(device_type=device.type, enabled=use_amp):
                        pred = model(noisy)
                    vpsnr += psnr(pred, gt).item()
                    vssim += ssim_m(pred, gt).item()
                    vmse += mse(pred, gt).item()
            vpsnr /= max(len(val_loader), 1)
            vssim /= max(len(val_loader), 1)
            vmse /= max(len(val_loader), 1)

            vlpips = None
            if lpips_fn is not None and (epoch + 1) % args.val_lpips_every == 0:
                with torch.no_grad():
                    vals = []
                    for noisy, gt in val_loader:
                        noisy, gt = noisy.to(device), gt.to(device)
                        vals.append(lpips_fn(model(noisy), gt).mean().item())
                vlpips = sum(vals) / len(vals)

            msg = (f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.5f} | "
                   f"PSNR {vpsnr:.3f} | SSIM {vssim:.4f} | MSE {vmse:.6f}"
                   + (f" | LPIPS {vlpips:.4f}" if vlpips is not None else ""))
            print(msg)
            logger.info(msg)

            latest_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_psnr": best_psnr,
                "loss": avg_loss,
                "psnr": vpsnr,
                "ssim": vssim,
                "mse": vmse,
                "lpips": vlpips,
                "args": vars(args),
            }
            torch.save(latest_payload, ckpt_dir / "latest.pth")

            if (epoch + 1) % args.save_every == 0:
                torch.save(latest_payload, ckpt_dir / f"epoch_{epoch+1}.pth")
                logger.info("Saved %s", ckpt_dir / f"epoch_{epoch+1}.pth")

            if vpsnr > best_psnr:
                best_psnr = vpsnr
                torch.save(latest_payload, ckpt_dir / "best.pth")
                logger.info("New best (PSNR %.3f) -> %s", vpsnr, ckpt_dir / "best.pth")

        total_min = (time.time() - t0) / 60
        logger.info("Training done in %.2f min. Best val PSNR %.3f dB (SSIM %.4f)", total_min, best_psnr, vssim)
        summary = ckpt_dir / "training_summary.txt"
        summary.write_text(
            f"Total time: {total_min:.2f} min\n"
            f"PyTorch: {torch.__version__}\n"
            f"Device: {device}\n"
            f"GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}\n"
            f"Best PSNR: {best_psnr:.3f} dB\n"
            f"Best SSIM: {vssim:.4f}\n"
            f"Loss: {args.loss} (alpha {args.alpha}, noise_weight {args.noise_weight})\n"
            f"Aug: {args.aug} (noise {args.noise_scale}, proportional {args.noise_proportional})\n"
            f"Seed: {args.seed} | warm start: {args.warm_start}\n",
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
