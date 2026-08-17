"""
Standalone evaluation / inference entry point for the KLA PS01 SR-UNet.

Usage
-----
    python evaluate.py --input-dir <DIR> --output-dir <DIR> [--checkpoint PATH] [--gt-dir DIR]

This script is the submission-ready replacement for ``scripts/predict.py``:

  * accepts input/output directories as CLI arguments (no source-code edits),
  * loads the model and checkpoint itself (default: ``checkpoints/best.pth``),
  * preserves the exact preprocessing/postprocessing conventions of the
    trained pipeline (see ``scripts/predict.py``), including the 2x super-resolution
    and the [0, 1] output clamp,
  * reports PSNR, SSIM and LPIPS when ground-truth images are provided via
    ``--gt-dir`` (matched by filename), and
  * runs on CPU when CUDA is unavailable; AMP is used only when CUDA is present.

Input contract (verified against ``src/data/dataset.py`` and
``scripts/predict.py``):

  * grayscale / single-channel images stored as raw 2-D ``.npy`` arrays,
  * float32 or float64 values in approximately [0, 1] (values outside the
    range are tolerated on input, as produced by speckle noise),
  * output is written as 2-D float32 ``.npy`` with the same basename as the
    input, at 2x the input resolution.

LPIPS (``--gt-dir`` only): uses the ``lpips`` package with the ``alex``
backbone. The pretrained weights are downloaded automatically on first use
and cached; if they cannot be fetched (e.g. offline environment), the script
fails with a clear error instead of producing a synthetic number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from torch.amp import autocast
except ImportError:  # PyTorch < 2.3
    from torch.cuda.amp import autocast

try:
    from torch import inference_mode
except ImportError:
    from torch import no_grad as inference_mode

from src import config
from src.models.sr_unet import SRUNet
from src.models.sr_unet_v2 import SRUNetV2
from src.metrics import SSIM, psnr

MODEL_REGISTRY = {"srunet": SRUNet, "srunet_v2": SRUNetV2}

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pth"


# --------------------------------------------------------------------------- #
# Model / checkpoint loading
# --------------------------------------------------------------------------- #
def load_model(
    device: torch.device,
    checkpoint_path: Path | str,
    model_name: str = "srunet",
) -> SRUNet:
    """Instantiate the requested model and load the checkpoint onto ``device``."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Use --checkpoint to point at a valid checkpoint file."
        )

    model = MODEL_REGISTRY[model_name]().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_models(
    device: torch.device,
    checkpoint_paths: list[Path | str],
    model_name: str = "srunet",
) -> list[SRUNet]:
    """Load one model per checkpoint for (weighted-average) ensembling."""
    return [load_model(device, path, model_name) for path in checkpoint_paths]


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _to_metric_tensors(restored: np.ndarray, gt: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack two 2-D float arrays into (1, 1, H, W) tensors on CPU."""
    restored_t = torch.from_numpy(np.asarray(restored, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    return restored_t, gt_t


def _lpips_fn(device: torch.device):
    """Build the LPIPS (AlexNet) metric, with a clear offline failure mode."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS is not installed. Install it with 'pip install lpips' "
            "to use --gt-dir evaluation."
        ) from exc

    try:
        return lpips.LPIPS(net="alex").to(device)
    except Exception as exc:
        raise RuntimeError(
            "LPIPS failed to initialize (pretrained weights unavailable?). "
            "LPIPS requires a one-time download of AlexNet weights; check "
            "network access and retry. No synthetic metric is produced."
        ) from exc


def _lpips_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Prepare a 2-D grayscale array for the LPIPS network.

    LPIPS expects (N, 3, H, W) RGB-like tensors in [-1, 1]. Grayscale is
    replicated across the three channels and mapped from [0, 1] -> [-1, 1].
    """
    gray = np.asarray(image, dtype=np.float32)
    rgb = np.repeat(gray[None, None, :, :], 3, axis=1)
    tensor = torch.from_numpy(rgb) * 2.0 - 1.0
    return tensor


def evaluate_pair(
    restored: np.ndarray,
    gt: np.ndarray,
    ssim_module: SSIM,
    lpips_fn,
    device: torch.device,
):
    """Return (psnr, ssim, lpips) for one restored/GT pair."""
    pred_t, gt_t = _to_metric_tensors(restored, gt)

    psnr_val = float(psnr(pred_t, gt_t).item())
    ssim_val = float(ssim_module(pred_t, gt_t).item())

    lpips_val = None
    if lpips_fn is not None:
        with torch.no_grad():
            pred_l = _lpips_tensor(restored).to(device)
            gt_l = _lpips_tensor(gt).to(device)
            lpips_val = float(lpips_fn(pred_l, gt_l).item())

    return psnr_val, ssim_val, lpips_val


def _predict(model, input_tensor, tta: bool, use_amp: bool, device) -> torch.Tensor:
    """Run one model on one image (optionally with 4-flip TTA)."""
    if tta:
        flips = [
            [],          # Identity
            [-1],        # Horizontal flip
            [-2],        # Vertical flip
            [-1, -2],    # Horizontal + Vertical flip
        ]
        predictions = []
        for flip_dims in flips:
            augmented_input = (
                torch.flip(input_tensor, dims=flip_dims)
                if flip_dims
                else input_tensor
            )
            with autocast(device_type=device.type, enabled=use_amp):
                augmented_pred = model(augmented_input)
            pred = (
                torch.flip(augmented_pred, dims=flip_dims)
                if flip_dims
                else augmented_pred
            )
            predictions.append(pred)
        return torch.stack(predictions).mean(dim=0)
    with autocast(device_type=device.type, enabled=use_amp):
        return model(input_tensor)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Standalone SR-UNet evaluation/inference for the KLA PS01 image "
            "restoration track. Restores every .npy in --input-dir at 2x "
            "resolution into --output-dir; reports PSNR/SSIM/LPIPS per image "
            "and on average when --gt-dir is provided."
        ),
        epilog=(
            "Example:\n"
            "  python evaluate.py --input-dir dataset/test/Test_NoisyLR/NoisyLR "
            "--output-dir outputs\n"
            "  python evaluate.py --input-dir <noisy> --output-dir <restored> "
            "--gt-dir <clean> --checkpoint checkpoints/best.pth"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing grayscale 2-D .npy degraded images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where restored .npy images will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to the trained checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help=(
            "Optional directory of ground-truth .npy images (same basenames "
            "as inputs) to compute PSNR, SSIM and LPIPS."
        ),
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help=(
            "Use 4-flip test-time augmentation (original, H, V, H+V flips "
            "averaged). Disabled by default; deterministic single-pass "
            "inference otherwise."
        ),
    )
    parser.add_argument(
        "--ensemble",
        default=None,
        metavar="CKPT[,CKPT,...]",
        help=(
            "Comma-separated checkpoint paths whose predictions are averaged "
            "(with --tta, each model's 4-flip prediction is averaged first). "
            "Defaults to the single --checkpoint model."
        ),
    )
    parser.add_argument(
        "--model",
        default="srunet",
        choices=sorted(MODEL_REGISTRY),
        help="Architecture matching the checkpoint: 'srunet' (default) or 'srunet_v2'.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        parser.error(f"No .npy files found in input directory: {input_dir}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint file does not exist: {checkpoint_path}")

    ensemble_paths = None
    if args.ensemble:
        ensemble_paths = [Path(p.strip()) for p in args.ensemble.split(",") if p.strip()]
        for p in ensemble_paths:
            if not p.is_file():
                parser.error(f"Ensemble checkpoint file does not exist: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.USE_AMP and device.type == "cuda"

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is not None and not gt_dir.is_dir():
        parser.error(f"Ground-truth directory does not exist: {gt_dir}")

    print(f"Device              : {device} (AMP: {'on' if use_amp else 'off'})")
    print(f"TTA                 : {'on' if args.tta else 'off'}")
    if ensemble_paths:
        print(f"Ensemble            : {len(ensemble_paths)} checkpoints")
        for p in ensemble_paths:
            print(f"  - {p}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Input dir           : {input_dir} ({len(files)} .npy files)")
    print(f"Output dir         : {output_dir}")
    print(f"GT dir              : {gt_dir if gt_dir else '(none — inference only)'}")
    print("-" * 78)

    if ensemble_paths:
        models = load_models(device, ensemble_paths, args.model)
    else:
        models = [load_model(device, checkpoint_path, args.model)]

    ssim_module = SSIM()
    lpips_fn = None
    if gt_dir is not None:
        lpips_fn = _lpips_fn(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    lpips_scores: list[float] = []
    evaluated = 0
    skipped_gt = 0
    timings_ms: list[float] = []

    with inference_mode():
        for i, file_path in enumerate(files):
            start = time.perf_counter()

            noisy_lr = np.load(file_path).astype(np.float32)
            if noisy_lr.ndim != 2:
                print(
                    f"SKIP {file_path.name}: expected 2-D grayscale .npy, "
                    f"got shape {noisy_lr.shape} (this model is single-channel)."
                )
                continue

            input_tensor = (
                torch.from_numpy(noisy_lr).unsqueeze(0).unsqueeze(0).to(device)
            )
            if device.type == "cuda":
                input_tensor = input_tensor.to(memory_format=torch.channels_last)

            model_preds = []
            for model in models:
                model_preds.append(_predict(model, input_tensor, args.tta, use_amp, device))
            pred_hr = torch.stack(model_preds).mean(dim=0).clamp(0.0, 1.0)
            restored = pred_hr.squeeze().cpu().numpy().astype(np.float32)

            np.save(output_dir / file_path.name, restored)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

            line = f"[{i + 1:4d}/{len(files)}] {file_path.name:>16s} -> "
            line += f"{restored.shape[1]}x{restored.shape[0]} "
            line += f"({elapsed_ms:7.1f} ms)"

            if gt_dir is not None:
                gt_path = gt_dir / file_path.name
                if not gt_path.is_file():
                    skipped_gt += 1
                    line += "  GT MISSING"
                elif np.asarray(np.load(gt_path)).shape != restored.shape:
                    skipped_gt += 1
                    line += "  GT SHAPE MISMATCH"
                else:
                    gt = np.load(gt_path).astype(np.float32)
                    p, s, lp = evaluate_pair(
                        restored, gt, ssim_module, lpips_fn, device
                    )
                    evaluated += 1
                    psnr_scores.append(p)
                    ssim_scores.append(s)
                    if lp is not None:
                        lpips_scores.append(lp)
                    line += f"  PSNR {p:6.3f}  SSIM {s:.4f}"
                    if lp is not None:
                        line += f"  LPIPS {lp:.4f}"

            print(line)

    print("-" * 78)
    if evaluated:
        n = evaluated
        print(
            f"Evaluated {n} images: "
            f"avg PSNR {float(np.mean(psnr_scores)):.3f} dB | "
            f"avg SSIM {float(np.mean(ssim_scores)):.4f}"
        )
        if lpips_scores:
            print(
                f"                     "
                f"avg LPIPS {float(np.mean(lpips_scores)):.4f} (lower is better)"
            )
    else:
        print(f"Inference only: restored {len(files)} images, no metrics computed.")

    if skipped_gt:
        print(f"Note: {skipped_gt} images skipped for metrics (missing/mismatched GT).")

    if timings_ms:
        print(
            f"Average inference time: {float(np.mean(timings_ms)):.1f} ms/image "
            f"(first load excluded; device: {device.type})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())