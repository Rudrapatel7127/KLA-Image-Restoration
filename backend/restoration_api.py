"""
PS01 KLA Image Restoration — FastAPI backend.

Dedicated PS01 inference API. Uses the validated E1 checkpoint
(experiments/E1_charb/checkpoints/best.pth) with 4-flip TTA on CUDA.

Run from the project root (GPU env):

    .venv-cuda\\Scripts\\python.exe -B -m uvicorn backend.restoration_api:app \
        --host 127.0.0.1 --port 8002

This module is PS01-only. It never touches PS02 files.
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.metrics import SSIM, psnr

CHECKPOINT = ROOT / "experiments" / "E1_charb" / "checkpoints" / "best.pth"
TTA_FLIPS = [[], [-1], [-2], [-1, -2]]
MODEL_NAME = "E1 Charbonnier + SSIM"
EXPECTED_INPUT_SHAPES = {(128, 128), (256, 256)}

_gpu_available = torch.cuda.is_available()
_model: SRUNet | None = None
_use_amp = config.USE_AMP and _gpu_available


def gpu_report() -> dict:
    if not _gpu_available:
        return {"cuda_available": False, "device_name": None}
    return {"cuda_available": True, "device_name": torch.cuda.get_device_name(0)}


def load_model() -> None:
    global _model
    if not _gpu_available:
        raise RuntimeError(
            "CUDA is not available on this machine. PS01 inference must run on the GPU; "
            "refusing to silently fall back to CPU."
        )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"E1 checkpoint not found at '{CHECKPOINT}'.")
    model = SRUNet().to("cuda")
    model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    _model = model


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
        print(f"[restoration_api] model loaded on {gpu_report()['device_name']}")
    except Exception as exc:  # noqa: BLE001 - surface in /api/health
        print(f"[restoration_api] startup model load FAILED: {exc}")
    yield


app = FastAPI(
    title="PS01 KLA Image Restoration API",
    description="GPU restoration of noisy low-resolution .npy wafer images (E1 checkpoint, 4-flip TTA).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "checkpoint": str(CHECKPOINT),
        "device": gpu_report(),
        "tta": "4-flip",
        "model": MODEL_NAME,
        "error": None if _model is not None else "CUDA unavailable or checkpoint missing — see server log.",
    }


def _read_npy_2d(raw_bytes: bytes, field: str) -> np.ndarray:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail=f"{field}: uploaded file is empty.")
    try:
        arr = np.load(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field}: failed to read .npy file ({exc}).") from exc
    if arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"{field}: expected a 2-D grayscale array (H, W), got shape {arr.shape}.",
        )
    if arr.dtype not in (np.float32, np.float64, np.uint8, np.uint16):
        raise HTTPException(
            status_code=400,
            detail=f"{field}: unsupported dtype {arr.dtype} (expected float32).",
        )
    return arr


@app.post("/api/restore")
async def restore(file: UploadFile = File(...)) -> Response:
    """Restore a noisy LR .npy at 2x with the E1 checkpoint and 4-flip TTA on GPU."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded (CUDA unavailable or checkpoint missing).",
        )

    filename = file.filename or "upload.npy"
    if not filename.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported.")

    arr = _read_npy_2d(await file.read(), "input")
    in_shape = tuple(int(d) for d in arr.shape)
    if in_shape not in EXPECTED_INPUT_SHAPES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected a {sorted(EXPECTED_INPUT_SHAPES)} input, got {in_shape}.",
        )

    start = time.perf_counter()
    with inference_mode():
        x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to("cuda")
        x = x.to(memory_format=torch.channels_last)
        preds: list[torch.Tensor] = []
        for dims in TTA_FLIPS:
            xi = torch.flip(x, dims=dims) if dims else x
            with autocast(device_type="cuda", enabled=_use_amp):
                pi = _model(xi)
            preds.append(torch.flip(pi, dims=dims) if dims else pi)
        pred = torch.stack(preds).mean(dim=0).clamp(0.0, 1.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    restored = pred.squeeze().cpu().numpy().astype(np.float32)
    out_shape = tuple(int(d) for d in restored.shape)

    buf = io.BytesIO()
    np.save(buf, restored)
    buf.seek(0)
    out_name = f"restored_{Path(filename).stem}.npy"

    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Processing-Time-Ms": f"{elapsed_ms:.3f}",
            "X-Input-Shape": ",".join(str(d) for d in in_shape),
            "X-Output-Shape": ",".join(str(d) for d in out_shape),
            "X-Device": "cuda",
            "X-GPU": gpu_report()["device_name"] or "none",
            "X-TTA": "4-flip",
            "X-Checkpoint": str(CHECKPOINT),
        },
    )


@app.post("/api/metrics")
async def metrics(
    restored: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
) -> dict:
    """Compute real PSNR / SSIM / LPIPS between a restored .npy and a GT .npy."""
    r = _read_npy_2d(await restored.read(), "restored")
    g = _read_npy_2d(await ground_truth.read(), "ground truth")
    if r.shape != g.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Shape mismatch: restored {r.shape} vs ground truth {g.shape}.",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_t = torch.from_numpy(r.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    g_t = torch.from_numpy(g.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    psnr_val = float(psnr(r_t, g_t).item())
    ssim_val = float(SSIM()(r_t, g_t).item())

    lpips_val = None
    try:
        import lpips

        lp = lpips.LPIPS(net="alex").to(device)
        rgb_r = torch.from_numpy(np.repeat(r.astype(np.float32)[None, None], 3, axis=1)) * 2.0 - 1.0
        rgb_g = torch.from_numpy(np.repeat(g.astype(np.float32)[None, None], 3, axis=1)) * 2.0 - 1.0
        with torch.no_grad():
            lpips_val = float(lp(rgb_r.to(device), rgb_g.to(device)).item())
    except Exception as exc:  # noqa: BLE001 - LPIPS is best-effort here
        print(f"[restoration_api] LPIPS unavailable: {exc}")

    return {
        "psnr": round(psnr_val, 4),
        "ssim": round(ssim_val, 4),
        "lpips": round(lpips_val, 4) if lpips_val is not None else None,
        "shape": list(r.shape),
        "device": str(device),
    }


@app.post("/api/restore-region")
async def restore_region(file: UploadFile = File(...)) -> Response:
    """2x-restore a small detected-region crop (e.g. 99x99 px) on the GPU.

    Used by the PS02 localization UI to show a sharper, higher-resolution
    view of the magnified detected region. Accepts PNG/JPEG, processes each
    RGB channel through the E1 SR-UNet with 4-flip TTA, returns a PNG.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Restoration model is not loaded (CUDA unavailable or E1 checkpoint missing).",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    rgb = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    h, w = rgb.shape[:2]
    if h < 16 or w < 16 or h > 512 or w > 512:
        raise HTTPException(
            status_code=400,
            detail=f"Region must be 16..512 px per side, got {w}x{h}.",
        )

    ph = int(np.ceil(h / 8.0)) * 8
    pw = int(np.ceil(w / 8.0)) * 8
    padded = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")

    start = time.perf_counter()
    channels: list[np.ndarray] = []
    with inference_mode():
        for c in range(3):
            x = torch.from_numpy(padded[:, :, c]).unsqueeze(0).unsqueeze(0).to("cuda")
            x = x.to(memory_format=torch.channels_last)
            preds: list[torch.Tensor] = []
            for dims in TTA_FLIPS:
                xi = torch.flip(x, dims=dims) if dims else x
                with autocast(device_type="cuda", enabled=_use_amp):
                    pi = _model(xi)
                preds.append(torch.flip(pi, dims=dims) if dims else pi)
            out = torch.stack(preds).mean(dim=0).clamp(0.0, 1.0)
            channels.append(out.squeeze().cpu().numpy()[: 2 * h, : 2 * w])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    restored_rgb = (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)
    out_buf = io.BytesIO()
    Image.fromarray(restored_rgb).save(out_buf, format="PNG")
    out_buf.seek(0)

    return Response(
        content=out_buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": f"{elapsed_ms:.3f}",
            "X-Input-Shape": f"{h},{w}",
            "X-Output-Shape": f"{2 * h},{2 * w}",
            "X-GPU": gpu_report()["device_name"] or "none",
            "X-TTA": "4-flip",
        },
    )
