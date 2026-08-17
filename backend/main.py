"""
FastAPI backend for the KLA Hackathon SR-UNet super-resolution service.

Run from the project root:

    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:

    http://localhost:8000/docs
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.inference import (
    PredictionResult,
    get_checkpoint_path,
    get_device,
    is_model_loaded,
    load_model,
    predict_numpy,
)
from backend.schemas import HealthResponse, PredictMetadata


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except FileNotFoundError as exc:
        # Allow the API to start so /health can report the missing checkpoint.
        print(f"[startup] {exc}")
    yield


app = FastAPI(
    title="KLA Hackathon SR-UNet API",
    description=(
        "Production-ready inference API for wafer super-resolution. "
        "Upload a noisy low-resolution `.npy` image and receive the restored output."
    ),
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


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return service health and model load status."""
    return HealthResponse(
        status="ok" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        checkpoint=str(get_checkpoint_path()),
        device=str(get_device()),
    )


@app.post(
    "/predict",
    tags=["inference"],
    summary="Run super-resolution on an uploaded wafer image",
    response_class=Response,
    responses={
        200: {
            "description": "Restored wafer image saved as a `.npy` file.",
            "content": {"application/octet-stream": {}},
            "headers": {
                "X-Processing-Time-Ms": {
                    "description": "Model inference time in milliseconds.",
                    "schema": {"type": "number"},
                },
                "X-Input-Shape": {
                    "description": "Input array shape as comma-separated dimensions.",
                    "schema": {"type": "string"},
                },
                "X-Output-Shape": {
                    "description": "Output array shape as comma-separated dimensions.",
                    "schema": {"type": "string"},
                },
            },
        },
        400: {"description": "Invalid upload or unsupported array shape."},
        503: {"description": "Model checkpoint is not loaded."},
    },
)
async def predict(file: UploadFile = File(..., description="Noisy LR wafer image (.npy)")) -> Response:
    """
    Accept a `.npy` wafer image, run SR-UNet inference, and return the restored `.npy` output.

    Timing and shape metadata are returned in response headers for easy frontend integration.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model is not loaded. Expected checkpoint at '{get_checkpoint_path()}'."
            ),
        )

    filename = file.filename or "upload.npy"
    if not filename.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only `.npy` files are supported.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        noisy_lr = np.load(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read `.npy` file: {exc}") from exc

    try:
        result = predict_numpy(noisy_lr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    output_name = _restored_filename(filename)
    output_buffer = io.BytesIO()
    np.save(output_buffer, result.restored)
    output_buffer.seek(0)

    return Response(
        content=output_buffer.getvalue(),
        media_type="application/octet-stream",
        headers=_prediction_headers(result, output_name),
    )


@app.get(
    "/predict/metadata",
    response_model=PredictMetadata,
    tags=["inference"],
    include_in_schema=False,
)
async def predict_metadata_example() -> PredictMetadata:
    """Schema helper for frontend clients that want structured metadata."""
    return PredictMetadata(
        filename="example.npy",
        processing_time_ms=0.0,
        input_shape=[256, 256],
        output_shape=[512, 512],
        device=str(get_device()),
        tta_enabled=False,
    )


def _restored_filename(original_name: str) -> str:
    stem = Path(original_name).stem
    return f"restored_{stem}.npy"


def _prediction_headers(result: PredictionResult, output_name: str) -> dict[str, str]:
    return {
        "Content-Disposition": f'attachment; filename="{output_name}"',
        "X-Processing-Time-Ms": f"{result.processing_time_ms:.3f}",
        "X-Input-Shape": ",".join(str(dim) for dim in result.input_shape),
        "X-Output-Shape": ",".join(str(dim) for dim in result.output_shape),
        "X-Device": result.device,
        "X-TTA-Enabled": str(result.tta_enabled).lower(),
    }
