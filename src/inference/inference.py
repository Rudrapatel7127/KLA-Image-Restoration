"""
Inference pipeline for the SRUNet 2x super-resolution model.

Loads a trained checkpoint, reconstructs the SRUNet architecture, runs
inference on a grayscale input image, and saves the super-resolved result.

Usage (CLI):
    python src/inference/infrence.py \
        --checkpoint checkpoints/best.pth \
        --input input.png \
        --output output.png

Usage (as a library):
    from src.inference.infrence import infer
    infer("checkpoints/best.pth", "input.png", "output.png")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------
# Make sure "src" is importable regardless of the current working directory
# (i.e. whether this script is run as `python src/inference/infrence.py`
# from the project root, or from somewhere else entirely).
# --------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.models.sr_unet import SRUNet
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "Could not import SRUNet from src.models.sr_unet. Make sure this "
        "script is run from within the project (or that the project root "
        "is on PYTHONPATH), and that src/models/sr_unet.py defines a class "
        "named SRUNet."
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sr_inference")


# ==========================================================================
# Custom exceptions — makes error handling explicit and testable instead of
# relying on generic exceptions bubbling up from deep inside torch/PIL.
# ==========================================================================
class CheckpointError(Exception):
    """Raised when a checkpoint file cannot be found or parsed."""


class ImageError(Exception):
    """Raised when an input image cannot be found, opened, or is invalid."""


# ==========================================================================
# Device selection
# ==========================================================================
def resolve_device(requested: str | None = None) -> torch.device:
    """
    Pick the torch device to run inference on.

    Args:
        requested: "cuda", "cpu", or None to auto-detect (prefers GPU).

    Returns:
        A torch.device instance.
    """
    if requested is not None:
        requested = requested.lower()
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    if torch.cuda.is_available():
        logger.info("CUDA GPU detected — using GPU for inference.")
        return torch.device("cuda")

    logger.info("No GPU detected — using CPU for inference.")
    return torch.device("cpu")


# ==========================================================================
# Checkpoint loading
# ==========================================================================
def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    """
    Load a checkpoint file and extract the model's state dict, regardless
    of whether it was saved as:
        - {"state_dict": ...}
        - {"model_state_dict": ...}
        - a raw state_dict (i.e. the checkpoint IS the state_dict)

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to map the checkpoint tensors onto while loading.

    Returns:
        The extracted state_dict (a plain dict of parameter name -> tensor).

    Raises:
        CheckpointError: If the file is missing or its format is unusable.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint path is not a file: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise CheckpointError(
            f"Failed to load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    # Case 1: raw state_dict (keys map directly to tensors)
    if isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ) and len(checkpoint) > 0:
        logger.info("Detected raw state_dict checkpoint.")
        return checkpoint

    # Case 2 / 3: wrapped dict with a known key
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                logger.info(f"Detected checkpoint wrapped under key '{key}'.")
                return checkpoint[key]

        raise CheckpointError(
            "Checkpoint is a dict but does not contain 'state_dict', "
            "'model_state_dict', or a raw tensor mapping. Found keys: "
            f"{list(checkpoint.keys())}"
        )

    raise CheckpointError(
        f"Unrecognized checkpoint format: expected a dict, got {type(checkpoint)}."
    )


def build_model(state_dict: dict, device: torch.device) -> SRUNet:
    """
    Instantiate SRUNet and load the given state_dict into it.

    NOTE: This assumes SRUNet() can be constructed with no required
    arguments. If your SRUNet signature requires constructor args
    (e.g. in_channels, base_channels, etc.), update the instantiation
    line below accordingly.

    Args:
        state_dict: Parameter dict to load into the model.
        device: Device to move the model onto.

    Returns:
        The model in eval() mode, on the target device.

    Raises:
        CheckpointError: If the state_dict is incompatible with SRUNet.
    """
    model = SRUNet()

    # Strip a common "module." prefix left behind by nn.DataParallel /
    # DistributedDataParallel training, if present.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            "Checkpoint state_dict is incompatible with the SRUNet "
            f"architecture defined in src/models/sr_unet.py: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model


# ==========================================================================
# Image pre / post-processing
# ==========================================================================
def preprocess_image(image_path: str | Path, device: torch.device) -> torch.Tensor:
    """
    Load a grayscale image and convert it to a (1, 1, H, W) float tensor
    normalized to [0, 1].

    Args:
        image_path: Path to the input image.
        device: Device to place the resulting tensor on.

    Returns:
        Tensor of shape (1, 1, H, W), dtype float32, values in [0, 1].

    Raises:
        ImageError: If the file is missing or cannot be opened as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise ImageError(f"Input image not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.load()
        img = img.convert("L")  # force single-channel grayscale
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"Could not open '{image_path}' as an image: {exc}") from exc

    array = np.asarray(img, dtype=np.float32) / 255.0  # -> [0, 1]
    tensor = torch.from_numpy(array).unsqueeze(0)  # (1, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(device)


def postprocess_tensor(output: torch.Tensor) -> Image.Image:
    """
    Convert a model output tensor back into a grayscale PIL image.

    Args:
        output: Tensor of shape (1, 1, H, W) or (1, H, W), float values
            expected roughly in [0, 1] (values are clamped defensively).

    Returns:
        A PIL Image in mode "L".
    """
    tensor = output.detach().cpu().clone()
    tensor = tensor.squeeze(0)  # drop batch dim -> (1, H, W) or (H, W)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # drop channel dim -> (H, W)

    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


# ==========================================================================
# Core inference
# ==========================================================================
@torch.no_grad()
def run_inference(model: SRUNet, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass through the model in evaluation mode.

    Args:
        model: An SRUNet instance already in eval() mode.
        input_tensor: Input of shape (1, 1, H, W).

    Returns:
        Output tensor of shape (1, 1, 2H, 2W).
    """
    model.eval()
    output = model(input_tensor)
    return output


def infer(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> Path:
    """
    End-to-end convenience wrapper: load checkpoint -> build model ->
    preprocess image -> run inference -> save result.

    Args:
        checkpoint_path: Path to the .pth checkpoint (e.g. checkpoints/best.pth).
        input_path: Path to the low-resolution grayscale input image.
        output_path: Path to write the super-resolved output image to.
        device: "cuda", "cpu", or None to auto-detect.

    Returns:
        The resolved Path of the saved output image.
    """
    resolved_device = resolve_device(device)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path, resolved_device)

    logger.info("Reconstructing SRUNet and loading weights...")
    model = build_model(state_dict, resolved_device)

    logger.info(f"Preprocessing input image: {input_path}")
    input_tensor = preprocess_image(input_path, resolved_device)
    logger.info(f"Input tensor shape: {tuple(input_tensor.shape)}")

    logger.info("Running inference...")
    output_tensor = run_inference(model, input_tensor)
    logger.info(f"Output tensor shape: {tuple(output_tensor.shape)}")

    result_image = postprocess_tensor(output_tensor)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)
    logger.info(f"Saved super-resolved image to: {output_path}")

    return output_path


# ==========================================================================
# CLI entry point
# ==========================================================================
def restore_image(input_path: str | Path, checkpoint_path: str | Path, device: str | None = None) -> Image.Image:
    """
    Restore/super-resolve a grayscale image using the trained SRUNet model.

    Args:
        input_path: Path to the low-resolution grayscale input image.
        checkpoint_path: Path to the .pth checkpoint (e.g. checkpoints/best.pth).
        device: "cuda", "cpu", or None to auto-detect.

    Returns:
        The super-resolved PIL Image.
    """
    from PIL import Image
    model = build_model(load_checkpoint(checkpoint_path, resolve_device(device)), resolve_device(device))
    output_tensor = run_inference(model, preprocess_image(input_path, resolve_device(device)))
    return postprocess_tensor(output_tensor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SRUNet 2x super-resolution inference on a grayscale image."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to the model checkpoint (default: checkpoints/best.pth).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input low-resolution grayscale image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save the super-resolved output image (default: output.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force a specific device. Defaults to auto-detect (GPU if available).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        infer(
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            device=args.device,
        )
    except CheckpointError as exc:
        logger.error(f"Checkpoint error: {exc}")
        return 1
    except ImageError as exc:
        logger.error(f"Image error: {exc}")
        return 1
    except Exception as exc:  # catch-all safety net for unexpected failures
        logger.error(f"Unexpected error during inference: {exc}")
        return 1

    logger.info("Inference completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())