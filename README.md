# KLA PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

End-to-end deep-learning restoration of semiconductor inspection images:
a single SR-UNet model jointly removes speckle/Gaussian noise and performs
2× super-resolution (128×128 → 256×256) on single-channel grayscale wafer images.

This repository also contains the **separate** PS02 localization pipeline
(`rerank_v3.py`, `backend/localization_api.py`, `frontend-localization/`).
That pipeline is intentionally untouched by the PS01 work.
Everything below is the **PS01 image restoration** workflow.

---

## Quick Start (Inference)

```bash
# 1. Clone
git clone https://github.com/<your-org>/KLA-PS01-Image-Restoration.git
cd KLA-PS01-Image-Restoration

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install PyTorch (choose ONE)
# CPU-only (works everywhere, slower):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR CUDA (recommended for H100 / any NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Obtain the trained checkpoint
# The checkpoint is tracked with Git LFS. After cloning with LFS installed:
git lfs pull
# Checkpoint will be at: experiments/E1_charb/checkpoints/best.pth

# 6. Run inference on test set (2× SR + 4-flip TTA on GPU)
python evaluate.py \
    --input-dir dataset/test/Test_NoisyLR/NoisyLR \
    --output-dir outputs/final_submission \
    --checkpoint experiments/E1_charb/checkpoints/best.pth \
    --tta
```

**Expected outputs:**
- 400 restored `.npy` files in `outputs/final_submission/`
- Each file: 256×256, float32, finite values, range [0, 1]
- Filenames preserved exactly (e.g., `000000.npy` → `000000.npy`)

---

## Checkpoint

| Checkpoint | Description | Size |
|------------|-------------|------|
| `experiments/E1_charb/checkpoints/best.pth` | **Final model (E1)** — Charbonnier+SSIM fine-tune, 25 epochs, 4-flip TTA | 243 MB (Git LFS) |
| `checkpoints/best.pth` | Original training (epoch 24, L1+SSIM) | 243 MB |

**Model details:**
- Architecture: SR-UNet v1 (U-Net encoder/decoder + PixelShuffle 2× head)
- Parameters: 21.2 M
- Input: 1-channel 128×128 noisy LR
- Output: 1-channel 256×256 restored (clamped to [0, 1])
- Training device: NVIDIA RTX 2000 Ada Generation Laptop GPU (8 GB VRAM)

---

## Evaluation Script (`evaluate.py`)

**Submission-ready standalone CLI.** No source-code edits required.

```bash
python evaluate.py \
    --input-dir <DIR> \
    --output-dir <DIR> \
    [--checkpoint PATH] \
    [--gt-dir DIR] \
    [--tta] \
    [--ensemble CKPT1,CKPT2] \
    [--model {srunet,srunet_v2}]
```

**Required:**
- `--input-dir` — directory of degraded grayscale `.npy` images (128×128)
- `--output-dir` — directory where restored `.npy` images are written (created if missing)

**Optional:**
- `--checkpoint` — trained model checkpoint (default: `checkpoints/best.pth`)
- `--gt-dir` — ground-truth directory (same basenames) to compute PSNR/SSIM/LPIPS
- `--tta` — 4-flip test-time augmentation (original, H, V, H+V averaged)
- `--ensemble` — comma-separated checkpoints for prediction averaging
- `--model` — architecture: `srunet` (default) or `srunet_v2`

**Behavior:**
- Auto-detects CUDA; uses AMP on GPU, CPU fallback with clear messaging
- Preserves input filenames exactly
- Output: 256×256 float32 `.npy`, clamped to [0, 1]
- Reports per-image and mean PSNR (dB), SSIM, LPIPS when `--gt-dir` given
- Prints average inference time (ms/image, GPU warmup excluded)

---

## Training Reproducibility

The final E1 model was produced by fine-tuning the original `checkpoints/best.pth`:

```bash
.venv-cuda\Scripts\python.exe -B train_v2.py \
    --warm-start checkpoints/best.pth \
    --epochs 25 \
    --lr 2e-5 \
    --loss charbonnier_ssim \
    --alpha 0.84 \
    --aug flip_rot \
    --seed 42 \
    --val-split 0.05 \
    --ckpt-dir experiments/E1_charb/checkpoints \
    --batch 8 \
    --val-lpips-every 5
```

**Training configuration:**
- Dataset: 3,200 paired samples (dataset/train/train/train/NoisyLR + GT)
- Split: deterministic 95/5 train/val (seed 42) — identical to original trainer
- Model: SR-UNet v1 (21.2 M params)
- Loss: Charbonnier + SSIM (α = 0.84)
- Augmentation: flip_rot (paired flips + 90° rotations)
- Optimizer: AdamW (lr 2e-5, weight decay 1e-5) + cosine annealing
- AMP: enabled on CUDA
- Batch size: 8
- Epochs: 25 (best at epoch 15: val PSNR 25.894 dB, SSIM 0.7365)
- Warm start: from `checkpoints/best.pth` (epoch 24, L1+SSIM)

---

## Measured Results

All metrics on **40 held-out training pairs** (filenames `000000`–`000039`), evaluated with `evaluate.py` (4-flip TTA).  
Official test set has no GT; these are held-out validation metrics.

| Configuration | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bilinear upscale | 23.99 | 0.588 | — |
| `best.pth` (single) | 26.08 | 0.729 | 0.277 |
| `best.pth` (4-flip TTA) | 26.14 | 0.735 | 0.278 |
| **E1 + 4-flip TTA (final)** | **26.21** | **0.750** | **0.280** |

**Training validation (160 held-out images, seed 42):**
- Original `best.pth`: 25.83 dB / 0.722 SSIM
- **E1 (epoch 15)**: 25.89 dB / 0.737 SSIM

**Inference time (RTX 2000 Ada, 4-flip TTA):** ~57 ms/image (batch 1, includes all 4 flips)

---

## Project Structure (PS01)

```
├── evaluate.py              # <-- Submission entry point (standalone)
├── train_v2.py              # Reproducible training script
├── requirements.txt         # Runtime + eval + training deps
├── experiments/
│   └── E1_charb/checkpoints/best.pth   # Final checkpoint (Git LFS)
├── checkpoints/
│   └── best.pth             # Original training checkpoint
├── outputs/
│   └── final_submission/    # 400 restored test images (256×256)
├── src/
│   ├── config.py            # Paths, hyperparameters, device
│   ├── data/dataset.py      # Paired NoisyLR/GT dataset
│   ├── losses.py            # Composite loss (Charbonnier + SSIM)
│   ├── metrics.py           # PSNR, SSIM, MSE
│   ├── models/
│   │   ├── sr_unet.py       # SR-UNet v1 (final architecture)
│   │   ├── sr_unet_v2.py    # Experimental stronger head
│   │   └── blocks.py        # DoubleConv, ResidualBlock, UpBlock
│   └── training/train.py    # Original training loop
├── backend/
│   ├── restoration_api.py   # PS01 FastAPI (port 8002, E1 + TTA)
│   └── main.py              # Legacy API (port 8000)
├── frontend-restoration/    # Demo UI (Vite + React, port 5174)
└── dataset/                 # KLA dataset (not in repo — download separately)
```

---

## Requirements

```text
numpy==2.5.1
torch>=2.5
torchvision>=0.18
lpips==0.1.4
tqdm==4.70.0
fastapi==0.141.1
uvicorn==0.52.1
python-multipart==0.0.32
pydantic==2.13.4
```

**PyTorch install:** Choose CPU or CUDA build first (see Quick Start), then `pip install -r requirements.txt`.

---

## Demo Frontend (Optional)

The PS01 demo UI is **not required for benchmarking**. It is a separate visualization tool.

```bash
# Terminal 1: PS01 API (E1 + TTA, CUDA)
.venv-cuda\Scripts\python.exe -B -m uvicorn backend.restoration_api:app --host 127.0.0.1 --port 8002

# Terminal 2: Frontend
cd frontend-restoration
npm install
npm run dev  # http://127.0.0.1:5174
```

Features:
- Upload 128×128 `.npy` → preview noisy input
- Click **Restore Image** → real GPU inference (4-flip TTA)
- Optional: add GT for real PSNR/SSIM/LPIPS
- Download restored `.npy` (actual float32 model output)
- BEFORE/AFTER panels use per-image normalization for display only

---

## PS02 Localization (Separate, Untouched)

This repo also contains the PS02 track (`rerank_v3.py`, `backend/localization_api.py`, `frontend-localization/`).
**It is completely separate from PS01 and remains unchanged.**

---

## Reproducibility Notes

- Seed fixed at 42 throughout
- `train_v2.py` uses the **same deterministic train/val split** as the original trainer (seed 42, 5% held out)
- Metrics implementations shared between training and evaluation (`src/metrics.py`)
- Exact float bit-for-bit reproducibility may vary with CUDA/CPU device and AMP state
- All reported numbers produced by `evaluate.py` (or identical internal code)

---

## Limitations

- Trained on 3,200 paired samples; generalization to unseen noise distributions not verified
- Model clamps outputs to [0, 1]; extreme bright regions (>1) in noisy input may lose absolute scale
- No official test-set GT exists; final submission uses held-out validation metrics only
- Residual grain visible on highest-noise inputs (PSNR ~19–22 dB on worst cases)
- Training required GPU (8 GB VRAM); CPU training impractical

---

## License

For SEMICON India Hackathon 2026 submission. See competition rules.

---

## Contact

Team: KLA 
Members: Rudra Patel, Parth Shah 
College: SAL Collage
Email: rudra0425742@gmail.com
