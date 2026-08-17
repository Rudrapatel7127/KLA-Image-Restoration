# PS01 Image Restoration — Final Report (GPU task)

Date: 2026-08-11
Scope: KLA / PS01 image restoration only (SR-UNet, 2x super-resolution +
speckle/Gaussian denoising, grayscale). All training/evaluation performed on
the NVIDIA RTX 2000 Ada GPU in a dedicated `.venv-cuda` environment. The
Applied Materials / PS02 localization pipeline was NOT modified (verified by
SHA-256 comparison — 37 recorded files, 0 changed).

## 1. Baseline (frozen, verified)

- Model: SR-UNet (U-Net encoder/decoder + PixelShuffle(2) residual head,
  ~21.2M parameters), 1 channel in/out, bilinear-base residual design.
- `checkpoints/best.pth` — epoch 24; trainer-held-out validation (160 images,
  seed 42): **PSNR 25.8306 dB / SSIM 0.7223**.
- External 40-pair evaluation (`000000-000039`): **26.0812 dB / 0.7291 /
  0.2774 LPIPS** (LPIPS = AlexNet, grayscale replicated to RGB,
  [0,1] -> [-1,1]).
- Baseline checkpoint was never overwritten; all experiment checkpoints live
  under `experiments/<name>/checkpoints/`.

## 2. Hardware / environment

- GPU: **NVIDIA RTX 2000 Ada Generation Laptop GPU** (8 GB VRAM, driver
  596.86). PyTorch CUDA version 12.8, CUDA availability verified (tensor
  tests pass). Baseline inference on GPU: 26.0812 dB @ **27.8 ms/image**.
- Dedicated GPU venv **`.venv-cuda`**: Python 3.14.5, torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, numpy, tqdm, lpips 0.1.4, pyyaml. The shared
  `.venv` (CPU-only torch 2.13.0+cpu) used by PS02 was NOT modified.
- All GPU training used AMP + cosine LR, batch 8, same deterministic seed-42
  val split (5% = 160 images) as the original trainer, so validation numbers
  are directly comparable.

## 3. What was changed (PS01 only)

| File | Change |
|---|---|
| `train_v2.py` | Extended: `--model {srunet,srunet_v2}` architecture registry and `--warm-strict {true,false}`; warm-start now filters to shape-matched tensors before `load_state_dict(strict=False)` (torch 2.11 raises on any size mismatch otherwise). |
| `evaluate.py` | Added `--model {srunet,srunet_v2}` to match experimental checkpoints. Default behavior still verified byte-identical to the original (8/8). |
| `src/models/sr_unet_v2.py` (new) | Experimental stronger SR head (64->64->64->256 /PixelShuffle->64->32->1, zero-init final conv; 21.31M params). Contract tests 128->256 and 256->512 pass. |
| `experiments/E1_charb/` (new) | **Selected** final checkpoint. |
| `experiments/E2_charb90/`, `experiments/E4_v2head/`, `experiments/E4b_v2transfer/`, `experiments/E5_speckle/` (new) | Rejected experiment checkpoints (kept for reproducibility). |
| `experiments/EXPERIMENTS.md` (new) | Full experiment log with configs and measured numbers. |
| `outputs/final_gpu_test/` (new) | Final 400-image GPU inference (TTA). |
| `logs/E*_*.log` (new) | Training logs + `training_summary.txt` per run. |
| `README.md` | Updated checkpoint/commands/measured-results sections. |

Intentionally untouched: `checkpoints/best.pth` and all original
`epoch_*.pth`, `outputs/final_test`, `outputs/final_submission`,
`src/config.py`, `src/metrics.py`, `src/losses.py`, `src/models/sr_unet.py`,
`scripts/`, `backend/` (PS02), `requirements.txt`, original dataset.

## 4. Experiments (all measured on the same 40 held-out pairs)

1. **E1 — Charbonnier+SSIM, alpha 0.84** (`experiments/E1_charb`, warm-start
   `best.pth`, 25 ep, LR 2e-5, flips+rot90): trainer val **25.894 / 0.7365**;
   40-pair **single 26.1690/0.7460/0.2762**, **TTA 26.2123/0.7502/0.2799**.
   **Selected.**
2. **E2 — same with alpha 0.90** (`experiments/E2_charb90`): trainer val
   25.905/0.7364; 40-pair TTA 26.2066/0.7454/0.2812. Loses to E1 on held-out.
   Discarded.
3. **E4 — SRUNetV2 fresh** (`experiments/E4_v2head`, 100 ep, LR 1e-4):
   unstable late training (val 24.6 -> 23.9 in the tail); best 25.769/0.7218
   @ ep 51. Discarded.
4. **E4b — SRUNetV2 transfer** (`experiments/E4b_v2transfer`, warm-start with
   shape-matched transfer, 25 ep, LR 2e-5): trainer val 25.825/0.7252;
   40-pair TTA 26.1517/0.7430/0.2850. Below E1 on every metric — the stronger
   head does not help this dataset. Discarded.
5. **E5 — speckle-aware augmentation** (`experiments/E5_speckle`, warm-start
   from E1, input speckle/Gaussian sigma 0.01): trainer val 25.870/0.7365;
   40-pair TTA 26.1818/0.7489/0.2829. Below E1. Discarded.
6. **Ensembles** (40-pair, all with TTA): E1+E5 26.2076/0.7501/0.2808;
   E1+best 26.2084/0.7456/0.2777. None beat E1 TTA alone.

## 5. Final selected model

- **`experiments/E1_charb/checkpoints/best.pth`** — Charbonnier+SSIM
  (alpha 0.84) fine-tune of the baseline, 25 epochs, LR 2e-5 cosine,
  flips+rot90, evaluated with **4-flip TTA** (original, H, V, H+V averaged).
- 40-pair held-out metrics: **PSNR 26.2123 dB, SSIM 0.7502, LPIPS 0.2799**.
  Improvement over the baseline single pass: **+0.131 dB PSNR, +0.0211 SSIM**,
  LPIPS +0.0025 (essentially unchanged).
- Trainer-held-out validation: E1 25.894 dB / 0.7365 vs `best.pth`
  25.8306 / 0.7223.

## 6. Final 400-image inference

- Command: `evaluate.py --input-dir dataset/test/Test_NoisyLR/NoisyLR
  --output-dir outputs/final_gpu_test --checkpoint
  experiments/E1_charb/checkpoints/best.pth --tta` on the RTX 2000 Ada GPU.
- Speed: **25.3 ms/image** (400 images, ~10 s total excluding model load).
- Output contract: **400/400 files**, basenames match inputs exactly (0
  missing, 0 extra), every output 256x256 float32, finite, in [0,1]
  (0 violations).
- Prior outputs `outputs/final_test` and `outputs/final_submission` untouched.

## 7. Reproduce

```bash
# Final inference (GPU)
.venv-cuda\Scripts\python.exe -B evaluate.py \
  --input-dir dataset/test/Test_NoisyLR/NoisyLR \
  --output-dir outputs/final_gpu_test \
  --checkpoint experiments/E1_charb/checkpoints/best.pth --tta

# Reproduce the selected fine-tune (GPU)
.venv-cuda\Scripts\python.exe -B -u train_v2.py --warm-start checkpoints/best.pth \
  --epochs 25 --lr 2e-5 --loss charbonnier_ssim --alpha 0.84 --aug flip_rot \
  --seed 42 --val-split 0.05 --ckpt-dir experiments/E1_charb/checkpoints --batch 8

# Metrics on paired data
python evaluate.py --input-dir <noisy> --output-dir <out> --gt-dir <gt> \
  --checkpoint experiments/E1_charb/checkpoints/best.pth --tta
```

## 8. Known limitations

- Held-out metrics use 40 training pairs (the official test set has no GT);
  absolute numbers may differ from official scoring.
- LPIPS of the final model is +0.0025 vs the baseline single pass — the
  PSNR/SSIM gains came at negligible perceptual cost; LPIPS was never a
  training objective.
- TTA multiplies inference cost 4x (still 25 ms/image on GPU, batch-friendly).
- Gains are modest because the original model was already near capacity;
  architecture changes (SRUNetV2) and heavier augmentation (E5) did not beat
  the simple loss+schedule fine-tune on held-out data.
- Test-domain brightness equals training brightness (measured earlier), so no
  brightness augmentation is applied; extreme outliers (>2.0) remain a
  residual risk for a handful of images.

## 9. Files changed (this task)

`train_v2.py`, `evaluate.py`, `src/models/sr_unet_v2.py` (new), `README.md`,
`experiments/EXPERIMENTS.md` (new), `experiments/E1_charb/*`,
`experiments/E2_charb90/*`, `experiments/E4_v2head/*`,
`experiments/E4b_v2transfer/*`, `experiments/E5_speckle/*`,
`outputs/final_gpu_test/*` (new), `logs/E1_charb.log`, `logs/E2_charb90.log`,
`logs/E4_v2head.log`, `logs/E4b_v2transfer.log`,
`logs/E5_speckle.log` (new).

PS02 status: **unchanged** — 37 recorded files (`rerank_v3.py`,
`backend/localization_api.py`, `backend/localization_schemas.py`,
`frontend-localization/` sources) verified identical by SHA-256 at the end of
this task; runtime artifacts (`vite.log`, `*.tsbuildinfo`) are regenerated by
the running dev server and were never part of the baseline record.
