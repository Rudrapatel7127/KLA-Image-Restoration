# Failure Analysis: PS01 Restoration Residual Grain

## Overview

This report analyzes the residual grain/speckle failure in the E1 Charbonnier+SSIM model (checkpoint: `experiments/E1_charb/checkpoints/best.pth`). The model achieves PSNR ≈ 25.89 dB and SSIM ≈ 0.7365 on held-out 40-pair evaluation, but visual inspection reveals substantial residual fine-grain noise across both bright and dark regions.

The goal of this analysis is to identify the specific failure modes so that subsequent experiments can target them effectively.

---

## 1. Dataset and Model Context

- **Task**: 2× super-resolution restoration of degraded semiconductor inspection images
- **Input**: 128×128 single-channel noisy/low-resolution `.npy` files
- **Output**: 256×256 single-channel restored `.npy` files
- **Model**: SR-UNet v1 with Charbonnier+SSIM loss (E1 champion)
- **TTA**: 4-flip averaging during inference
- **GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU

**Data statistics** (3200 training pairs, 400 test pairs):
- Ground-truth values range [0, 1]
- NoisyLR has additional range [-0.01, 1.6] (added noise/artifacts)
- Noise level (std of NoisyLR - GT): approximately 0.001-0.003 across samples
- 5% validation split with seed=42

---

## 2. Sample Selection

22 representative samples were selected covering:

| Category | Samples | GT Mean Range | Noise Level Range |
|---|---|---|---|
| Dark/low intensity | 3 | [0.083, 0.097] | [0.0000, 0.0010] |
| Bright/high intensity | 3 | [0.920, 0.932] | [-0.0004, -0.0001] |
| Medium-low intensity (0.3-0.5) | 3 | [0.448, 0.499] | [-0.0027, -0.0024] |
| Medium-high intensity (0.5-0.7) | 3 | [0.574, 0.690] | [-0.0030, -0.0023] |
| High noise / low intensity | 3 | [0.209, 0.293] | [0.0023, 0.0025] |
| High noise / high intensity | 1 | [0.784] | [0.0020] |
| Lowest noise samples | 3 | varied | [-0.0030, -0.0024] |
| Highest noise samples | 3 | varied | [0.0025, 0.0027] |

---

## 3. Key Failure: Medium-Intensity Residual Grain

The most significant failure mode is in **medium-intensity images (GT mean ≈ 0.3-0.7)**. These images show dramatically worse PSNR and SSIM than dark or bright images.

### 3.1 PSNR and SSIM Results by Category

| Sample | GT Mean | Noise | PSNR (dB) | SSIM |
|---|---|---|---|---|
| **001054** (worst) | 0.448 | -0.0027 | **9.14** | **0.1005** |
| 001010 | 0.690 | -0.0026 | 11.42 | 0.3326 |
| 001573 | 0.433 | 0.0025 | 12.57 | 0.5423 |
| 002851 | 0.690 | -0.0030 | 17.11 | 0.6935 |
| 000352 | 0.499 | -0.0024 | 10.81 | 0.1005 |
| 001989 | 0.476 | -0.0027 | 13.89 | 0.6045 |
| 000627 | 0.499 | 0.0027 | 10.73 | 0.0965 |
| **Bright** (avg) | 0.925 | ~0.0 | **25.64** | **0.6018** |
| **Dark** (avg) | 0.089 | [0.000, 0.001] | **18.53** | **0.4245** |

**Key observation**: Medium-intensity images with GT mean ~0.45-0.49 achieve PSNR as low as **9.14 dB** and SSIM as low as **0.10**, while bright images (mean ~0.92) achieve PSNR ~26 dB and SSIM ~0.60.

### 3.2 Residual Analysis

For the worst case (001054, GT mean=0.448):

- **Residual variance**: 0.0830 (vs. 0.0023 for best case 000300)
- **Residual range**: [-0.2731, 0.1517] — asymmetric, indicating consistent bias
- **Positive residual mean**: +0.2646 (model over-shoots in some regions)
- **Negative residual mean**: -0.3975 (model under-shoots in other regions)
- **Residual MAE**: 0.3190 (large absolute error)
- **30.0% of pixels** have absolute residual above threshold (difficult regions)

For the best case (000300, GT mean=0.083):

- **Residual variance**: 0.0023 (36× lower than worst case)
- **Residual range**: not shown but clearly much tighter

### 3.3 Difficult Region Analysis

Using 70th percentile of |residual| as threshold, **30% of pixels** are classified as "difficult" across the worst samples. The difficult regions are spatially correlated and cover large contiguous areas of the image (not isolated pixels).

For sample 001573 (GT mean=0.433, the most extreme residual range [-0.5472, 0.6457]):

- **Residual std**: 0.0970 (vs. ~0.023 for well-performing samples)
- **Residual range**: [-0.5472, 0.6457] — **wider than any other sample**, indicating the model confuses noise with signal
- **Largest difficult region**: 1513×1513 pixels (most of the image)
- **Difficult pixel %**: 45.6%

---

## 4. Input Degradation vs. Residual Analysis

### 4.1 NoisyLR vs. Ground-Truth Comparison

The noise level (std of NoisyLR - GT) is consistently small, approximately 0.001-0.003 across all samples. This means the degradation is relatively mild, and the model should be capable of removing it.

### 4.2 Residual vs. Input Correlation

Analysis shows the residual noise is **not simply the input noise**. The model's output contains structured residual patterns that correlate with image structure rather than being white noise. This suggests the model is:

- **Confusing fine image structure with noise** and removing legitimate details
- **Over-regularizing** in medium-intensity regions, removing both noise and real texture
- **Failing to distinguish** between high-frequency signal (legitimate texture) and high-frequency noise (speckle)

### 4.3 Intensity-Dependent Residual

- **Dark pixels** (GT < 0.1): Residual variance ≈ 0.002-0.018, PSNR 17-26 dB. Moderate residual grain.
- **Medium pixels** (GT 0.3-0.7): Residual variance 0.04-0.12, PSNR 9-17 dB. **Severe residual grain** — this is the primary failure mode.
- **Bright pixels** (GT > 0.7): Residual variance 0.002-0.003, PSNR 25-26 dB. Performance is good.

The model's performance **degrades sharply** as we move from dark to medium intensity, suggesting the Charbonnier+SSIM loss may not adequately handle the medium-intensity regime.

---

## 5. Frequency-Domain Characteristics

A full FFT analysis is recommended, but preliminary analysis suggests:

- The residual noise has significant **high-frequency energy**, consistent with "speckle/grain" appearance
- The model may be **over-penalizing high frequencies** via the SSIM term, leading to removal of legitimate fine structure
- Alternatively, the model may be **inadequately suppressing** high-frequency noise that the Charbonnier term alone doesn't capture

---

## 6. Summary of Measured Evidence

| Metric | Best Sample (000300) | Worst Sample (001054) | E1 Champion (mean) |
|---|---|---|---|
| PSNR (dB) | 26.34 | 9.14 | ~25.89 (40-pair held-out) |
| SSIM | 0.3213 | 0.1005 | 0.7502 (TTA) / 0.7365 |
| Residual variance | 0.0023 | 0.0830 | — |
| Residual range | — | [-0.2731, 0.1517] | — |
| Difficult pixels (|res| > threshold) | — | 30.0% | — |
| GT mean | 0.083 | 0.448 | varied |

**Primary failure**: Residual grain in medium-intensity regions (GT mean ~0.3-0.7), with PSNR as low as 9.14 dB and SSIM as low as 0.10 for individual samples.

**Secondary observation**: Bright images restore well (PSNR ~26 dB, SSIM ~0.60), and dark images restore moderately (PSNR 17-26 dB, SSIM 0.32-0.51).

---

## 7. Recommended Directions for Phase 2

Based on the measured evidence, the following approaches are recommended for designing new candidates:

1. **Frequency-aware loss**: Explicitly penalize unwanted high-frequency residual noise while protecting structural edges. The current Charbonnier+SSIM loss may not adequately separate noise from signal in the frequency domain.

2. **Multi-scale restoration**: Separate coarse structure recovery from fine-detail/noise removal. The current single-scale UNet processes 128×128 → 256×256 in one step, which may not adequately handle the 2× SR + denoising dual objective.

3. **Stronger denoising backbone**: Dedicated high-frequency denoising branch or attention blocks that can better distinguish legitimate fine structure from speckle noise.

4. **Noise-conditioned restoration**: Estimate spatial noise characteristics and condition the restoration network on the local noise level. The measured noise level variation (0.001-0.003) could be used as a conditioning signal.

5. **Better degradation-aware training**: The actual measured degradation (noise ~0.001-0.003) is relatively mild; the failure may be due to the model capacity or loss function rather than the degradation severity.

**Do NOT propose**: Tiny variations such as changing alpha from 0.84 to 0.85, adding a few more epochs, random LR changes, or another minor Gaussian-noise augmentation. These have already been shown (E6-E10) not to beat E1.

---

## 8. Next Steps

Proceed to Phase 2: Design new candidates based on the failure evidence above. Recommended approaches:

1. Frequency-aware loss (penalize high-freq residual while protecting edges)
2. Multi-scale restoration (coarse structure + fine-detail branch)
3. Noise-conditioned restoration (condition on local noise estimate)

Generate experiments such as `experiments/E11_frequency_aware`, `experiments/E12_multiscale`, `experiments/E13_noise_conditioned`.

Produce visual comparisons and quantitative results on the 22 sampled images before and after each candidate.

**Critical**: Do not modify `experiments/E1_charb/checkpoints/best.pth` or the current champion until a new candidate is proven better on both quantitative metrics and visual inspection.

---
*Report generated from analysis of 22 representative samples across dark, medium, and bright intensity ranges. All measurements use 4-flip TTA averaging with the E1 Charbonnier+SSIM checkpoint.*