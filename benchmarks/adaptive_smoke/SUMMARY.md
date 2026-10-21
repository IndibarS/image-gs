# Measured adaptive initialization smoke benchmark

CPU, PyTorch 2.4.1+cpu, two PyTorch threads, seed 123. Three generated **16 × 16 RGB images**, **17 Gaussians**, **16-bit quantization** for every parameter group. Each mode ran in a fresh process. Baseline executed all 10,000 steps with original progressive allocation and early stopping disabled. Adaptive runs started with all 17 Gaussians. Residual refinement was disabled.

These are small synthetic experiments, not a representative compression-quality study. LPIPS and MS-SSIM are unavailable at this resolution and recorded as null. No GPU measurements were made. Run the larger-image/multiple-seed commands in `../README.md` before drawing general conclusions.

## Final quantized quality and encoding wall time

| Input | Mode | Adam steps | Encoding seconds | PSNR dB | SSIM |
|---|---|---:|---:|---:|---:|
| gradient.png | baseline | 10000 | 44.6419 | 57.948 | 0.999940 |
| gradient.png | A | 0 | 0.0324 | 22.942 | 0.860409 |
| gradient.png | B | 100 | 0.4847 | 29.519 | 0.964072 |
| gradient.png | C | 300 | 1.3251 | 43.546 | 0.998840 |
| gradient.png | D | 0 | 0.0321 | 23.723 | 0.873988 |
| gradient.png | E | 300 | 1.5913 | 43.777 | 0.998886 |
| edges.png | baseline | 10000 | 44.9141 | 59.953 | 0.999760 |
| edges.png | A | 0 | 0.0240 | 25.822 | 0.927646 |
| edges.png | B | 100 | 0.5072 | 49.217 | 0.999101 |
| edges.png | C | 300 | 1.4519 | 55.953 | 0.999558 |
| edges.png | D | 0 | 0.0375 | 35.292 | 0.981486 |
| edges.png | E | 300 | 1.3846 | 57.825 | 0.999705 |
| texture.png | baseline | 10000 | 49.5448 | 6.554 | 0.448085 |
| texture.png | A | 0 | 0.0254 | 6.030 | 0.176218 |
| texture.png | B | 100 | 0.4865 | 4.501 | 0.201288 |
| texture.png | C | 300 | 1.3915 | 4.238 | 0.233289 |
| texture.png | D | 0 | 0.0322 | 6.058 | 0.178314 |
| texture.png | E | 300 | 1.4639 | 4.129 | 0.233178 |

## Mean stage timings across the three inputs

| Mode | Initialization s | Color fitting s | Adam s | Encoding wall s | Process wall s | Peak RSS range MiB |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0060 | 0.0000 | 45.6572 | 46.3669 | 50.4728 | 354.6–354.8 |
| A | 0.0089 | 0.0000 | 0.0000 | 0.0273 | 4.2464 | 348.9–349.6 |
| B | 0.0090 | 0.0000 | 0.4582 | 0.4928 | 4.6682 | 354.3–354.6 |
| C | 0.0091 | 0.0000 | 1.3420 | 1.3895 | 5.5883 | 354.3–354.6 |
| D | 0.0101 | 0.0070 | 0.0000 | 0.0339 | 4.2801 | 349.8–351.7 |
| E | 0.0100 | 0.0071 | 1.4241 | 1.4800 | 5.6608 | 354.6–357.0 |

## Interpretation

- **No quality-equivalent speedup demonstrated.** On the smooth gradient, C reached 43.55 dB in 1.33 s versus 57.95 dB in 44.64 s for baseline. The approximately 14.4 dB quality gap prevents treating this as an equivalent-quality speedup.
- On the chromatic edge, E reached 57.82 dB in 1.38 s versus baseline 59.95 dB in 44.91 s. This is closer on one tiny input but still does not establish equivalence.
- Fine texture is a failure case at this budget: baseline 6.55 dB, A 6.03 dB, C 4.24 dB, E 4.13 dB. Short Adam refinement does not guarantee better PSNR. The original objective is L1 + SSIM, not squared error; no specific cause for this regression has been isolated.
- Direct color fitting improves unquantized fixed-geometry L2 error (histories are retained in JSON). This does not imply equivalence after quantization or with the baseline.
- Every mode has a theoretical **272-byte parameter payload** (17 × 8 parameters × 16 bits). This is not an on-disk encoded file size. Actual training checkpoint bytes are reported separately in CSV/JSON and differ because optimizer states are absent at zero steps.
- Encoding wall time includes image loading, initialization, fitting, refinement, Adam, evaluation, and image/checkpoint I/O. Process wall time also includes interpreter startup and imports. Very small runs are dominated by startup if measured as a shell command.
- Peak RSS includes Python/PyTorch import costs; these tiny results do not demonstrate bounded memory on full-resolution fitting. Sparse fitting has an explicit entry limit and no dense pixel-by-Gaussian matrix.

Reproduce with:

```bash
uv run python scripts/benchmark_adaptive.py --size 16 --gaussians 17 --output benchmarks/new_smoke
```

Exact input PNGs, source hashes, effective run configs, logs, output images, checkpoints, CSV, and JSON are retained alongside this report.
