# Adaptive initialization experiments

Run the complete requested matrix in isolated Python processes:

```bash
uv run python scripts/benchmark_adaptive.py --size 16 --gaussians 17 --output benchmarks/my_smoke
uv run python scripts/benchmark_adaptive.py --images image1.png image2.png --size 0 \
  --gaussians 10000 --seeds 123 456 789 --output benchmarks/full_resolution
```

The first command generates smooth RGB gradients, an equal-luminance chromatic
edge, and a fine checker texture. The second preserves supplied image resolution.
All modes use the same input PNG, seed, Gaussian budget, 16-bit parameter
quantization (change with `--bits`), top-K renderer, and evaluation pipeline.

| Mode | Initialization | Color-fit sweeps | Adam steps |
|---|---|---:|---:|
| baseline | gradient, original progressive allocation | 0 | 10,000 |
| A | Laplacian, full budget | 0 | 0 |
| B | Laplacian, full budget | 0 | 100 |
| C | Laplacian, full budget | 0 | 300 |
| D | Laplacian, full budget | 5 | 0 |
| E | Laplacian, full budget | 5 | 300 |

Early stopping is disabled for **all** benchmark modes to execute those exact
step counts. Baseline progressive allocation and learning rates are retained;
all other modes use the full budget from initialization. Run ordinary `main.py`
for the original default early-stopping behavior. Residual refinement is off in
this matrix so it cannot be confused with the initializer/color-fitting effect.

`results.csv` / `results.json` contain one row per input, mode, and seed:

- `initialization_seconds`: construction, image loading/output, Gaussian analysis,
  parameter initialization, and optimizer creation; excludes direct fitting.
- `direct_fitting_seconds`: weight extraction, renderer agreement check, and fitting.
- `residual_refinement_seconds`: optional separate refinement (zero in this matrix).
- `adam_seconds`: synchronized forward/backward/Adam steps, excluding periodic
  metrics and progressive insertion. These overheads are included in total wall time.
- `encoding_wall_seconds`: construction through final metrics/checkpoint/image I/O;
  includes final quantization and evaluation. Excludes JSON reporting itself.
- `process_wall_seconds`: also includes Python imports/process startup.
- PSNR, SSIM, LPIPS, FLIP, MS-SSIM: final saved/quantized model, evaluated through
  the same model code. Null means unavailable or nonfinite; tiny images cannot
  support LPIPS/MS-SSIM. PSNR uses the existing near-zero-MSE infinity threshold.
- `parameter_payload_bytes`: theoretical packed parameter bits, **not an encoded
  file**. The repository does not write a packed bitstream. No new file format is
  claimed. `checkpoint_file_bytes` is the actual serialized training checkpoint,
  including tensors and optimizer state; zero-step checkpoints have no Adam moments.
- `peak_rss_mib`: isolated Linux process high-water RSS, including Python/PyTorch
  imports; `peak_cuda_allocated_mib` reports PyTorch allocation when CUDA is used.

Inputs, config, logs, checkpoints, renders, environment metadata, and source
SHA-256 hashes are retained. Use a fresh output directory for each experiment.
`--modes A D` selects a subset; `--threads` controls PyTorch CPU threads (default 2).
On NVIDIA systems use `uv run --extra cuda ... --device cuda:0` and verify that
reported `device` actually is CUDA; automatic CPU fallback remains possible.

Small synthetic measurements establish execution correctness and expose tradeoffs;
they cannot establish equivalent quality or speedups on representative photographs
at 10,000 Gaussians. Compare PSNR/SSIM/LPIPS *and total wall time* on real images,
multiple seeds, and matched precision before making that claim. First-run metric
weight downloads and disk caches can also affect total time.

## Initializer step sweep

To isolate initialization rather than progressive allocation, compare both
methods at full budget from step zero:

```bash
uv run python scripts/benchmark_adaptive.py --size 16 --gaussians 17 \
  --step-sweep --output benchmarks/new_step_sweep
uv run python scripts/report_step_sweep.py benchmarks/new_step_sweep
```

This executes independent gradient and Laplacian runs at 100, 300, 1,000, 3,000,
and 10,000 steps. Progressive allocation, early stopping, direct color fitting,
and residual refinement are disabled for both. Identical seed, precision, loss,
and learning rates are retained. This avoids the original gradient schedule
silently extending short runs, and keeps the Gaussian count equal at every step.
It is an initializer ablation, distinct from the original progressive baseline.
The report produces PSNR/SSIM versus total encoding-time PNG/SVG plots plus tables.
