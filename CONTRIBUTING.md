# Contributing to Image-GS

## Development setup

Use uv for dependency management. The supported platform is Linux x86_64
with CPython 3.11.10. Core development does not require NVIDIA hardware:

```bash
git clone https://github.com/IndibarS/image-gs.git
cd image-gs
uv python install 3.11.10
uv sync --locked
uv run python main.py --help
uv run python -m pytest -q tests
```

Plain `uv sync` includes CPU PyTorch and the default `dev` group, without
building the CUDA extensions. `uv sync --no-dev` omits development tools.
The default `device: auto` selects CUDA when the native backend is available,
otherwise CPU. `--device=cpu` forces the differentiable PyTorch renderer.
Optimization, quantization, progressive insertion, checkpoints, FLIP, and
saliency inference work on CPU. Saliency still requires its EMLNet weights.
CPU training is slower; use a small image and `--disable_prog_optim` for short
runs so the progressive schedule does not increase `--max_steps`.

## CUDA development and experiments

Install a compatible NVIDIA driver, CUDA 12.4 development toolkit (`nvcc`),
and C++ compiler. uv installs Python dependencies, not the system toolkit.

```bash
uv sync --locked --extra cuda
uv run --extra cuda python -c "import torch, gsplat, fused_ssim; assert torch.cuda.is_available(); print(torch.__version__)"
uv run --extra cuda python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize
```

Prepare the datasets described in the README before running an experiment.
The CUDA extra selects PyTorch 2.4.1 CUDA 12.4 wheels, the local editable
`gsplat`, and the pinned Git revision of `fused-ssim`. Both extensions build
against the environment's PyTorch with build isolation disabled. Python
changes in `gsplat` are immediately visible; native changes need recompilation.
Always pass `--extra cuda` to `uv run` for CUDA work. Plain `uv sync` switches
back to core dependencies and removes the optional extensions.

## Dependencies and lockfile

`pyproject.toml` is the source of truth. Keep torch, torchvision, and torchaudio
version pins aligned in core dependencies and both extras. CPU wheels come
from the CPU index; the extra selects the explicit CUDA index. PyYAML is a
runtime dependency because the CLI reads YAML configuration files.

The optional `cpu` extra is equivalent to the default installation. Its declared
conflict with `cuda` ensures uv resolves the two wheel variants separately;
do not select both extras or use `--all-extras`. Core package sources are pinned
to PyPI so the PyTorch index does not shadow their versions.

After changing dependencies, run `uv lock` and commit `uv.lock` with the change.
The declared extension metadata lets resolution run without compiling CUDA;
update that metadata if extension packaging changes. `uv sync --locked`
enforces the committed lockfile. The lockfile does not pin drivers or toolkits.

## Validation and troubleshooting

```bash
uv lock --check
uv sync --locked --no-build
uv run --locked python -m pytest -q tests
uv run --locked python -m compileall -q main.py model.py utils gsplat/gsplat
```

CI checks lockfile freshness, syntax, a core install without source builds,
CPU PyTorch selection, absence of CUDA extensions, imports, CLI help, renderer
outputs and gradients against a dense reference, and small CPU optimization /
checkpoint / render runs. Hosted CPU runners cannot verify CUDA compilation
or CUDA output parity.
Validate CUDA changes separately on NVIDIA hardware with a reproducible
input/configuration and before/after outputs where feasible.

If `nvcc` is missing, install the development toolkit; a driver alone is not
enough. For extension build failures, check toolkit and compiler compatibility
with PyTorch 2.4.1 CUDA 12.4. Missing CUDA or extensions trigger a logged CPU
fallback before allocating model tensors. Checkpoint loading maps tensors to
the selected device. Small-image MS-SSIM/LPIPS and unavailable LPIPS weights
are recorded as NaN with a warning rather than preventing checkpoint saving.
The tiled CPU implementation mirrors CUDA projection and blend formulas;
floating-point rounding and tied top-k selections can differ.

## Adaptive initialization experiments

`utils/adaptive_partition_utils.py` implements CPU-only channel-aware activity,
capacity-constrained binary partitioning, and region moments. Its output uses
physical scales and normalized coordinates; `model.py` handles inverse-scale
storage. Do not change the rasterizer's coordinate or checkpoint conventions to
accommodate an initializer.

`utils/color_fitting_utils.py` reuses the CPU renderer's contributor selection
for sparse normalized weights. Always verify weight-matrix reconstruction against
the active backend before fitting. Keep L2 color-fitting claims separate from
L1+SSIM Adam results and final quantized metrics. Budgets beyond available pixels
and unsupported unnormalized direct fitting fail explicitly.

Run `uv run python -m pytest -q tests` for partition coverage/determinism,
chromatic edges, geometry, sparse fitting, short/zero-step schedules, and checkpoint
round trips. Use `scripts/benchmark_adaptive.py` and `benchmarks/README.md` for
isolated mode comparisons. Report total encoding time, precision, input size,
seed, final quality, and memory—not just iteration count. The retained tiny CPU
smoke results do not establish quality-equivalent speedups on real images.
