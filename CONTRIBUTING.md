# Contributing to Image-GS

## Supported development environment

This repository uses **uv only** for Python dependency management. Its custom `gsplat` rasterizer and `fused-ssim` extension are compiled code, not interchangeable with CPU-only PyTorch. The currently documented supported target is **Linux x86_64, CPython 3.11.10, NVIDIA GPU, PyTorch 2.4.1 (CUDA 12.4 wheels)** with a compatible NVIDIA driver, CUDA toolkit (`nvcc`), and C++ build toolchain. `uv` does not install the system CUDA compiler.

## Development setup

```bash
git clone https://github.com/IndibarS/image-gs.git
cd image-gs
uv python install 3.11.10
uv sync --locked
uv run python -c "import torch, gsplat, fused_ssim; assert torch.cuda.is_available(); print(torch.__version__)"
```

`uv sync` installs the `dev` group by default; use `uv sync --no-dev` for execution only. Package resolution (`uv lock`) can run without a GPU because CUDA-extension metadata is declared in `pyproject.toml`, but environment installation and native compilation require the CUDA toolchain. The `gsplat` package is installed editable, so Python edits are visible without reinstalling, but changes to C++/CUDA code still require recompilation.

## Run an experiment

Place the required datasets under `media/` as described in the [README](README.md), then:

```bash
uv run python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize
```

## Dependencies and lockfile

`pyproject.toml` is the source of truth. Keep the pinned PyTorch/CUDA versions together (`torch`, `torchvision`, `torchaudio`) and do not switch to CPU wheels: native extensions depend on the CUDA build. The local `gsplat` package and a pinned `fused-ssim` Git revision are defined in `[tool.uv.sources]`. Both disable build isolation so they compile against the installed PyTorch; their dependency metadata in `pyproject.toml` must be updated if their upstream packaging changes.

When updating dependencies, use `uv add ...` or edit `pyproject.toml`, then run `uv lock` and **commit the updated `uv.lock`** with the changes. `uv sync --locked` installs from the committed lockfile, while CI checks that the lockfile matches `pyproject.toml` with `uv lock --check`. Lockfiles pin Python package artifacts, but not system CUDA toolchains or GPU drivers.

## Checks and troubleshooting

```bash
uv run python -m compileall -q main.py model.py utils gsplat/gsplat
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

- If `nvcc` is not found, install a compatible CUDA development toolkit and ensure `nvcc` is on `PATH`; a working NVIDIA driver alone is not sufficient for extension builds.
- If the C++/CUDA extensions fail to build, verify toolkit compatibility with the installed PyTorch 2.4.1 CUDA 12.4 wheels and check your compiler setup before changing package versions.
- If `torch.cuda.is_available()` is false, this implementation cannot run on CPU as-is; changing `device: cuda:0` in `cfgs/default.yaml` will not produce a CPU renderer.
- The GitHub Actions configuration checks syntax and uv dependency resolution only; hosted CPU runners cannot verify the CUDA extensions or model output.

Please keep changes to scientific code accompanied by a reproducible input/configuration and, where feasible, before/after outputs.
