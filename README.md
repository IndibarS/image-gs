<div align="center">

<h1>Image-GS: Content-Adaptive Image Representation via 2D Gaussians</h1>

[**Yunxiang Zhang**](https://yunxiangzhang.github.io/)<sup>1\*</sup>,
[**Bingxuan Li**](https://bingxuan-li.github.io/)<sup>1\*</sup>,
[**Alexandr Kuznetsov**](https://alexku.me/)<sup>3&dagger;</sup>,
[**Akshay Jindal**](https://www.akshayjindal.com/)<sup>2</sup>,
[**Stavros Diolatzis**](https://www.sdiolatz.info/)<sup>2</sup>,
[**Kenneth Chen**](https://kenchen10.github.io/)<sup>1</sup>,
[**Anton Sochenov**](https://www.intel.com/content/www/us/en/developer/articles/community/gpu-researchers-anton-sochenov.html)<sup>2</sup>,
[**Anton Kaplanyan**](http://kaplanyan.com/)<sup>2</sup>,
[**Qi Sun**](https://qisun.me/)<sup>1</sup>

\* Equal contribution &emsp; &dagger; Work done while at Intel

<sup>1</sup>
<a href="https://www.immersivecomputinglab.org/research/"><img width="30%" src="assets/images/NYU-logo.png" style="vertical-align: top;" alt="NYU logo"></a>
&emsp;
<sup>2</sup>
<a href="https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/overview.html"><img width="22%" src="assets/images/Intel-logo.png" style="vertical-align: top;" alt="Intel logo"></a>
&emsp;
<sup>3</sup>
<a href="https://www.amd.com/en.html"><img width="33%" src="assets/images/AMD-logo.png" style="vertical-align: top;" alt="AMD logo"></a>

<a href="https://arxiv.org/abs/2407.01866"><img src="https://img.shields.io/badge/arXiv-2407.01866-ff5a5f" alt="arXiv"></a>
<a href="https://www.immersivecomputinglab.org/publication/image-gs-content-adaptive-image-representation-via-2d-gaussians/"><img src="https://img.shields.io/badge/project-ImageGS-9fa0ff" alt="Project Page"></a>
<a href="https://github.com/NYU-ICL/image-gs"><img src="https://visitor-badge.laobi.icu/badge?page_id=NYU-ICL.image-gs&left_color=gray&right_color=lightgreen" alt="Visitors"></a>
<a href="https://www.youtube.com/watch?v=_WjU5d26Cc4"><img src="https://img.shields.io/badge/Two Minute Papers-ffee32" alt="Two Minute Papers"></a>
<a href="https://learnopencv.com/image-gs-image-reconstruction-using-2d-gaussians/"><img src="https://img.shields.io/badge/LearnOpenCV-4cc9f0" alt="LearnOpenCV"></a>

</div>

<div style="width: 90%; margin: 0 auto;">
  <p>
  Neural image representations have emerged as a promising approach for encoding and rendering visual data. Combined with learning-based workflows, they demonstrate impressive trade-offs between visual fidelity and memory footprint. Existing methods in this domain, however, often rely on fixed data structures that suboptimally allocate memory or compute-intensive implicit models, hindering their practicality for real-time graphics applications.
  </p>
  <p>
  Inspired by recent advancements in radiance field rendering, we introduce Image-GS, a content-adaptive image representation based on 2D Gaussians. Leveraging a custom differentiable renderer, Image-GS reconstructs images by adaptively allocating and progressively optimizing a group of anisotropic, colored 2D Gaussians. It achieves a favorable balance between visual fidelity and memory efficiency across a variety of stylized images frequently seen in graphics workflows, especially for those showing non-uniformly distributed features and in low-bitrate regimes. Moreover, it supports hardware-friendly rapid random access for real-time usage, requiring only 0.3K MACs to decode a pixel. Through error-guided progressive optimization, Image-GS naturally constructs a smooth level-of-detail hierarchy. We demonstrate its versatility with several applications, including texture compression, semantics-aware compression, and joint image compression and restoration.
  </p>
  <img src="assets/images/teaser.jpg" width="100%" />
  <sub>
  Figure 1: Image-GS reconstructs an image by adaptively allocating and progressively optimizing a set of colored 2D Gaussians. It achieves favorable rate-distortion trade-offs, hardware-friendly random access, and flexible quality control through a smooth level-of-detail stack. (a) visualizes the optimized spatial distribution of Gaussians (20% randomly sampled for clarity). (b) Image-GS’s explicit content-adaptive design effectively captures non-uniformly distributed image features and better preserves fine details under constrained memory budgets. In the inset error maps, brighter colors indicate larger errors.
  </sub>
</div>

## Setup

Image-GS uses **[uv](https://docs.astral.sh/uv/)** to manage Python and its dependencies. The supported research environment is **Linux x86_64, Python 3.11.10, PyTorch 2.4.1, and CUDA 12.4**. An NVIDIA GPU, matching driver, and CUDA development toolkit (including `nvcc`) are required to compile and run the custom rasterizer. A differentiable PyTorch CPU fallback supports optimization and rendering without NVIDIA hardware.

For CPU execution and development on Linux x86_64:

```bash
uv sync
uv run python main.py --input_path="img.jpg" --num_gaussians=10000 --quantize
```

The default `device: auto` uses the native CUDA backend when available and
falls back to CPU when CUDA or its extensions are unavailable. Use
`--device=cpu` to force CPU. Input paths are resolved under `--data_root`
(default `media`), then relative to the current directory if not found there.
CPU training is much slower; start with a small image and fewer Gaussians:

```bash
uv run python main.py --input_path="small.png" --num_gaussians=256 \
  --max_steps=100 --disable_prog_optim --eval_steps=10 --quantize
```

`--disable_prog_optim` prevents the progressive schedule from increasing a
short `--max_steps` run to its configured minimum. The CPU rasterizer uses
tiles, bounded Gaussian batches, and recomputation during backpropagation to
limit intermediate memory. It supports top-k normalization, unnormalized
blending, no-tile rendering, quantization, and checkpoint evaluation. Numerical
rounding and tied top-k selections may differ from CUDA; NVIDIA parity has not
been validated on hardware.

FLIP and saliency inference follow the selected device. Saliency initialization
still needs the EMLNet weights described below. LPIPS needs pretrained AlexNet
weights (downloaded on first use); if they cannot be downloaded, it is logged
as unavailable (`NaN`) and the checkpoint is still saved. Small images also
record unavailable LPIPS (<64 pixels) and MS-SSIM (<=160 pixels) as `NaN`.

Run tests with `uv run python -m pytest -q tests`.

For CUDA execution:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and verify your CUDA toolchain (`nvcc --version` and `nvidia-smi`).
2. Clone the repository and install the project dependencies:

    ```bash
    git clone https://github.com/IndibarS/image-gs.git
    cd image-gs
    uv python install 3.11.10
    uv sync --locked --extra cuda --no-dev
    ```

    uv installs the pinned CUDA 12.4 PyTorch wheels and builds the local `gsplat` and Git-hosted `fused-ssim` extensions against the project environment. The first installation compiles CUDA code and requires a working C++/CUDA compiler. `uv lock` resolves dependency metadata without compiling CUDA code; a compatible NVIDIA system is still needed for `uv sync --extra cuda` and native CUDA execution.

3. Confirm the CUDA runtime and custom modules:

    ```bash
    uv run --locked --extra cuda --no-dev python -c "import torch, gsplat, fused_ssim; print(torch.__version__, torch.cuda.is_available())"
    ```

4. Download the image and texture datasets from [OneDrive](https://1drv.ms/u/c/3a8968df8a027819/EeshjZJlMtdCmvvmESiN2pABM71EDaoLYmEwuOvecg0tAA?e=GybqBv) and place them under `media/images` and `media/textures`.

5. (Optional) For saliency-guided initialization, download the pretrained [EML-Net](https://github.com/SenJia/EML-NET-Saliency) models ([res_imagenet.pth](https://drive.google.com/open?id=1-a494canr9qWKLdm-DUDMgbGwtlAJz71), [res_places.pth](https://drive.google.com/open?id=18nRz0JSRICLqnLQtAvq01azZAsH0SEzS), [res_decoder.pth](https://drive.google.com/open?id=1vwrkz3eX-AMtXQE08oivGMwS4lKB74sH)) and put them in `models/emlnet/`.

Always include `--extra cuda` in `uv run` for CUDA work; plain `uv sync` switches
back to the core environment and removes optional extensions.

Commands in the examples below use `python main.py`; after uv setup, run them as **`uv run --extra cuda --no-dev python main.py ...`** instead. See [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency updates, and troubleshooting.

## Quick Start

#### Image Compression 
- Optimize an Image-GS representation for an input image `anime-1_2k.png` using `10000` Gaussians with half-precision parameters
```bash
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize
```
- Render the corresponding optimized Image-GS representation at a new resolution with height `4000` (aspect ratio is maintained)
```bash
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize --eval --render_height=4000
```

#### Texture Stack Compression
- Optimize an Image-GS representation for an input texture stack `alarm-clock_2k` using `30000` Gaussians with half-precision parameters
```bash
python main.py --input_path="textures/alarm-clock_2k" --exp_name="test/alarm-clock_2k" --num_gaussians=30000 --quantize
```
- Render the corresponding optimized Image-GS representation at a new resolution with height `3000` (aspect ratio is maintained)
```bash
python main.py --input_path="textures/alarm-clock_2k" --exp_name="test/alarm-clock_2k" --num_gaussians=30000 --quantize  --eval --render_height=3000
```

#### Control bit precision of Gaussian parameters
- Optimize an Image-GS representation for an input image `anime-1_2k.png` using `10000` Gaussians with 12-bit-precision parameters
```bash
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize --pos_bits=12 --scale_bits 12 --rot_bits 12 --feat_bits 12
```

#### Switch to saliency-guided Gaussian position initialization 
- Optimize an Image-GS representation for an input image `anime-1_2k.png` using `10000` Gaussians with half-precision parameters and saliency-guided initialization
```bash
python main.py --input_path="images/anime-1_2k.png" --exp_name="test/anime-1_2k" --num_gaussians=10000 --quantize --init_mode="saliency"
```

### Command Line Arguments
Please refer to `cfgs/default.yaml` for the full list of arguments and their default values.

**Post-optimization rendering**
- `--eval` render the optimized Image-GS representation.
- `--render_height` image height for rendering (aspect ratio is maintained).

**Bit precision control**: 32 bits (float32) per dimension by default 
- `--quantize` enable bit precision control of Gaussian parameters.
- `--pos_bits` bit precision of individual coordinate dimension.
- `--scale_bits` bit precision of individual scale dimension.
- `--rot_bits` bit precision of Gaussian orientation angle.
- `--feat_bits` bit precision of individual feature dimension.

**Logging**
- `--exp_name` path to the logging directory.
- `--vis_gaussians`: visualize Gaussians during optimization.
- `--save_image_steps` frequency of rendering intermediate results during optimization.
- `--save_ckpt_steps` frequency of checkpointing during optimization.

**Input image**
- `--input_path` path to an image file or a directory containing a texture stack.
- `--downsample` load a downsampled version of the input image or texture stack as the optimization target to evaluate image upsampling performance.
- `--downsample_ratio` downsampling ratio.
- `--gamma` optimize in a gamma-corrected space, modify with caution.

**Gaussian**
- `--num_gaussians` number of Gaussians (for compression rate control).
- `--init_scale` initial Gaussian scale in number of pixels.
- `--disable_topk_norm` disable top-K normalization.
- `--disable_inverse_scale` disable inverse Gaussian scale optimization.
- `--init_mode` Gaussian position initialization mode, valid values include "gradient", "saliency", and "random".
- `--init_random_ratio` ratio of Gaussians with randomly initialized position.

**Optimization**
- `--disable_tiles` disable tile-based rendering (warning: optimization and rendering without tiles will be way slower).
- `--max_steps` maximum number of optimization steps.
- `--pos_lr` Gaussian position learning rate.
- `--scale_lr` Gaussian scale learning rate.
- `--rot_lr` Gaussian orientation angle learning rate.
- `--feat_lr` Gaussian feature learning rate.
- `--disable_lr_schedule` disable learning rate decay and early stopping schedule.
- `--disable_prog_optim` disable error-guided progressive optimization.

## Acknowledgements
We would like to thank the [gsplat](https://github.com/nerfstudio-project/gsplat) team, as well as the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Fused SSIM](https://github.com/rahul-goel/fused-ssim), and [EML-Net](https://github.com/SenJia/EML-NET-Saliency) for their outstanding work, which provided an important foundation for the development of Image-GS. We also thank [Károly Zsolnai-Fehér](https://users.cg.tuwien.ac.at/zsolnai/about/) for featuring our work on [Two Minute Papers](https://www.youtube.com/watch?v=_WjU5d26Cc4) and introducing it to a broader audience, and [Shubham Anand](https://www.linkedin.com/in/shubham-anand-91a10b211/) for creating a clear, thorough, and well-explained tutorial on our work on [LearnOpenCV](https://learnopencv.com/image-gs-image-reconstruction-using-2d-gaussians/).

## License
This project is licensed under the terms of the MIT license.

## Citation
If you find this project helpful to your research, please consider citing [BibTeX](assets/docs/image-gs.bib):
```bibtex
@inproceedings{zhang2025image,
  title={Image-gs: Content-adaptive image representation via 2d gaussians},
  author={Zhang, Yunxiang and Li, Bingxuan and Kuznetsov, Alexandr and Jindal, Akshay and Diolatzis, Stavros and Chen, Kenneth and Sochenov, Anton and Kaplanyan, Anton and Sun, Qi},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--11},
  year={2025}
}
```

## Laplacian-guided adaptive initialization

`--init_mode=laplacian` calculates Gaussian geometry from image structure on CPU,
then optionally fits colors or refines all parameters with Adam on the selected
CPU/CUDA backend. It always initializes the **full requested Gaussian budget**.
It does not silently expand a requested 100- or 300-step experiment to satisfy
the original progressive schedule. Existing random, gradient, and saliency modes
retain their default progressive behavior. `--max_steps=0` or `--skip_adam`
initializes the full budget in any mode and still renders, evaluates, and saves.

```bash
# Original gradient baseline (exactly 10,000 steps: disable early stopping)
uv run python main.py --input_path=img.jpg --exp_name=compare/baseline \
  --init_mode=gradient --num_gaussians=1000 --max_steps=10000 --disable_lr_schedule --quantize

# Direct image-derived geometry and regional colors, no Adam
uv run python main.py --input_path=img.jpg --exp_name=compare/laplacian0 \
  --init_mode=laplacian --num_gaussians=1000 --max_steps=0 --partition_visualize --quantize

# Full initialization followed by a short Adam refinement
uv run python main.py --input_path=img.jpg --exp_name=compare/laplacian300 \
  --init_mode=laplacian --num_gaussians=1000 --max_steps=300 --disable_lr_schedule --quantize

# Fixed-geometry sparse L2 color fitting, no Adam
uv run python main.py --input_path=img.jpg --exp_name=compare/fitted0 \
  --init_mode=laplacian --num_gaussians=1000 --color_fit_sweeps=5 --skip_adam --quantize

# Optional residual replacement/refit ablation (preserves total count)
uv run python main.py --input_path=img.jpg --exp_name=compare/residual \
  --init_mode=laplacian --num_gaussians=1000 --color_fit_sweeps=5 \
  --residual_refine_count=4 --residual_refine_sweeps=2 --skip_adam --quantize
```

For native NVIDIA execution add `--extra cuda` to `uv run`. Use smaller images
for CPU experiments; this initializer does not make a large Adam step cheap.
Choose separate `--exp_name` values for experiments: existing training behavior
clears the output directory for a repeated name/configuration. Use the same
settings and `--eval` to load a saved representation; initialization/fitting are
not repeated during checkpoint evaluation.

### Algorithm and controls

All options below are also defaults in `cfgs/default.yaml` and CLI arguments.

| Option | Default | Meaning |
|---|---:|---|
| `laplacian_smoothing` | 0.8 | Spatial Gaussian prefilter sigma in pixels; 0 disables it |
| `activity_epsilon` | 0.05 | Uniform activity floor relative to mean-normalized derivatives |
| `activity_alpha` | 0 | Gradient magnitude weight; 0 gives pure Laplacian activity |
| `activity_beta` | 1 | Absolute Laplacian weight |
| `partition_strategy` | balanced | `balanced` trades mass accuracy against shape; `longest` prefers longest-axis cuts |
| `partition_scale_factor` | 1.5 | Multiplier on covariance-derived pixel standard deviations |
| `partition_min_scale` | 0.35 | Minimum physical scale in pixels |
| `partition_visualize` | false | Save `partition.png` showing initial region boundaries |
| `color_fit_sweeps` | 0 | Optional cyclic L2 coordinate-descent sweeps |
| `color_fit_epsilon` | 1e-10 | Stabilizer for color-fit denominators |
| `color_fit_max_entries` | 20,000,000 | Maximum estimated sparse pixel/contributor entries; fail clearly if exceeded |
| `skip_adam` | false | Fit/initialize and save without Adam |
| `residual_refine_count` | 0 | Replace up to 32 low-impact Gaussians in one optional batch |
| `residual_refine_sweeps` | 2 | Fit colors sharing affected pixels after replacement |

Derivatives are calculated **per channel**, then absolute Laplacian / gradient
magnitudes are averaged across channels and normalized by spatial mean. This
preserves equal-luminance chromatic edges. Constant images use uniform activity;
noise sensitivity is controlled by smoothing. Texture stacks use all channels.
The analysis uses the loaded, gamma-transformed optimization target.

An integral image supports budget-aware binary cuts near the target child mass.
The partition has exactly N nonempty, nonoverlapping regions covering the image.
Pixel capacity constrains child budgets; mass balance and aspect ratio cannot both
be exact near strong impulses or for tiny regions. Requests exceeding the pixel
count raise a clear error rather than duplicating regions. No K-means or global
pixel-to-Gaussian distance matrix is used. Partitioning is deterministic without
random sampling, so different seeds produce the same adaptive geometry.

Each region supplies an activity-weighted centroid/covariance and a regional mean
feature vector. Pixel footprint variance stabilizes single-pixel regions; physical
scales are clamped positive and isotropic orientation is fixed to zero. Coordinates
are divided by image width/height, rotations use gsplat's clockwise convention,
and physical scales are inverted when inverse-scale storage is enabled.
`disable_color_init` retains its old meaning in legacy modes; Laplacian mode always
uses regional colors because it is a direct initializer.

### Sparse fitting and residual refinement

The optional color solver freezes geometry, extracts the CPU renderer's actual
normalized top-K weights, and stores W in sparse form with O(pixels × K) entries.
Contributor selection and normalization are shared with the CPU renderer.
Projection and quantization use the active model's conventions. Before solving,
`W @ C` is compared against the active renderer; a mismatch (for example native
CUDA tie handling) raises an error recommending `--device=cpu`, rather than
silently solving an inaccurate model. Native CUDA parity is not hardware-verified.
Unnormalized tiled blending is currently unsupported by this optional solver and
is explicitly rejected. Normal rendering/Adam still supports it.

Cyclic coordinate descent updates all feature channels together and immediately
updates the residual after each column. This minimizes **unquantized squared
error for fixed geometry**, not L1 + SSIM. Quantization, clamping, gamma correction,
and later geometry updates may change the resulting quality. Final reported
metrics are measured after the existing quantization/save path. Zero-weight
columns are left unchanged. Sparse storage is bounded explicitly; full-resolution
images can require hundreds of MB, and extraction currently checks Gaussian tile
bounds per tile. It is not a constant-memory streaming solver.

Optional residual refinement replaces a bounded batch of low-contribution
Gaussians at separated high-error pixels, rebuilds membership once, and fits only
colors sharing affected pixels. It accepts the batch only if actual rendered MSE
improves, otherwise restores the original tensors. The total Gaussian budget is
unchanged. This stage still needs a full sparse membership rebuild and validation
render; it is disabled by default and timed separately.

### Measurements

Every completed training run writes `encoding_metrics.json` with initialization,
color-fitting, residual refinement, Adam, and total encoding wall times plus final
quality and size accounting. Timing on CUDA is synchronized. Parameter payload
bytes are an estimated packed bit budget; the `.pt` file is a larger training
checkpoint, not a newly implemented compressed bitstream.

See [benchmarks/README.md](benchmarks/README.md) and
`scripts/benchmark_adaptive.py` for the reproducible six-mode comparison (baseline
10,000 steps; adaptive 0/100/300; fitted adaptive 0/300), JSON/CSV exports, isolated
process peak-memory measurements, and the interpretation limits of small tests.
No quality-equivalent speedup is assumed.
