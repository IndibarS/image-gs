#!/usr/bin/env python3
"""Plot a completed initializer step sweep and write a measured comparison report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEPS = (100, 300, 1000, 3000, 10000)
COLORS = {'gradient': '#007c91', 'laplacian': '#cb5724'}


def psnr_value(row):
    return 70.0 if row['psnr'] is None else row['psnr']


def psnr_text(row):
    return '≥70 (floor)' if row['psnr'] is None else f'{row["psnr"]:.3f}'



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    rows = json.loads((directory/'results.json').read_text())
    env = json.loads((directory/'environment.json').read_text())
    images = list(dict.fromkeys(row['image'] for row in rows))
    seeds = list(dict.fromkeys(row['seed'] for row in rows))
    if len(seeds) != 1:
        raise ValueError('This report plots a single seed; report seeds separately rather than silently averaging')
    if len(rows) != len(images)*2*len(STEPS):
        raise ValueError('Step sweep is incomplete')
    for row in rows:
        assert row['adam_steps'] == int(row['mode'].rsplit('-',1)[1])
        assert row['gaussians'] == env['arguments']['gaussians']
        assert row['direct_fitting_seconds'] == 0
        if row['psnr'] is None:
            import torch
            run = directory/f'{Path(row["image"]).stem}-{row["seed"]}-{row["mode"]}'
            checkpoint = torch.load(run/'checkpoints'/f'ckpt_step-{row["adam_steps"]}.pt',
                                    map_location='cpu',weights_only=False)
            if checkpoint['psnr'] != float('inf'):
                raise ValueError('Null PSNR was not caused by the existing infinity threshold')
    def series(image, initializer):
        return sorted((r for r in rows if r['image'] == image and r['mode'].startswith(initializer+'-')),
                      key=lambda r: r['adam_steps'])
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(2, len(images), figsize=(5*len(images),8), squeeze=False, layout='constrained')
    for column, image in enumerate(images):
        for initializer in COLORS:
            subset = series(image,initializer)
            times = [r['encoding_wall_seconds'] for r in subset]
            for row_index, key in enumerate(('psnr','ssim')):
                ax = axes[row_index,column]
                ax.plot(times,[psnr_value(r) if key == 'psnr' else r[key] for r in subset],'-o',color=COLORS[initializer],
                        label=initializer.capitalize(),linewidth=1.8,markersize=5)
                if row_index == 0:
                    for r in subset:
                        ax.annotate(f'{r["adam_steps"]:,}', (r['encoding_wall_seconds'],psnr_value(r)),
                                    xytext=(3,7 if initializer == 'gradient' else -13),
                                    textcoords='offset points',fontsize=8,color=COLORS[initializer])
                if key == 'psnr':
                    capped = [r for r in subset if r['psnr'] is None]
                    ax.scatter([r['encoding_wall_seconds'] for r in capped],[70]*len(capped),
                               marker='^',s=65,color=COLORS[initializer],zorder=4)
                ax.set_xscale('log')
                ax.set_xlabel('Total encoding wall time (seconds; log scale)')
                ax.set_ylabel('PSNR (dB; higher is better)' if key == 'psnr' else 'SSIM (higher is better)')
                ax.grid(True,alpha=0.18,which='both')
                ax.margins(x=0.16,y=0.16)
        axes[0,column].set_title(Path(image).stem.capitalize(),fontweight='bold')
        axes[0,column].legend(frameon=False,loc='best')
    fig.suptitle('Gradient vs Laplacian initialization — quality against total encoding time\n'
                 f'{env["arguments"]["size"]}×{env["arguments"]["size"]} synthetic RGB • '
                 f'{env["arguments"]["gaussians"]} Gaussians from step 0 • '
                 f'{env["arguments"]["bits"]}-bit parameters • CPU • seed {seeds[0]}\n'
                 'Point labels are Adam steps; triangles mean PSNR ≥70 dB (reporting floor).\n'
                 'No progressive allocation, early stopping, or direct color fitting.',fontsize=12)
    fig.savefig(directory/'quality_vs_time.png',dpi=180)
    fig.savefig(directory/'quality_vs_time.svg')
    plt.close(fig)
    lines = ['# Initializer step sweep: measured results', '',
             '## Protocol', '',
             f'Three synthetic {env["arguments"]["size"]}×{env["arguments"]["size"]} RGB images, '
             f'{env["arguments"]["gaussians"]} Gaussians, {env["arguments"]["bits"]}-bit quantization, '
             f'seed {seeds[0]}, CPU with {env["arguments"]["threads"]} PyTorch threads. '
             'Both initializers use the entire Gaussian budget from step zero. Progressive allocation and '
             'early stopping are disabled for both; direct color fitting and residual refinement are off. '
             'The model/renderer are unchanged for this experiment.', '',
             'Each plotted point is an independent process/run at 100, 300, 1,000, 3,000, or 10,000 Adam steps, '
             'evaluated after the usual final quantization. Encoding wall time includes construction, initialization, '
             'Adam, periodic evaluation, final metrics, and image/checkpoint I/O. Interpreter imports/process startup '
             'are included separately in `process_wall_seconds` in CSV/JSON. This is an initializer ablation, '
             'not a rerun of the earlier progressive baseline; results should not be conflated.', '',
             '![Quality versus time](quality_vs_time.png)', '',
             '## All measured points', '']
    for image in images:
        lines += [f'### {image}', '',
                  '| Adam steps | Gradient seconds | Laplacian seconds | Gradient PSNR dB | Laplacian PSNR dB | Gradient SSIM | Laplacian SSIM |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        for g,l in zip(series(image,'gradient'),series(image,'laplacian')):
            lines.append(f'| {g["adam_steps"]:,} | {g["encoding_wall_seconds"]:.3f} | {l["encoding_wall_seconds"]:.3f} | '
                         f'{psnr_text(g)} | {psnr_text(l)} | {g["ssim"]:.6f} | {l["ssim"]:.6f} |')
        lines.append('')
    lines += ['## Endpoint and sampled threshold checks', '',
              'The following exact numerical comparison uses the full-budget gradient 10,000-step PSNR and SSIM '
              'as a target. It is not a perceptual-equivalence criterion. No interpolation or monotonicity is assumed.', '',
              '| Input | Laplacian minus gradient PSNR at 10,000 (dB) | Earliest sampled Laplacian step meeting both gradient endpoint metrics |',
              '|---|---:|---|']
    for image in images:
        baseline=series(image,'gradient')[-1]
        adaptive=series(image,'laplacian')
        matches=[r for r in adaptive if baseline['psnr'] is not None and psnr_value(r) >= baseline['psnr'] and r['ssim'] >= baseline['ssim']]
        match=f'{matches[0]["adam_steps"]:,}' if matches else 'None of the tested points'
        if baseline['psnr'] is None:
            delta = 'Unresolved: baseline ≥70'
        else:
            delta = ('≥' if adaptive[-1]['psnr'] is None else '') + f'{psnr_value(adaptive[-1])-baseline["psnr"]:+.3f}'
        lines.append(f'| {image} | {delta} | {match} |')
    lines += ['', '## Meaning of dB', '',
              'Here dB means **decibels**, the unit used for peak signal-to-noise ratio (PSNR). '
              'For normalized pixels with peak value 1, `PSNR = 10 log10(1 / MSE)`, where MSE is '
              'mean squared reconstruction error over pixels and channels. A +3.0103 dB increase halves '
              'MSE; +10 dB divides MSE by ten. For example, 20/30/40 dB correspond to MSE '
              '0.01/0.001/0.0001. Higher PSNR is better pixelwise reconstruction, not necessarily '
              'better perceived appearance. It is unrelated to file size or number of Gaussians.', '',
              'The current project returns infinite PSNR for MSE <=1e-7; its JSON reporter converts nonfinite '
              'metrics to null. For every null PSNR in this sweep, the saved checkpoint was checked for positive infinity. '
              'Those points are plotted at 70 dB with triangles and listed as ≥70 dB. The true value may be higher; '
              'the apparent flat line at 70 is a reporting cap, not evidence of a quality plateau.', '',
              '## Limits', '',
              '- One seed, tiny synthetic inputs, and a very small Gaussian budget: do not generalize to full-resolution photographs.',
              '- LPIPS and MS-SSIM are unavailable at this resolution. PSNR and SSIM can disagree; the optimizer minimizes L1 + SSIM, not MSE.',
              '- Five sample points cannot establish an asymptotic plateau or prove that more steps would never help.',
              '- Identical learning rates do not necessarily suit both initializations equally. No learning-rate tuning was performed.',
              '- Timings are sequential single-run observations, not repeated timing confidence intervals. CPU contention, imports, and caches can affect them.',
              '- These results isolate initialization at full budget. The original progressive gradient baseline remains available and unchanged.', '',
              'Reproduce:', '', '```bash',
              'uv run python scripts/benchmark_adaptive.py --size 16 --gaussians 17 --step-sweep --output benchmarks/new_step_sweep',
              'uv run python scripts/report_step_sweep.py benchmarks/new_step_sweep', '```', '']
    (directory/'SUMMARY.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
