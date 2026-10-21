#!/usr/bin/env python3
"""Run isolated, reproducible baseline/A/B/C/D/E encodes; save JSON and CSV.

Example: uv run python scripts/benchmark_adaptive.py --size 16 --gaussians 17
For a representative suite: --images image1.png image2.png --size 0 --gaussians 10000
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODES = {'baseline': ('gradient',10000,0), 'A': ('laplacian',0,0),
         'B': ('laplacian',100,0), 'C': ('laplacian',300,0),
         'D': ('laplacian',0,5), 'E': ('laplacian',300,5)}


def worker(task):
    import torch
    from model import GaussianSplatting2D
    from utils.misc_utils import load_cfg
    torch.set_num_threads(task['threads'])
    torch.hub.set_dir(str(ROOT/'models/torch'))
    args = load_cfg(str(ROOT/'cfgs/default.yaml'),argparse.ArgumentParser()).parse_args([])
    if 'steps' in task:
        mode,steps,sweeps = task['initializer'],task['steps'],0
    else:
        mode,steps,sweeps = MODES[task['mode']]
    args.init_mode=mode; args.max_steps=steps; args.color_fit_sweeps=sweeps
    args.device=task['device']; args.seed=task['seed']; args.num_gaussians=task['gaussians']
    args.quantize=True
    args.pos_bits=args.scale_bits=args.rot_bits=args.feat_bits=task['bits']
    args.input_path=task['image']; args.log_dir=task['run_dir']
    # Enforce requested steps, rather than allowing early stopping. Baseline
    # retains its original progressive allocation; adaptive modes use full N.
    args.disable_lr_schedule=True
    args.disable_prog_optim=task.get("full_budget", False)
    args.eval_steps=100
    args.save_ckpt_steps=args.save_image_steps=100000
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model=GaussianSplatting2D(args)
    model.optimize()
    metrics_path=Path(args.log_dir)/'encoding_metrics.json'
    metrics=json.loads(metrics_path.read_text())
    metrics.update(peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   peak_cuda_allocated_mib=(torch.cuda.max_memory_allocated()/2**20 if model.device.startswith('cuda') else None),
                   torch_version=torch.__version__, python_version=platform.python_version())
    metrics_path.write_text(json.dumps(metrics,indent=2,allow_nan=False))


def prepare_images(args,out):
    import numpy as np
    from PIL import Image
    folder=out/'inputs'; folder.mkdir(exist_ok=True)
    result=[]
    if args.images:
        for i,filename in enumerate(args.images):
            path=Path(filename).resolve()
            image=Image.open(path).convert('RGB')
            if args.size:
                image.thumbnail((args.size,args.size))
            target=folder/f'{i}-{path.stem}.png'
            image.save(target); result.append(target)
    else:
        if args.size < 2:
            raise ValueError('--size must be >=2 for synthetic inputs')
        size=args.size
        y,x=np.mgrid[:size,:size]
        for pattern in ('gradient','edges','texture'):
            if pattern=='gradient':
                image=np.stack([x/(size-1),y/(size-1),(x+y)/(2*(size-1))],-1)
            elif pattern=='edges':
                image=np.zeros((size,size,3)); image[:,:size//2,0]=1; image[:,size//2:,1]=0.2126/0.7152
            else:
                image=np.stack([(x+y)%2,x%2,y%2],-1)
            target=folder/f'{pattern}.png'
            Image.fromarray((image*255).astype(np.uint8)).save(target); result.append(target)
    return result


def save_results(rows,out):
    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    if rows:
        with (out/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',nargs='+')
    parser.add_argument('--size',type=int,default=32,help='Thumbnail bound; 0 preserves supplied image resolution')
    parser.add_argument('--gaussians',type=int,default=64)
    parser.add_argument('--bits',type=int,default=16)
    parser.add_argument('--seeds',type=int,nargs='+',default=[123])
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',default='benchmarks/adaptive')
    parser.add_argument('--step-sweep',action='store_true',help='Compare full-budget gradient and Laplacian at 100/300/1000/3000/10000 steps; no direct color fitting')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(json.loads(Path(args.worker).read_text())); return
    if not 4 <= args.bits <= 32 or args.gaussians < 1 or args.threads < 1:
        parser.error('Use 4..32 bits, positive Gaussian and thread counts')
    out=Path(args.output).resolve()
    # Never reuse a directory containing results: model setup clears run dirs.
    if (out/'results.json').exists():
        parser.error('Output already contains results; choose a new --output directory')
    out.mkdir(parents=True,exist_ok=True)
    images=prepare_images(args,out)
    metadata=dict(arguments=vars(args),platform=platform.platform(),
                  cpu=platform.processor(),logical_cpu_count=os.cpu_count(),
                  timing='encoding wall includes model construction, fitting, Adam, evaluation and image/checkpoint I/O; process wall additionally includes interpreter imports',
                  precision='all modes use identical quantization and the existing save/evaluate path',
                  payload='parameter_payload_bytes is theoretical packed parameter bits only; checkpoint_file_bytes is the actual torch training checkpoint',
                  baseline='10000 gradient-init Adam steps, original progressive allocation, early stopping disabled for fixed step counts')
    # Preserve hashes of the exact checkout, including uncommitted changes.
    metadata['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT/'model.py', ROOT/'main.py',ROOT/'cfgs/default.yaml',*sorted((ROOT/'utils').rglob('*.py'))]}
    if args.step_sweep:
        metadata['baseline']='Full-budget gradient versus full-budget Laplacian; progressive allocation and early stopping disabled for both; no direct color fitting'
    (out/'environment.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for image in images:
        for seed in args.seeds:
            experiments = [(f'{init}-{steps}', init, steps) for init in ('gradient','laplacian') for steps in (100,300,1000,3000,10000)] if args.step_sweep else [(mode,None,None) for mode in args.modes]
            for mode,initializer,steps in experiments:
                name=f'{image.stem}-{seed}-{mode}'
                task=dict(image=str(image),run_dir=str(out/name),mode=mode,seed=seed,
                          gaussians=args.gaussians,bits=args.bits,device=args.device,threads=args.threads)
                if args.step_sweep:
                    task.update(initializer=initializer,steps=steps,full_budget=True)
                taskfile=out/f'{name}.task.json'; taskfile.write_text(json.dumps(task))
                started=perf_counter()
                with (out/f'{name}.log').open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(taskfile)],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                elapsed=perf_counter()-started
                row=dict(image=image.name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),mode=mode,seed=seed,bits=args.bits)
                row.update(json.loads((Path(task['run_dir'])/'encoding_metrics.json').read_text()))
                row['process_wall_seconds']=elapsed
                rows.append(row); save_results(rows,out)
                print(f'{name}: {row["encoding_wall_seconds"]:.3f}s, PSNR={row["psnr"]}, SSIM={row["ssim"]}',flush=True)


if __name__=='__main__':
    main()
