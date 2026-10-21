import argparse
import json

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.sparse import csr_matrix

from utils.adaptive_partition_utils import activity_map, initialize_gaussians, partition_activity
from utils.color_fitting_utils import fit_colors, renderer_weights
from utils import cpu_backend as cpu
from utils.misc_utils import load_cfg


def image_pattern(kind, h=13, w=19):
    y,x = np.mgrid[:h,:w]
    if kind == 'uniform':
        image = np.full((3,h,w),0.4)
    elif kind == 'gradient':
        image = np.stack([x/max(1,w-1), y/max(1,h-1), (x+y)/max(1,h+w-2)])
    elif kind == 'edges':
        image = np.zeros((3,h,w)); image[0,:,:w//2] = 1; image[1,:,w//2:] = 0.2126/0.7152
    else:
        image = np.repeat(((x+y)%2)[None],3,axis=0)
    return image.astype(np.float32)


@pytest.mark.parametrize('kind', ['uniform','gradient','edges','texture'])
@pytest.mark.parametrize('shape,count', [((13,19),17),((1,7),5),((3,3),8),((2,2),4),((1,1),1)])
def test_partitions_and_parameters(kind, shape, count):
    image = image_pattern(kind,*shape)
    result = initialize_gaussians(image,count)
    again = initialize_gaussians(image,count)
    coverage = np.zeros(shape,dtype=int)
    for y0,x0,y1,x1 in result.regions:
        assert y1 > y0 and x1 > x0
        coverage[y0:y1,x0:x1] += 1
    assert len(result.regions) == count
    np.testing.assert_array_equal(coverage,1)
    np.testing.assert_array_equal(result.regions,again.regions)
    for values in (result.positions,result.scales,result.rotations,result.features):
        assert np.isfinite(values).all()
    assert (result.scales > 0).all()
    assert ((result.positions >= 0) & (result.positions < 1)).all()


def test_zero_activity_all_budgets():
    for h,w in [(3,5),(2,9),(5,4)]:
        for n in range(1,h*w+1):
            for strategy in ['balanced','longest']:
                regions = partition_activity(np.zeros((h,w)),n,strategy)
                coverage = np.zeros((h,w),dtype=int)
                for y0,x0,y1,x1 in regions:
                    coverage[y0:y1,x0:x1] += 1
                assert len(regions) == n
                assert (coverage == 1).all()


def test_invalid_budgets_and_activity():
    for n in [0,-1,5]:
        with pytest.raises(ValueError):
            initialize_gaussians(np.ones((3,2,2)),n)
    with pytest.raises(ValueError):
        activity_map(np.full((3,2,2),np.nan))
    with pytest.raises(ValueError):
        activity_map(np.ones((3,2,2)),epsilon=-1)


def test_chromatic_edges_and_hybrid():
    image = image_pattern('edges')
    pure = activity_map(image,smoothing=0)
    assert pure[:,8:11].mean() > pure[:,:3].mean()*5
    hybrid = activity_map(image,alpha=1,beta=1)
    assert np.isfinite(hybrid).all()
    assert np.isclose(hybrid.mean(),1)
    assert np.isfinite(activity_map(np.ones((3,3,4)),epsilon=0)).all()


def test_equal_mass_uniform_and_localized_activity():
    activity = np.ones((32,48))
    regions = partition_activity(activity,24)
    masses = [activity[y0:y1,x0:x1].sum() for y0,x0,y1,x1 in regions]
    assert max(masses)/min(masses) < 1.5
    activity[:,:24] = 10
    regions = partition_activity(activity,24)
    assert sum((r[1]+r[3])/2 < 24 for r in regions) > 12


@pytest.mark.parametrize('tiled',[True,False])
def test_sparse_weights_and_color_solver(tiled):
    torch.manual_seed(7)
    means = torch.rand(13,2)
    scales = torch.rand(13,2)*2+1
    rot = torch.rand(13,1)
    colors = torch.rand(13,3)
    xy,radius,conic,hits = cpu.project_gaussians_2d_scale_rot(means,scales,rot,17,19,(2,2,1))
    weights = renderer_weights(xy,radius,conic,17,19,tiled=tiled)
    if tiled:
        rendered = cpu.rasterize_gaussians_sum(xy,radius,conic,hits,colors,17,19,topk_norm=True)
    else:
        rendered = cpu.rasterize_gaussians_no_tiles(xy,conic,colors,17,19)
    np.testing.assert_allclose(weights@colors.numpy(),rendered.reshape(-1,3).numpy(),atol=2e-7)
    target = weights@colors.numpy()
    fitted,history = fit_colors(weights,target,np.zeros_like(colors.numpy()),sweeps=30)
    assert np.all(np.diff(history) <= 1e-12)
    assert history[-1] < history[0]*0.01
    with pytest.raises(ValueError,match='exceeds'):
        renderer_weights(xy,radius,conic,17,19,max_entries=1)


def test_zero_weight_columns_and_local_updates():
    matrix = csr_matrix([[1.,0.,0.],[0.,1.,0.]])
    colors = np.zeros((3,3),dtype=np.float32)
    fitted,history = fit_colors(matrix,np.ones((2,3)),colors,2,indices=[0,2])
    np.testing.assert_allclose(fitted[0],1)
    np.testing.assert_array_equal(fitted[1:],0)
    assert history[-1] < history[0]


@pytest.mark.parametrize('inverse', [True,False])
def test_zero_step_full_budget_checkpoint(tmp_path,inverse):
    from model import GaussianSplatting2D
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        image = image_pattern('edges',16,24)
        path = tmp_path/'input.png'
        Image.fromarray((image.transpose(1,2,0)*255).astype(np.uint8)).save(path)
        args = load_cfg('cfgs/default.yaml',argparse.ArgumentParser()).parse_args([])
        args.device='cpu'; args.init_mode='laplacian'; args.num_gaussians=17
        args.input_path=str(path); args.log_dir=str(tmp_path/'run')
        args.max_steps=0; args.quantize=True; args.color_fit_sweeps=3
        args.partition_visualize=True; args.disable_inverse_scale=not inverse
        args.residual_refine_count=1
        model = GaussianSplatting2D(args)
        assert model.num_gaussians == 17 and model.max_steps == 0
        assert model.disable_prog_optim
        model.optimize()
        assert model.step == 0
        stats=json.loads((tmp_path/'run/encoding_metrics.json').read_text())
        assert stats['adam_steps'] == 0 and stats['adam_seconds'] == 0
        assert stats['direct_fitting_seconds'] > 0 and stats['psnr'] > 0
        assert (tmp_path/'run/partition.png').exists()
        args.eval=True
        restored=GaussianSplatting2D(args)
        torch.testing.assert_close(model._render_images(),restored._render_images())
    finally:
        torch.set_num_threads(previous)


def test_short_run_and_skip_adam_do_not_expand_schedule(tmp_path):
    from model import GaussianSplatting2D
    path=tmp_path/'input.png'
    Image.fromarray((image_pattern('gradient',8,9).transpose(1,2,0)*255).astype(np.uint8)).save(path)
    for mode,skip in [('laplacian',False),('gradient',True)]:
        args=load_cfg('cfgs/default.yaml',argparse.ArgumentParser()).parse_args([])
        args.device='cpu'; args.init_mode=mode; args.skip_adam=skip
        args.num_gaussians=7; args.max_steps=300
        args.input_path=str(path); args.log_dir=str(tmp_path/mode)
        model=GaussianSplatting2D(args)
        assert model.num_gaussians == 7
        assert model.max_steps == (0 if skip else 300)
        assert args.max_steps == 300


def test_color_fit_rejects_unsupported_blending_and_renderer_mismatch(tmp_path, monkeypatch):
    from model import GaussianSplatting2D
    path=tmp_path/'input.png'
    Image.fromarray((image_pattern('gradient',8,9).transpose(1,2,0)*255).astype(np.uint8)).save(path)
    args=load_cfg('cfgs/default.yaml',argparse.ArgumentParser()).parse_args([])
    args.device='cpu'; args.init_mode='laplacian'; args.num_gaussians=7
    args.max_steps=0; args.input_path=str(path); args.log_dir=str(tmp_path/'run')
    model=GaussianSplatting2D(args)
    model.disable_topk_norm=True
    with pytest.raises(ValueError,match='requires top-K'):
        model._fitting_weights()
    model.disable_topk_norm=False
    render=model._render_images
    monkeypatch.setattr(model,'_render_images',lambda: render()+1)
    with pytest.raises(RuntimeError,match='disagree'):
        model._fitting_weights()
