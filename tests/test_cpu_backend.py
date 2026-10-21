import argparse

import numpy as np
import pytest
import torch
from PIL import Image

from utils import cpu_backend as cpu
from utils.flip import LDRFLIPLoss
from utils.misc_utils import load_cfg


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('normalize', [True, False])
def test_projection_and_rasterization_against_dense_reference(normalize):
    torch.manual_seed(4)
    means = (torch.rand(13, 2, dtype=torch.double)*0.4 + 0.3).requires_grad_()
    scales = (torch.rand(13, 2, dtype=torch.double)+2).requires_grad_()
    rot = torch.rand(13, 1, dtype=torch.double, requires_grad=True)
    colors = torch.rand(13, 3, dtype=torch.double, requires_grad=True)
    xy, radius, conic, hits = cpu.project_gaussians_2d_scale_rot(means, scales, rot, 8, 8, (1,1,1))
    c, s = rot[:, 0].cos(), rot[:, 0].sin()
    rotation = torch.stack([c, s, -s, c], -1).reshape(-1, 2, 2)
    cov = rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)
    inv = torch.linalg.inv(cov)
    torch.testing.assert_close(conic, torch.stack([inv[:,0,0],inv[:,0,1],inv[:,1,1]], -1))
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
    delta = torch.stack([xx, yy], -1).double().reshape(-1,1,2)-xy
    weights = torch.exp(-0.5*torch.einsum('pni,nij,pnj->pn', delta, inv, delta))
    if normalize:
        weights, ids = weights.topk(10, dim=1)
        expected = (weights[...,None]*colors[ids]).sum(1)/(weights.sum(1,keepdim=True)+1e-4)
    else:
        expected = weights @ colors
    actual = cpu.rasterize_gaussians_sum(xy, radius, conic, hits, colors, 8, 8, topk_norm=normalize)
    torch.testing.assert_close(actual.reshape(-1,3), expected)
    params = (means, scales, rot, colors)
    actual_grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), params)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)


def test_empty_tiles_and_no_tiles_backward():
    xy = torch.tensor([[200.,200.]], requires_grad=True)
    conic = torch.tensor([[1.,0.,1.]], requires_grad=True)
    color = torch.ones(1,3, requires_grad=True)
    image = cpu.rasterize_gaussians_sum(xy, torch.ones(1), conic, torch.zeros(1), color, 17, 19)
    assert image.shape == (17,19,3)
    assert image.count_nonzero() == 0
    image.sum().backward()
    assert torch.isfinite(xy.grad).all()
    result = cpu.rasterize_gaussians_no_tiles(xy, conic, color, 3, 4)
    result.sum().backward()
    assert torch.isfinite(result).all()


def test_ssim_and_flip_on_cpu():
    x = torch.rand(1,3,24,24, requires_grad=True)
    assert cpu.ssim(x,x).item() == pytest.approx(1)
    loss = 1-cpu.ssim(x, torch.zeros_like(x))
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert LDRFLIPLoss()(x.detach(),x.detach()).item() < 1e-5


@pytest.mark.parametrize('no_tiles', [False, True])
def test_cpu_optimize_checkpoint_and_render(tmp_path, no_tiles):
    from model import GaussianSplatting2D
    image = np.random.default_rng(1).integers(0,256,(24,24,3),dtype=np.uint8)
    path = tmp_path/'input.png'
    Image.fromarray(image).save(path)
    args = load_cfg('cfgs/default.yaml', argparse.ArgumentParser()).parse_args([])
    args.device = 'cpu'
    args.input_path = str(path)
    args.log_dir = str(tmp_path/'run')
    args.num_gaussians = 8
    args.max_steps = 2
    args.disable_prog_optim = True
    args.disable_lr_schedule = True
    args.disable_tiles = no_tiles
    args.eval_steps = 1
    args.quantize = True
    model = GaussianSplatting2D(args)
    before = model.xy.detach().clone()
    model.optimize()
    assert not torch.equal(before, model.xy)
    assert (tmp_path/'run/checkpoints/ckpt_step-2.pt').exists()
    assert np.isfinite(model.flip_final)
    args.eval = True
    restored = GaussianSplatting2D(args)
    torch.testing.assert_close(restored.xy, model.xy)
    restored.render(render_height=28)
    assert list((tmp_path/'run/eval').glob('render_*.jpg'))


def test_topk_across_gaussian_chunks():
    torch.manual_seed(12)
    xy = torch.rand(520,2)*4
    conic = torch.tensor([1.,0.,1.]).expand(520,-1)
    colors = torch.rand(520,3, requires_grad=True)
    result = cpu.rasterize_gaussians_no_tiles(xy, conic, colors, 4, 4)
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    pixels = torch.stack([xx,yy],-1).reshape(-1,1,2)
    weights = torch.exp(-0.5*(pixels-xy).square().sum(-1))
    best, ids = weights.topk(10,dim=-1)
    expected = (best[...,None]*colors[ids]).sum(1)/(best.sum(1,keepdim=True)+1e-7)
    torch.testing.assert_close(result.reshape(-1,3), expected)
    result.sum().backward()
    assert torch.isfinite(colors.grad).all()


def test_progressive_optimization_on_constant_image(tmp_path):
    from model import GaussianSplatting2D
    path = tmp_path/'flat.png'
    Image.fromarray(np.full((24,24,3),128,dtype=np.uint8)).save(path)
    args = load_cfg('cfgs/default.yaml', argparse.ArgumentParser()).parse_args([])
    args.device = 'auto'
    args.input_path = str(path)
    args.log_dir = str(tmp_path/'run')
    args.num_gaussians = 8
    args.add_times = 1
    args.add_steps = 1
    args.post_min_steps = 1
    args.max_steps = 2
    args.eval_steps = 1
    model = GaussianSplatting2D(args)
    model.optimize()
    assert model.num_gaussians == 8
    assert (tmp_path/'run/checkpoints/ckpt_step-2.pt').exists()
