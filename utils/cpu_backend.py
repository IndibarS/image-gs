"""Differentiable, tiled PyTorch reference backend for Image-GS.

Matches the custom kernels' pixel coordinates, covariance, tile bounds and
normalization. Chunking and activation recomputation avoid an N x H x W
allocation. Floating point rounding and tied top-k selections can differ.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def tile_bbox(xys, radii, height, width, block_h=16, block_w=16):
    block = xys.new_tensor([block_w, block_h])
    limit = xys.new_tensor([(width + block_w - 1) // block_w,
                            (height + block_h - 1) // block_h])
    low = torch.trunc((xys - radii[:, None]) / block).clamp_min(0)
    high = torch.trunc((xys + radii[:, None]) / block + 1).clamp_min(0)
    return torch.minimum(low, limit).long(), torch.minimum(high, limit).long()


def project_gaussians_2d_scale_rot(means, scales, rotation, height, width, tile_bounds):
    xy = means * means.new_tensor([width, height])
    c, s = rotation[:, 0].cos(), rotation[:, 0].sin()
    sx, sy = scales.square().unbind(-1)
    a, b, d = c*c*sx + s*s*sy, c*s*(sy-sx), s*s*sx + c*c*sy
    det = a*d - b*b
    valid = torch.isfinite(det) & (det > 0)
    safe_det = torch.where(valid, det, torch.ones_like(det))
    conics = torch.stack([d, -b, a], -1) / safe_det[:, None]
    conics = torch.where(valid[:, None], conics, torch.zeros_like(conics))
    with torch.no_grad():
        mid = (a + d) / 2
        radius = (3 * (mid + (mid.square()-det).clamp_min(0.1).sqrt()).clamp_min(0).sqrt()).ceil()
        radius = torch.where(valid, radius, torch.zeros_like(radius)).long()
        low, high = tile_bbox(xy, radius, height, width)
        hits = (high-low).prod(-1) * valid
    return xy, radius, conics, hits


def topk_weights(xys, conics, ids, pixels, topk=10, eps=1e-4):
    """Exact reference-renderer contributor IDs and effective normalized weights."""
    best = xys.new_zeros((pixels.shape[0], 0))
    best_ids = ids.new_empty((pixels.shape[0], 0))
    for chunk in ids.split(512):
        weight = gaussian_weights(xys, conics, chunk, pixels)
        joined = torch.cat([best, weight], -1)
        joined_ids = torch.cat([best_ids, chunk.expand(pixels.shape[0], -1)], -1)
        best, indices = joined.topk(min(topk, joined.shape[-1]), dim=-1)
        best_ids = joined_ids.gather(-1, indices)
    return best_ids, best / (best.sum(-1, keepdim=True) + eps)


def gaussian_weights(xys, conics, ids, pixels):
    delta = xys[ids][None] - pixels[:, None]
    dx, dy = delta.unbind(-1)
    a, b, c = conics[ids].unbind(-1)
    sigma = 0.5*(a*dx.square() + c*dy.square()) + b*dx*dy
    valid = torch.isfinite(sigma) & (sigma >= 0)
    return torch.exp(-torch.where(valid, sigma, torch.zeros_like(sigma))) * valid


def _tile(xys, conics, colors, ids, pixels, normalize, eps, topk):
    total = colors.new_zeros((pixels.shape[0], colors.shape[1]))
    if normalize:
        best_ids, weights = topk_weights(xys, conics, ids, pixels, topk, eps)
        total = (weights[..., None] * colors[best_ids]).sum(1)
    else:
        for chunk in ids.split(512):
            total = total + gaussian_weights(xys, conics, chunk, pixels) @ colors[chunk]
    return total + (xys.sum() + conics.sum() + colors.sum()) * 0


def _rasterize(xys, radii, conics, colors, height, width, block_h, block_w, normalize, tiled, topk):
    low, high = tile_bbox(xys.detach(), radii, height, width, block_h, block_w)
    rows = []
    for y in range(0, height, block_h):
        tiles = []
        for x in range(0, width, block_w):
            h, w = min(block_h, height-y), min(block_w, width-x)
            if tiled:
                tx, ty = x // block_w, y // block_h
                mask = (low[:, 0] <= tx) & (high[:, 0] > tx) & (low[:, 1] <= ty) & (high[:, 1] > ty) & (radii > 0)
                ids = mask.nonzero().flatten()
            else:
                ids = torch.arange(xys.shape[0], device=xys.device)
            yy, xx = torch.meshgrid(torch.arange(y, y+h, device=xys.device),
                                    torch.arange(x, x+w, device=xys.device), indexing='ij')
            pixels = torch.stack([xx, yy], -1).reshape(-1, 2).to(xys)
            def render_tile(xy, co, feat, ids=ids, pixels=pixels):
                return _tile(xy, co, feat, ids, pixels, normalize, 1e-4 if tiled else 1e-7, topk)
            if torch.is_grad_enabled() and any(t.requires_grad for t in (xys, conics, colors)):
                tile = checkpoint(render_tile, xys, conics, colors, use_reentrant=False)
            else:
                tile = render_tile(xys, conics, colors)
            tiles.append(tile.reshape(h, w, -1))
        rows.append(torch.cat(tiles, dim=1))
    return torch.cat(rows, dim=0)


def rasterize_gaussians_sum(xys, radii, conics, hits, colors, height, width,
                           block_h=16, block_w=16, topk_norm=False, *, topk=10):
    return _rasterize(xys, radii, conics, colors, height, width, block_h, block_w, topk_norm, True, topk)


def rasterize_gaussians_no_tiles(xys, conics, colors, height, width, *, topk=10):
    # The original no-tiles kernel always normalizes the top ten weights.
    return _rasterize(xys, torch.zeros_like(xys[:, 0]), conics, colors, height, width, 16, 16, True, False, topk)


def ssim(image, target):
    """11x11 Gaussian SSIM with zero padding, as in fused-ssim."""
    coords = torch.arange(11, dtype=image.dtype, device=image.device) - 5
    kernel = torch.exp(-coords.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(image.shape[1], 1, 11, 11)
    def blur(value):
        return F.conv2d(value, window, padding=5, groups=image.shape[1])
    mu1, mu2 = blur(image), blur(target)
    var1, var2 = blur(image.square())-mu1.square(), blur(target.square())-mu2.square()
    cov = blur(image*target)-mu1*mu2
    return (((2*mu1*mu2 + 0.01**2)*(2*cov + 0.03**2)) /
            ((mu1.square()+mu2.square()+0.01**2)*(var1+var2+0.03**2))).mean()
