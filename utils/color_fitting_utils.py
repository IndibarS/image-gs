"""Sparse fixed-geometry L2 fitting using the reference renderer's exact weights."""
import numpy as np
import torch
from scipy.sparse import coo_matrix

from utils.cpu_backend import tile_bbox, topk_weights


@torch.no_grad()
def renderer_weights(xys, radii, conics, height, width, topk=10, tiled=True,
                     max_entries=20_000_000):
    """Return CSR W, O(H*W*K) storage, never a dense pixel x Gaussian array.

    Accept *already projected* geometry from the active backend. Selection and
    normalization are shared with the CPU renderer. Native CUDA may select tied
    contributors differently: callers must validate W@C against that renderer.
    """
    n = len(xys)
    if height*width*min(topk,n) > max_entries:
        raise ValueError('Color-fit sparse matrix exceeds color_fit_max_entries; reduce resolution/budget/topk or increase the explicit limit')
    xys, conics, radii = xys.detach().cpu(), conics.detach().cpu(), radii.detach().cpu()
    low, high = tile_bbox(xys, radii, height, width)
    rows, columns, values = [], [], []
    for y in range(0,height,16):
        for x in range(0,width,16):
            if tiled:
                mask = (low[:,0] <= x//16) & (high[:,0] > x//16) & (low[:,1] <= y//16) & (high[:,1] > y//16) & (radii > 0)
                ids = mask.nonzero().flatten()
            else:
                ids = torch.arange(n)
            yy, xx = torch.meshgrid(torch.arange(y,min(y+16,height)),torch.arange(x,min(x+16,width)),indexing='ij')
            pixels = torch.stack([xx,yy],-1).reshape(-1,2).to(xys)
            selected, weight = topk_weights(xys,conics,ids,pixels,topk,1e-4 if tiled else 1e-7)
            row = (yy*width+xx).reshape(-1,1).expand_as(selected)
            keep = weight > 0
            rows.append(row[keep].numpy().astype(np.int32))
            columns.append(selected[keep].numpy().astype(np.int32))
            values.append(weight[keep].numpy())
    return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(height*width,n)).tocsr()


def fit_colors(weights, target, colors, sweeps=3, epsilon=1e-10, indices=None):
    """Cyclic coordinate descent; update residual immediately after each column.

    Fits unquantized colors to squared error, not L1+SSIM. Quantizing the result
    can increase error and is deliberately left to the normal encoding path.
    """
    if sweeps < 0 or epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError('Fitting sweeps must be nonnegative and epsilon positive')
    matrix = weights.tocsc()
    colors = np.array(colors,dtype=np.float64,copy=True)
    target = np.asarray(target,dtype=np.float64)
    if target.shape != (matrix.shape[0], colors.shape[1]) or len(colors) != matrix.shape[1]:
        raise ValueError('Incompatible color-fitting shapes')
    if not np.isfinite(target).all() or not np.isfinite(colors).all() or not np.isfinite(matrix.data).all():
        raise ValueError('Fitting inputs must be finite')
    residual = target - matrix @ colors
    history = [float(np.mean(residual**2))]
    indices = range(len(colors)) if indices is None else indices
    for _ in range(sweeps):
        for i in indices:
            start,end = matrix.indptr[i:i+2]
            rows, data = matrix.indices[start:end], matrix.data[start:end].astype(np.float64)
            denominator = data @ data
            if denominator <= epsilon:
                continue
            delta = (data[:,None]*residual[rows]).sum(0)/(denominator+epsilon)
            colors[i] += delta
            residual[rows] -= data[:,None]*delta
        history.append(float(np.mean(residual**2)))
    return colors.astype(np.float32), history
