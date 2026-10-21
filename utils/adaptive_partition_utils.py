"""CPU image analysis and deterministic, budget-constrained binary partitioning."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel


@dataclass
class GaussianInitialization:
    regions: np.ndarray  # half-open y0, x0, y1, x1
    positions: np.ndarray  # normalized x, y
    scales: np.ndarray  # physical pixel standard deviations (not reciprocals)
    rotations: np.ndarray  # clockwise radians, matching gsplat's column-major R
    features: np.ndarray
    activity: np.ndarray


def activity_map(image, smoothing=0.8, epsilon=0.05, alpha=0.0, beta=1.0):
    """CHW float image -> positive HW activity, preserving chromatic edges.

    Each derivative magnitude is averaged over channels then normalized by its
    spatial mean. epsilon is therefore relative to mean edge activity, not input
    bit depth. alpha=0 is pure Laplacian; alpha>0 enables the hybrid mode.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError('Expected a nonempty finite CHW image')
    if not all(np.isfinite(v) and v >= 0 for v in (smoothing, epsilon, alpha, beta)):
        raise ValueError('Activity parameters must be finite and nonnegative')
    smoothed = gaussian_filter(image, (0, smoothing, smoothing), mode='reflect') if smoothing else image
    activity = np.full(image.shape[1:], epsilon, dtype=np.float64)
    for weight, kind in ((alpha, 'gradient'), (beta, 'laplacian')):
        if not weight:
            continue
        if kind == 'laplacian':
            # Do not apply the spatial Laplacian along the channel dimension.
            magnitude = np.mean([np.abs(laplace(channel, mode='reflect')) for channel in smoothed], axis=0)
        else:
            magnitude = np.mean([np.hypot(sobel(channel, axis=0, mode='reflect'),
                                          sobel(channel, axis=1, mode='reflect')) for channel in smoothed], axis=0)
        mean = magnitude.mean()
        if mean > np.finfo(float).eps:
            activity += weight * magnitude / mean
    if activity.sum() == 0:
        activity.fill(1)
    return activity / activity.mean()


def integral_image(activity):
    return np.pad(np.asarray(activity, dtype=np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def region_sum(table, region):
    y0, x0, y1, x1 = region
    return table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]


def partition_activity(activity, count, strategy='balanced'):
    """Exactly count covering leaves; reject count > pixels rather than duplicate.

    Child budgets constrain feasible cuts. Search cumulative mass with integral
    queries (O(log(side)) per split), then trade mass error against leaf-adjusted
    aspect ratio. Indivisible pixels impose an unavoidable activity imbalance.
    """
    activity = np.asarray(activity, dtype=np.float64)
    if activity.ndim != 2 or min(activity.shape) < 1 or not np.isfinite(activity).all() or (activity < 0).any():
        raise ValueError('Activity must be a finite nonnegative HW array')
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= activity.size:
        raise ValueError('Gaussian budget must be between 1 and the number of pixels')
    if strategy not in ('balanced', 'longest'):
        raise ValueError('partition_strategy must be balanced or longest')
    if not activity.any():
        activity = np.ones_like(activity)
    table = integral_image(activity)
    h, w = activity.shape
    stack, leaves = [((0, 0, h, w), count)], []
    while stack:
        region, budget = stack.pop()
        if budget == 1:
            leaves.append(region)
            continue
        y0, x0, y1, x1 = region
        height, width = y1-y0, x1-x0
        left_budget, right_budget = budget//2, budget-budget//2
        mass = region_sum(table, region)
        candidates = []
        for axis, length, cross in ((0, height, width), (1, width, height)):
            lo = max(1, (left_budget+cross-1)//cross)
            hi = min(length-1, length-(right_budget+cross-1)//cross)
            child_left, child_right = left_budget, right_budget
            if length < 2:
                continue
            if lo > hi:
                # At near pixel-sized budgets, a half-budget split may be
                # impossible (e.g. 8 leaves in 3x3). Adjust child budgets.
                cut = length//2
                child_left = int(np.clip(round(budget*cut/length),
                                        max(1,budget-(length-cut)*cross), min(budget-1,cut*cross)))
                child_right = budget-child_left
                lo = max(1,(child_left+cross-1)//cross)
                hi = min(length-1,length-(child_right+cross-1)//cross)
            axis_target = mass*child_left/budget
            def children(cut):
                if axis == 0:
                    return (y0,x0,y0+cut,x1), (y0+cut,x0,y1,x1)
                return (y0,x0,y1,x0+cut), (y0,x0+cut,y1,x1)
            start, end = lo, hi
            while start < end:
                mid = (start+end)//2
                if region_sum(table, children(mid)[0]) < axis_target:
                    start = mid+1
                else:
                    end = mid
            # Also consider the geometric split to avoid thin cuts on flat ties.
            cuts = {start, max(lo,start-1), min(hi,max(lo,round(length*child_left/budget)))}
            for cut in sorted(cuts):
                first, second = children(cut)
                error = abs(region_sum(table, first)-axis_target)/max(mass, 1e-15)
                # A child may be elongated if its remaining leaf budget can fix it.
                aspect = 0.0
                for child, n in ((first,child_left),(second,child_right)):
                    ch, cw = child[2]-child[0], child[3]-child[1]
                    aspect += max(0.0, np.log(max(ch/cw,cw/ch)/n))
                score = error + 0.08*aspect
                if strategy == 'longest':
                    score += 2.0 if length < cross else 0.0
                candidates.append((score, axis, cut, first, second, child_left, child_right))
        if not candidates:
            raise RuntimeError('No feasible split despite sufficient pixels')
        _, _, _, first, second, left_budget, right_budget = min(candidates)
        stack.append((second,right_budget))
        stack.append((first,left_budget))
    return np.asarray(leaves, dtype=np.int64)


def initialize_gaussians(image, count, smoothing=0.8, epsilon=0.05, alpha=0.0,
                         beta=1.0, strategy='balanced', scale_factor=1.5, min_scale=0.35):
    if not np.isfinite(scale_factor) or scale_factor <= 0 or not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError('Scale calibration and minimum scale must be finite and positive')
    image = np.asarray(image, dtype=np.float64)
    activity = activity_map(image, smoothing, epsilon, alpha, beta)
    regions = partition_activity(activity, count, strategy)
    _, height, width = image.shape
    positions, scales, rotations, features = [], [], [], []
    for y0,x0,y1,x1 in regions:
        weights = activity[y0:y1,x0:x1]
        weights = weights / weights.sum() if weights.sum() > 0 else np.full_like(weights, 1/weights.size)
        yy, xx = np.ogrid[y0:y1,x0:x1]
        mx, my = (weights*xx).sum(), (weights*yy).sum()
        dx, dy = xx-mx, yy-my
        # Pixel footprints contribute 1/12 variance, even for single pixels.
        covariance = np.array([[(weights*dx*dx).sum()+1/12, (weights*dx*dy).sum()],
                               [(weights*dx*dy).sum(), (weights*dy*dy).sum()+1/12]])
        values, vectors = np.linalg.eigh(covariance)
        values, vector = values[::-1], vectors[:, -1]
        angle = -np.arctan2(vector[1], vector[0])
        angle = (angle + np.pi/2) % np.pi - np.pi/2
        if values[0]-values[1] <= 1e-6*max(1.0,values[0]):
            angle = 0.0
        positions.append([mx/width,my/height])
        scales.append(np.clip(scale_factor*np.sqrt(np.maximum(values,0)), min_scale, max(height,width)*scale_factor))
        rotations.append([angle])
        features.append(image[:,y0:y1,x0:x1].mean(axis=(1,2)))
    arrays = [np.asarray(a,dtype=np.float32) for a in (positions,scales,rotations,features)]
    return GaussianInitialization(regions, *arrays, activity)


def save_partition_preview(image, regions, path):
    from PIL import Image, ImageDraw
    rgb = image[:3]
    if len(rgb) < 3:
        rgb = np.repeat(rgb[:1],3,axis=0)
    canvas = Image.fromarray((np.clip(rgb.transpose(1,2,0),0,1)*255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for y0,x0,y1,x1 in regions:
        draw.rectangle((int(x0),int(y0),int(x1-1),int(y1-1)), outline=(255,0,255))
    canvas.save(path)
