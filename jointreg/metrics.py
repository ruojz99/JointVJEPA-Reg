"""Evaluation. 
"""
import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates


def warp(img, disp, order=1, mode='nearest'):
    idx = np.meshgrid(*[np.arange(s) for s in img.shape], indexing='ij')
    return map_coordinates(img, [idx[i] + disp[i] for i in range(3)], order=order, mode=mode)


def hd95(a, b, spacing):
    """Symmetric 95th-percentile surface distance (mm) between two non-empty binary masks."""
    idx = np.argwhere(a | b)
    box = tuple(slice(max(lo - 2, 0), hi + 3) for lo, hi in zip(idx.min(0), idx.max(0)))
    a, b = a[box], b[box]
    sa, sb = a & ~binary_erosion(a), b & ~binary_erosion(b)
    d = np.concatenate([distance_transform_edt(~sb, sampling=spacing)[sa],
                        distance_transform_edt(~sa, sampling=spacing)[sb]])
    return float(np.percentile(d, 95))


def jacobian_determinant(disp):
    """det J of x + u(x) by central differences, without a 2-voxel border (Learn2Reg)."""
    disp = disp[None].astype(np.float64)
    kernels = [np.array([-0.5, 0, 0.5]).reshape(s) for s in ((1, 3, 1, 1), (1, 1, 3, 1), (1, 1, 1, 3))]
    grad = np.concatenate([np.stack([scipy.ndimage.correlate(disp[:, i], k, mode='constant', cval=0.0)
                                     for i in range(3)], axis=1) for k in kernels], 0)
    J = (grad + np.eye(3, 3).reshape(3, 3, 1, 1, 1))[:, :, 2:-2, 2:-2, 2:-2]
    return (J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
            - J[1, 0] * (J[0, 1] * J[2, 2] - J[0, 2] * J[2, 1])
            + J[2, 0] * (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]))


def sdlogj(disp):
    return float(np.log((jacobian_determinant(disp) + 3).clip(1e-9, 1e9))[2:-2, 2:-2, 2:-2].std())


def abdomen_metrics(disp, mov_lab, fix_lab, spacing, organs):
    """Per-organ Dice and HD95 (mm); NaN where an organ is missing."""
    near = warp(mov_lab, disp, order=0, mode='nearest')
    const = warp(mov_lab, disp, order=0, mode='constant')
    dice, hd = [], []
    for o in organs:
        a, b = fix_lab == o, near == o
        dice.append(float((2.0 * np.sum(a * b) + 1e-5) / (np.sum(a) + np.sum(b) + 1e-5)) if a.any() and b.any()
                    else np.nan)
        a, b = fix_lab == o, const == o
        hd.append(hd95(a, b, spacing) if a.any() and b.any() else np.nan)
    return dice, hd


def muregpro_metrics(disp, mov_lab, fix_lab, spacing=0.8):
    """Gland Dice, gland HD95 (mm), challenge TRE (mm), number of landmark regions used.
    Label channel 0 is the gland, the others are landmark regions."""
    a, b = fix_lab[..., 0], warp(mov_lab[..., 0].astype(np.float32), disp) > 0.5
    dice = float(2 * (a & b).sum() / (a.sum() + b.sum()))
    hd = hd95(a, b, (spacing,) * 3) if b.any() else np.nan
    mae = []
    for j in range(1, fix_lab.shape[-1]):
        if fix_lab[..., j].any() and mov_lab[..., j].any():
            w = warp(mov_lab[..., j].astype(np.float32), disp)
            if w.sum() > 0:
                cw = [(w.sum(axis=tuple(k for k in range(3) if k != i)) * np.arange(w.shape[i])).sum() / w.sum()
                      for i in range(3)]
                mae.append(np.abs(spacing * (np.array(cw) - np.argwhere(fix_lab[..., j]).mean(0))).mean())
    tre = float(np.sqrt(np.mean(np.square(mae)))) if mae else np.nan
    return dice, float(hd), tre, len(mae)
