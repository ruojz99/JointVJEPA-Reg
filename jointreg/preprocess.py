"""Preprocessing of both datasets.
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops


# ── AbdomenMRCT ────────────────────────────────────────────────────

def _window(img, wl, ww):
    return (np.clip(img, wl - ww / 2, wl + ww / 2) - (wl - ww / 2)) / ww


def _mr_normalize(img, quantile=97):
    top = np.percentile(img, quantile)
    img = np.clip(img, img.min(), top)
    return (img - img.min()) / (top - img.min())


def _slice_mask(image, threshold):
    """Foreground of one slice: threshold, largest component, holes filled, 5 x 5 opening, largest component."""
    thresh = np.where(image > threshold, 1, 0).astype(np.uint8)
    if np.sum(thresh) <= 1000:
        return thresh
    lab, _ = ndimage.label(thresh)
    counts = np.bincount(lab.flat)[1:]
    largest = (lab == (np.argmax(counts) + 1 if counts.size > 0 else 0)).astype(np.uint8)
    filled = ndimage.binary_fill_holes(largest).astype(np.uint8)
    opened = ndimage.binary_opening(filled, structure=np.ones((5, 5), dtype=np.uint8))
    lab = label(opened)
    best, area = None, 0
    for region in regionprops(lab):
        if region.area > area:
            best, area = region, region.area
    out = np.zeros_like(image)
    if best is not None:
        out[lab == best.label] = 1
    return out


def preprocess_abdomen(mov_ct, fix_mr):
    """-> (moving CT, fixed MR, kept slice indices, moving mask, fixed mask), the volumes and masks on the kept
    slices only."""
    keep = [i for i in range(fix_mr.shape[2]) if fix_mr[:, :, i].max() != fix_mr[:, :, i].min()]
    fix = _mr_normalize(fix_mr[:, :, keep])
    mov = _window(mov_ct[:, :, keep], wl=50, ww=400)
    mask_fix, mask_mov = np.zeros_like(fix), np.zeros_like(mov)
    for k in range(fix.shape[2]):
        mask_fix[:, :, k] = _slice_mask(fix[:, :, k], 0.05)
        mask_mov[:, :, k] = _slice_mask(mov[:, :, k], 0.005)
    return mov, fix, keep, mask_mov, mask_fix


# ── mu-RegPro ─────────────────────────────────────────────────────────────────

def centre_pad(v, shape):
    pad = [((a - b) // 2, a - b - (a - b) // 2) for a, b in zip(shape, v.shape[:3])]
    return np.pad(v, pad + [(0, 0)] * (v.ndim - 3))


def preprocess_muregpro(mr, us):
    """MR (fixed) and centre-padded ultrasound (moving) -> (fixed, moving, common field-of-view mask)."""
    fov = binary_closing(us > 0, iterations=2) & binary_closing(mr > 0, iterations=2)
    us_n = (us / 255.0 * fov).astype(np.float32)
    mr_n = (np.clip(mr / np.percentile(mr[fov], 99), 0, 1) * fov).astype(np.float32)
    return mr_n, us_n, fov


# ── shared ────────────────────────────────────────────────────────────────────

def shifted(vol, b):
    """out(x) = vol(x + b) for an integer shift b (one entry per leading axis), zero outside."""
    out = np.zeros_like(vol)
    src = tuple(slice(max(v, 0), n + min(v, 0)) for v, n in zip(b, vol.shape))
    dst = tuple(slice(max(-v, 0), n + min(-v, 0)) for v, n in zip(b, vol.shape))
    out[dst] = vol[src]
    return out


def unit_energy(f_fix, f_mov, m_fix, m_mov):
    """Scale both feature volumes by one factor so that the mean squared value inside the masks is 1."""
    e = 0.5 * (np.mean(f_fix[m_fix] ** 2) + np.mean(f_mov[m_mov] ** 2))
    s = np.float32(1.0 / np.sqrt(e))
    return f_fix * s, f_mov * s
