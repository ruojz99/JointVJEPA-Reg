"""V-JEPA 2.1 features for registration: cross-modal joint encoding (and separate encoding as the ablation).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PATCH, TUBELET = 16, 2
N_CH = 24
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LAST_TIMES = {}                                    # seconds of the last features() call, and its forward passes


def load_encoder(repo, ckpt):
    """V-JEPA 2.1 ViT-L/16 encoder (torch hub `vjepa2_1_vit_large_384`, encoder only) """
    sys.path.insert(0, repo)
    from app.vjepa_2_1.models import vision_transformer as vt
    enc = vt.vit_large(patch_size=PATCH, img_size=(384, 384), num_frames=64, tubelet_size=TUBELET, use_sdpa=True,
                       use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                       img_temporal_dim_size=1, interpolate_rope=True)
    state = torch.load(ckpt, map_location='cpu')['ema_encoder']
    enc.load_state_dict({k.replace('module.', '').replace('backbone.', ''): v for k, v in state.items()}, strict=True)
    sys.path.remove(repo)
    return enc.cuda().eval()


def clip_starts(n, slices=TUBELET):
    idx = list(range(0, n - slices + 1, slices))
    return idx + ([n - slices] if idx[-1] != n - slices else [])


def _run(enc, x):
    """(frames, H, W) slices in [0, 1] -> (frames / 2 * fh * fw, 1024) tokens of the last encoder block."""
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1, 1)
    return enc((x[None, None].expand(1, 3, -1, -1, -1) - mean) / std)[0]


def _upsample(a, size):
    """(X, Y, S) volume -> (S, fh * 16, fw * 16) slices"""
    fh, fw = size
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(a, np.float32)).cuda().permute(2, 0, 1)[:, None],
                         (fh * PATCH, fw * PATCH), mode='bilinear', align_corners=False)[:, 0]


def _stack(layers):
    centres = sorted(layers)
    return (torch.stack([layers[c][0] for c in centres]), torch.stack([layers[c][1] for c in centres]),
            np.array(centres, np.float32))


@torch.no_grad()
def encode_separate(enc, vol, mask, size, half=True):
    """One volume in two-slice clips. Returns tokens (L, fh, fw, 1024), foreground flags (L, fh, fw) and
    the through-plane centre of every token layer."""
    fh, fw = size
    v, m = _upsample(vol, size), _upsample(mask.astype(np.float32), size)
    layers = {}
    for z in clip_starts(vol.shape[2]):
        with torch.autocast('cuda', dtype=torch.float16, enabled=half):
            t = _run(enc, v[z:z + TUBELET]).view(fh, fw, -1)
        inside = F.avg_pool2d(m[z:z + TUBELET][None], PATCH)[0] > 0.99
        layers.setdefault(z + 0.5, ((t.half() if half else t.float()).cpu(), inside.all(0).cpu()))
    return _stack(layers)


@torch.no_grad()
def encode_joint(enc, fix, mov, mask_fix, mask_mov, size, half=True, moving_first=True, slices=TUBELET):
    """Both volumes of a pair in one clip per position."""
    fh, fw = size
    assert slices % TUBELET == 0
    L = slices // TUBELET                                         # token layers per volume and clip
    vols = (_upsample(fix, size), _upsample(mov, size))
    masks = (_upsample(mask_fix.astype(np.float32), size), _upsample(mask_mov.astype(np.float32), size))
    layers = ({}, {})
    for z in clip_starts(fix.shape[2], slices):
        x = torch.cat([v[z:z + slices] for v in (vols[::-1] if moving_first else vols)])
        with torch.autocast('cuda', dtype=torch.float16, enabled=half):
            t = _run(enc, x).view(2, L, fh, fw, -1)
        t = t.flip(0) if moving_first else t                     # -> (fixed, moving)
        for side in range(2):
            for l in range(L):
                zl = z + l * TUBELET
                inside = F.avg_pool2d(masks[side][zl:zl + TUBELET][None], PATCH)[0] > 0.99
                layers[side].setdefault(zl + 0.5, ((t[side, l].half() if half else t[side, l].float()).cpu(),
                                                   inside.all(0).cpu()))
    return _stack(layers[0]), _stack(layers[1])


@torch.no_grad()
def _to_voxels(lat, centres, size, shape):
    """(L, fh, fw, C) token layers -> (X, Y, S, C)"""
    fh, fw = size
    g = lat.permute(1, 2, 0, 3).contiguous()                       # (fh, fw, L, C)
    axes = []
    for d, n_tok in enumerate((fh, fw)):                          # token k sits at pixel 8 + 16 k
        c = (torch.arange(shape[d], device='cuda') + 0.5) * n_tok * PATCH / shape[d]
        axes.append(2 * ((c - PATCH / 2) / PATCH) / (n_tok - 1) - 1)
    grid = torch.stack(torch.meshgrid(axes[0], axes[1], indexing='ij'), -1)[..., [1, 0]]
    src = g.permute(2, 3, 0, 1)                                   # (L, C, fh, fw)
    f = F.grid_sample(src, grid[None].expand(len(src), -1, -1, -1), mode='bilinear', padding_mode='border',
                      align_corners=True).permute(2, 3, 0, 1)      # (X, Y, L, C)
    z = torch.arange(shape[2], device='cuda', dtype=torch.float32)
    c = torch.from_numpy(centres).cuda()
    hi = torch.searchsorted(c, z).clamp(1, len(c) - 1)
    w = ((z - c[hi - 1]) / (c[hi] - c[hi - 1])).clamp(0, 1).view(1, 1, -1, 1)
    return (f[:, :, hi - 1] * (1 - w) + f[:, :, hi] * w).cpu().numpy()


@torch.no_grad()
def features(enc, fix_n, mov_n, mask_fix, mask_mov, size, mode='joint', moving_first=True, slices=TUBELET):
    """(fixed, moving) 24-channel feature volumes (X, Y, S, 24) on the grid of the preprocessed volumes.
    mode: 'joint' or 'separate'; size: (fh, fw) token grid."""
    torch.cuda.synchronize(); t0 = time.time()
    if mode == 'joint':
        enc_out = encode_joint(enc, fix_n, mov_n, mask_fix, mask_mov, size, moving_first=moving_first, slices=slices)
        forwards = len(clip_starts(fix_n.shape[2], slices))
    else:
        enc_out = [encode_separate(enc, v, m, size) for v, m in ((fix_n, mask_fix), (mov_n, mask_mov))]
        forwards = sum(len(clip_starts(v.shape[2])) for v in (fix_n, mov_n))
    torch.cuda.synchronize(); t1 = time.time()
    n, s, ss = 0, 0, 0                                            # exact PCA over the foreground tokens of both
    for tokens, fg, _ in enc_out:
        for t, g in zip(tokens, fg):
            x = t.cuda()[g.cuda()].double()
            n, s, ss = n + len(x), s + x.sum(0), ss + x.T @ x
    mean = s / n
    basis = torch.linalg.eigh(ss / n - torch.outer(mean, mean))[1][:, -N_CH:].flip(1).float()
    torch.cuda.synchronize(); t2 = time.time()
    out = []
    for (tokens, fg, centres), v in zip(enc_out, (fix_n, mov_n)):
        lat = torch.stack([(t.cuda().float() @ basis) * g.cuda()[..., None] for t, g in zip(tokens, fg)])
        out.append(_to_voxels(lat, centres, size, v.shape))
        del lat
    torch.cuda.empty_cache()
    LAST_TIMES.update(encode=t1 - t0, pca=t2 - t1, voxels=time.time() - t2, forwards=forwards)
    return out[0], out[1]
