"""
Learn2Reg AbdomenMRCT (MR fixed, CT moving) 
"""
import argparse
import csv
import json
import os
import time

import nibabel as nib
import numpy as np
import torch

from jointreg import convex_adam as CA
from jointreg import data as D
from jointreg import metrics as M
from jointreg import preprocess as P
from jointreg import vjepa as VJ

HERE = os.path.dirname(os.path.abspath(__file__))


def to_fixed_frame(f, b):
    """Moving-frame volume (X, Y, Z[, C]) -> fixed frame for the translation b: out(x) = f(x + b)"""
    idx = [np.clip(np.arange(n) + int(v), 0, n - 1) for n, v in zip(f.shape[:3], b)]
    return f[np.ix_(*idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.environ.get('ABDOMEN_ROOT', 'data/AbdomenMRCT'))
    ap.add_argument('--vjepa_repo', default=os.environ.get('VJEPA_REPO', 'vjepa2'),
                    help='clone of github.com/facebookresearch/vjepa2')
    ap.add_argument('--vjepa_ckpt', default=os.environ.get('VJEPA_CKPT', 'vjepa2_1_vitl_dist_vitG_384.pt'))
    ap.add_argument('--translations', default=os.path.join(HERE, 'data', 'abdomen_translations.json'))
    ap.add_argument('--mode', choices=['joint', 'separate'], default='joint')
    ap.add_argument('--order', choices=['moving', 'fixed'], default='moving',
                    help='joint clips: moving frames first (the paper\'s Eq. 1) or fixed frames first')
    ap.add_argument('--slices', type=int, default=2, help='joint clips: adjacent slices of each volume per clip')
    ap.add_argument('--grids', type=int, nargs='+', default=[42],
                    help='tokens along the longer in-plane axis; the grid is G x (6G/7), e.g. 42 -> 42 x 36')
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.5], help='ConvexAdam regularisation weight gamma')
    ap.add_argument('--grid_sps', type=int, nargs='+', default=[1], help='ConvexAdam control-grid spacing s')
    ap.add_argument('--num_iter', type=int, default=100)
    ap.add_argument('--pairs', type=int, nargs='*', default=None, help='indices into the 8 pairs (default: all)')
    ap.add_argument('--out', default='results/abdomen.csv')
    args = ap.parse_args()

    enc = VJ.load_encoder(args.vjepa_repo, args.vjepa_ckpt)
    shifts = json.load(open(args.translations))
    pairs = D.abdomen_pairs(args.data_root)
    organs = list(D.ORGANS)
    names = list(D.ORGANS.values())
    fields = (['pair', 'mode', 'order', 'slices', 'grid', 'tokens', 'lam', 'grid_sp', 'dice', 'hd95', 'sdlogj']
              + [f'dice_{n}' for n in names] + [f'hd95_{n}' for n in names] + ['encode_s', 'forwards', 'time_s'])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        for i in (args.pairs if args.pairs else range(len(pairs))):
            mov_path, fix_path = pairs[i]
            case = D.abdomen_case(mov_path, fix_path)
            fix_img = nib.load(fix_path)
            mov_arr = nib.load(mov_path).get_fdata(dtype=np.float32)          # CT (moving)
            fix_arr = fix_img.get_fdata(dtype=np.float32)                     # MR (fixed)
            spacing = np.sqrt((fix_img.affine[:3, :3] ** 2).sum(axis=0))
            mov_lab, fix_lab = D.abdomen_labels(mov_path), D.abdomen_labels(fix_path)
            shape = fix_arr.shape
            b = np.asarray(shifts[case], int)

            mov_n, fix_n, keep, mask_mov, mask_fix = P.preprocess_abdomen(mov_arr, fix_arr)

            def full(a):                                                      
                out = np.zeros(shape + a.shape[3:], np.float32)
                out[:, :, keep] = a
                return out
            m_fix = full(mask_fix) > 0.5
            m_mov = to_fixed_frame(full(mask_mov), b) > 0.5

            for G in args.grids:
                size = (G, G * 6 // 7)
                t0 = time.time()
                if args.mode == 'joint':                    # clips pair the slices of the translated CT
                    mov_t, mask_t = (P.shifted(full(a), b)[:, :, keep] for a in (mov_n, mask_mov))
                    f, m = VJ.features(enc, fix_n, mov_t, mask_fix, mask_t, size, 'joint',
                                       moving_first=args.order == 'moving', slices=args.slices)
                    m = P.shifted(full(m), tuple(-b))      
                else:
                    f, m = VJ.features(enc, fix_n, mov_n, mask_fix, mask_mov, size, 'separate')
                    m = full(m)
                f, m = P.unit_energy(full(f), to_fixed_frame(m, b), m_fix, m_mov)   
                t_feat = time.time() - t0
                T = dict(VJ.LAST_TIMES)
                for lam in args.lambdas:
                    for gs in args.grid_sps:
                        t1 = time.time()
                        u = CA.convex_adam(f, m, lam, gs, args.num_iter)
                        t = t_feat + time.time() - t1
                        u = (u.astype(np.float64) + b.reshape(3, 1, 1, 1)).astype(np.float32)   # field on the CT
                        dice, hd = M.abdomen_metrics(u, mov_lab, fix_lab, spacing, organs)
                        row = dict(pair=case, mode=args.mode, order=args.order if args.mode == 'joint' else '',
                                   slices=args.slices if args.mode == 'joint' else 2, grid=f'{size[0]}x{size[1]}',
                                   tokens=G, lam=lam, grid_sp=gs, dice=round(float(np.nanmean(dice)), 4),
                                   hd95=round(float(np.nanmean(hd)), 3), sdlogj=round(M.sdlogj(u), 4),
                                   encode_s=round(T['encode'], 2), forwards=T['forwards'], time_s=round(t, 1))
                        row.update({f'dice_{n}': round(v, 4) for n, v in zip(names, dice)})
                        row.update({f'hd95_{n}': round(v, 3) for n, v in zip(names, hd)})
                        writer.writerow(row)
                        fh.flush()
                        print(f"[{i}] {case[:28]} {args.mode} {row['grid']:7s} gamma={lam:g} s={gs}  Dice {row['dice']:.4f}  "
                              f"HD95 {row['hd95']:.2f} mm  SDlogJ {row['sdlogj']:.3f}  encoder {T['encode']:.1f}s "
                              f"({T['forwards']} passes)", flush=True)
                del f, m
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
