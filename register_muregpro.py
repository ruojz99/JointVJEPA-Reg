"""
mu-RegPro (MR fixed, transrectal ultrasound moving) 
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from jointreg import convex_adam as CA
from jointreg import data as D
from jointreg import metrics as M
from jointreg import preprocess as P
from jointreg import vjepa as VJ

SPACING = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.environ.get('MUREGPRO_ROOT', 'data/muRegPro'))
    ap.add_argument('--vjepa_repo', default=os.environ.get('VJEPA_REPO', 'vjepa2'),
                    help='clone of github.com/facebookresearch/vjepa2')
    ap.add_argument('--vjepa_ckpt', default=os.environ.get('VJEPA_CKPT', 'vjepa2_1_vitl_dist_vitG_384.pt'))
    ap.add_argument('--split', choices=['train', 'val'], default='val')
    ap.add_argument('--mode', choices=['joint', 'separate'], default='joint')
    ap.add_argument('--order', choices=['moving', 'fixed'], default='moving')
    ap.add_argument('--grids', type=int, nargs='+', default=[28], help='tokens along a 128-voxel slice edge')
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.5], help='ConvexAdam regularisation weight gamma')
    ap.add_argument('--grid_sps', type=int, nargs='+', default=[1], help='ConvexAdam control-grid spacing s')
    ap.add_argument('--num_iter', type=int, default=100)
    ap.add_argument('--n_pairs', type=int, default=0, help='first N cases of the split (0: all)')
    ap.add_argument('--out', default='results/muregpro.csv')
    args = ap.parse_args()

    enc = VJ.load_encoder(args.vjepa_repo, args.vjepa_ckpt)
    cases = D.muregpro_cases(args.data_root, args.split)
    cases = cases[:args.n_pairs] if args.n_pairs else cases
    fields = ['pair', 'mode', 'order', 'grid', 'tokens', 'lam', 'grid_sp', 'dice', 'hd95', 'tre', 'n_lm', 'sdlogj',
              'encode_s', 'forwards', 'time_s']
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        for case in cases:
            load = lambda kind: D.muregpro_load(args.data_root, args.split, case, kind)
            mr, mr_lab = load('mr_images'), load('mr_labels')
            us, us_lab = (P.centre_pad(load(k), mr.shape) for k in ('us_images', 'us_labels'))
            fix_n, mov_n, fov = P.preprocess_muregpro(mr, us)
            mask = fov.astype(np.float32)
            for G in args.grids:
                size = tuple(max(8, round(n * G / 128)) for n in mr.shape[:2])
                t0 = time.time()
                f, m = VJ.features(enc, fix_n, mov_n, mask, mask, size, args.mode,
                                   moving_first=args.order == 'moving')
                f, m = P.unit_energy(f, m, fov, fov)
                t_feat = time.time() - t0
                T = dict(VJ.LAST_TIMES)
                for lam in args.lambdas:
                    for gs in args.grid_sps:
                        t1 = time.time()
                        u = CA.convex_adam(f, m, lam, gs, args.num_iter)
                        t = t_feat + time.time() - t1
                        dice, hd, tre, n_lm = M.muregpro_metrics(u, us_lab, mr_lab, SPACING)
                        row = dict(pair=f'{args.split}_{case}', mode=args.mode,
                                   order=args.order if args.mode == 'joint' else '', grid=f'{size[0]}x{size[1]}',
                                   tokens=G, lam=lam, grid_sp=gs, dice=round(dice, 4), hd95=round(hd, 3),
                                   tre=round(tre, 3), n_lm=n_lm, sdlogj=round(M.sdlogj(u), 4),
                                   encode_s=round(T['encode'], 2), forwards=T['forwards'], time_s=round(t, 1))
                        writer.writerow(row)
                        fh.flush()
                        print(f"{row['pair']} {args.mode} {row['grid']:7s} gamma={lam:g} s={gs}  Dice {dice:.4f}  "
                              f"HD95 {hd:5.2f} mm  TRE {tre:5.2f} mm  encoder {T['encode']:.1f}s "
                              f"({T['forwards']} passes)", flush=True)
                del f, m
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
