"""Cross-validated selection of the token grid, gamma and s (offline, on the CSVs of register_*.py).
    python cv_select.py abdomen results/abdomen_*.csv
    python cv_select.py muregpro results/muregpro_*.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 0
ORGANS = ['liver', 'spleen', 'rkidney', 'lkidney']


def folds(n, k):
    return np.array_split(np.random.RandomState(SEED).permutation(n), k)


def load(files):
    d = pd.concat([pd.read_csv(f) for f in files])
    d['method'] = d['mode'] + np.where(d['mode'] == 'joint', '_' + d['order'].fillna('') + '_first', '')
    d = d.rename(columns={'tokens': 'g', 'grid_sp': 'gs'})
    return d.drop_duplicates(['method', 'pair', 'g', 'lam', 'gs'], keep='last')


def select(d, score, n_folds):
    pairs = sorted(d.pair.unique())
    fd = folds(len(pairs), n_folds)
    table, curves, chosen = [], [], []
    for method, dm in d.groupby('method'):
        votes = []
        for test in fd:                                      # per-fold choice of (grid, gamma, s)
            te = {pairs[i] for i in test}
            s = dm[~dm.pair.isin(te)].groupby(['g', 'lam', 'gs']).apply(score)
            votes.append(s.idxmax()[0])
        v = pd.Series(votes).value_counts()
        g = int(min(v[v == v.max()].index))                  # grid selected in most folds, ties -> coarser
        for k, test in enumerate(fd):
            te = {pairs[i] for i in test}
            tr, out = dm[~dm.pair.isin(te)], dm[dm.pair.isin(te)]
            s = tr[tr.g == g].groupby(['lam', 'gs']).apply(score)
            lam, gs = s.idxmax()
            chosen.append(dict(method=method, fold=k, grid=g, lam=lam, gs=gs, grid_votes=votes[k]))
            table.append(out[(out.g == g) & (out.lam == lam) & (out.gs == gs)].assign(fold=k))
            curves.append(out[(out.lam == lam) & (out.gs == gs)].assign(fold=k))
    table, curves = pd.concat(table), pd.concat(curves)
    assert table.groupby('method').pair.nunique().eq(len(pairs)).all()
    return table, curves, pd.DataFrame(chosen)


def main():
    kind, files = sys.argv[1], sys.argv[2:]
    d = load(files)
    if kind == 'abdomen':
        dc = [f'dice_{o}' for o in ORGANS]
        score = lambda r: np.nanmean(r[dc].to_numpy(float))
        table, curves, chosen = select(d, score, 4)
        per_pair = lambda t: t.set_index('pair').sort_index()[dc].mean(axis=1)
        summary = table.groupby('method').apply(lambda r: pd.Series(dict(
            dice=np.nanmean(r[dc].to_numpy(float)),
            hd95=np.nanmean(r[[f'hd95_{o}' for o in ORGANS]].to_numpy(float)),
            sdlogj=r.sdlogj.mean(), encode_s=r.encode_s.mean(), forwards=r.forwards.mean())))
        curve = curves.groupby(['method', 'g']).apply(score).unstack()
    else:
        score = lambda r: r.dice.mean()
        table, curves, chosen = select(d, score, 5)
        per_pair = lambda t: t.set_index('pair').sort_index().dice
        summary = table.groupby('method').agg(dice=('dice', 'mean'), hd95=('hd95', 'mean'), tre=('tre', 'mean'),
                                              sdlogj=('sdlogj', 'mean'), encode_s=('encode_s', 'mean'),
                                              forwards=('forwards', 'mean'))
        curve = curves.groupby(['method', 'g']).dice.mean().unstack()
    pd.set_option('display.width', 200)
    print('selected per fold:')
    print(chosen.to_string(index=False))
    print('\ncross-validated results:')
    print(summary.round(4).to_string())
    print('\nDice against the token grid (out of fold):')
    print(curve.round(4).to_string())
    methods = sorted(table.method.unique())
    if len(methods) > 1:
        print('\npaired Wilcoxon over pairs (Dice; TRE for mu-RegPro):')
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                ta, tb = table[table.method == a], table[table.method == b]
                da, db = per_pair(ta), per_pair(tb)
                s = f'  {a} vs {b}: dDice {np.nanmean(da - db):+.4f} p={wilcoxon(da, db, nan_policy="omit").pvalue:.4f}'
                if kind == 'muregpro':
                    ra, rb = ta.set_index('pair').sort_index().tre, tb.set_index('pair').sort_index().tre
                    s += f'  dTRE {np.nanmean(ra - rb):+.3f} p={wilcoxon(ra, rb, nan_policy="omit").pvalue:.4f}'
                print(s)


if __name__ == '__main__':
    main()
