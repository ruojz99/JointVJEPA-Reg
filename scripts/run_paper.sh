#!/bin/bash
# Every run of the paper. 
# Needs VJEPA_REPO, VJEPA_CKPT, ABDOMEN_ROOT and MUREGPRO_ROOT (see README.md).
set -e
cd "$(dirname "$0")/.."

SWEEP="--lambdas 0.5 1 2 --grid_sps 1 2"

# ── AbdomenMRCT: token grid x gamma x s (Table 1, Fig. 2) ──
python register_abdomen.py --order moving --grids 28 42 56 70 84 112 $SWEEP --out results/abdomen_joint_moving.csv
python register_abdomen.py --order fixed  --grids 28 42 56 70 84 112 $SWEEP --out results/abdomen_joint_fixed.csv
python register_abdomen.py --mode separate --grids 28 42 56 70 84 112 $SWEEP --out results/abdomen_separate.csv

# ── AbdomenMRCT: clip length at 42 x 36 (Sec. 3.4) ──
for k in 2 4 8 16 32; do
    python register_abdomen.py --grids 42 --lambdas 0.5 --grid_sps 1 --slices $k --out results/cliplength_abdomen.csv
done

# ── mu-RegPro: token grid x gamma x s on all 73 pairs (Table 2, Fig. 3) ──
for split in train val; do
    python register_muregpro.py --split $split --order moving  --grids 14 21 28 42 56 70 84 112 $SWEEP --out results/muregpro_joint_moving.csv
    python register_muregpro.py --split $split --order fixed   --grids 14 21 28 42 56 70 84 112 $SWEEP --out results/muregpro_joint_fixed.csv
    python register_muregpro.py --split $split --mode separate --grids 14 21 28 42 56 70 84 112 $SWEEP --out results/muregpro_separate.csv
done

# ── cross-validated tables, grid curves and tests ──
python cv_select.py abdomen results/abdomen_joint_moving.csv results/abdomen_joint_fixed.csv results/abdomen_separate.csv
python cv_select.py muregpro results/muregpro_joint_moving.csv results/muregpro_joint_fixed.csv results/muregpro_separate.csv
