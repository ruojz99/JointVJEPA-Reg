"""Dataset readers.

AbdomenMRCT (Learn2Reg 2021): the 8 intra-patient MR-CT pairs.

mu-RegPro (Zenodo 8004388): 73 pairs in total; 65 training and 8 validation pairs of prostate MR and transrectal ultrasound.
"""
import json
import os

import nibabel as nib
import numpy as np

ORGANS = {1: 'liver', 2: 'spleen', 3: 'rkidney', 4: 'lkidney'}


def abdomen_pairs(root):
    """(moving CT, fixed MR) paths."""
    meta = json.load(open(os.path.join(root, 'AbdomenMRCT_dataset.json')))
    full = lambda p: os.path.join(root, p.lstrip('./').replace('AbdomenMRCT_A', 'AbdomenMRCT_'))
    pairs = [(full(p['1']), full(p['0'])) for p in meta['training_paired_images']]        # '0' = MR, '1' = CT
    pairs += [(full(p['moving']), full(p['fixed'])) for p in meta['registration_val']]
    return list(dict.fromkeys(pairs))


def abdomen_case(mov_path, fix_path):
    return os.path.basename(mov_path).split('.')[0] + '_to_' + os.path.basename(fix_path).split('.')[0]


def abdomen_labels(img_path):
    return nib.load(img_path.replace('/imagesTr/', '/labelsTr/')).get_fdata().astype(np.int16)


def muregpro_cases(root, split):
    return sorted(f[:-len('.nii.gz')] for f in os.listdir(os.path.join(root, split, 'mr_images')))


def muregpro_load(root, split, case, kind):
    """kind: mr_images | mr_labels | us_images | us_labels -> (X, Y, Z[, C]) with axial slices along Z."""
    v = np.asarray(nib.load(os.path.join(root, split, kind, case + '.nii.gz')).dataobj)
    v = v[..., 0] if kind.endswith('images') else v > 0
    return np.ascontiguousarray(np.moveaxis(v, 0, 2))
