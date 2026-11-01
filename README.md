# Cross-Modal Joint Encoding with a Video Foundation Model for Training-Free Multimodal Image Registration

Code for the ISBI 2027 paper. A frozen video foundation model, V-JEPA 2.1 (ViT-L/16), encodes the fixed and the moving
volume **together**: every clip holds two adjacent slices of each volume, moving frames first,
`[M_z, M_z+1, F_z, F_z+1]`, so the fixed and moving tokens attend to each other inside the encoder. The tokens are
reduced to 24 channels by a PCA shared by both volumes and registered with ConvexAdam. No network is trained or
fine-tuned.


## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/vjepa2.git     
# V-JEPA 2.1 ViT-L checkpoint 
export VJEPA_REPO=$PWD/vjepa2
export VJEPA_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export ABDOMEN_ROOT=/path/to/AbdomenMRCT        # Learn2Reg 2021, with AbdomenMRCT_dataset.json
export MUREGPRO_ROOT=/path/to/muRegPro          # Zenodo 8004388: train/ and val/ with {mr,us}_{images,labels}
```


## Usage

```bash
# AbdomenMRCT, the selected setting: 42 x 36 tokens, gamma 0.5, s 1
python register_abdomen.py --grids 42 --lambdas 0.5 --grid_sps 1 --out results/abdomen.csv
# mu-RegPro validation split, 28 x 28 tokens
python register_muregpro.py --split val --grids 28 --lambdas 0.5 --grid_sps 1 --out results/muregpro.csv
```

To reproduce the paper's tables and figures, run `scripts/run_paper.sh`; its last step, `cv_select.py`, prints the
cross-validated results.

## Acknowledgements

- ConvexAdam (Siebert et al.) for the solver
- DINO-Reg (Song et al., MICCAI 2024) for the AbdomenMRCT preprocessing
- V-JEPA 2.1 (Meta) for the encoder
- VoxCor (Tombak et al.) for the global translation
