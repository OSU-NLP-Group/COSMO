
# COSMO

Code for the paper ["Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs"](https://arxiv.org/pdf/2401.00608.pdf)

[**Project webpage**](https://osu-nlp-group.github.io/COSMO/)
**Authors**: Vardaan Pahuja, Weidi Luo, Yu Gu, Cheng-Hao Tu, Hong-You Chen, Tanya Berger-Wolf, Charles Stewart, Song Gao, Wei-Lun Chao, and Yu Su

## Installation

```
pip install -r requirements.txt
```

## Data Preprocessing

### iWildCam2020-WILDS
```
bash preprocess_iwildcam.sh
```
Note: The dir. `data/iwildcam_v2.0/train/` contains images for all splits.

### Snapshot Mountain Zebra
1. Download snapshot_mountain_zebra.zip from [this link](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EWI05mXQsopNskBo78a_l_ABSZJHl0uCsdNMu72aXmNNiA?e=LOtm5Q) and uncompress it into a directory `data/snapshot_mountain_zebra/`.
2. Download images using the command `gsutil -m cp -r "gs://public-datasets-lila/snapshot-safari/MTZ/MTZ_public" data/snapshot_mountain_zebra/`
2. Run `bash preprocess_mountain_zebra.sh`


## Training

Note: The below commands will use the DistMult model by default. Use the following hyperparameter configuration:

- For iWildCam2020-WILDS, set `DATA_DIR` to `data/iwildcam_v2.0/`, `IMG_DIR` to `data/iwildcam_v2.0/train/`, and `DATASET` to `iwildcam`
- For Snapshot Mountain Zebra, set `DATA_DIR` to `data/snapshot_mountain_zebra/` and `IMG_DIR` to `data/snapshot_mountain_zebra/`, and `DATASET` to `mountain_zebra`.
- For ConvE, use `--kg-embed-model conve --embedding-dim 200` in args.


### Image-only model (ERM baseline)
```
python -u run_image_only_model.py --data-dir DATA_DIR --img-dir IMG_DIR --save-dir CKPT_DIR > CKPT_DIR/log.txt
```

### COSMO, no-context baseline
```
python -u main.py --data-dir DATA_DIR --img-dir IMG_DIR --save-dir CKPT_DIR > CKPT_DIR/log.txt
```

### COSMO, taxonomy
```
python -u main.py --data-dir DATA_DIR --img-dir IMG_DIR --save-dir CKPT_DIR --add-id-id > CKPT_DIR/log.txt
```

### COSMO, location
```
python -u main.py --data-dir DATA_DIR --img-dir IMG_DIR --save-dir CKPT_DIR --add-image-location > CKPT_DIR/log.txt
```

### COSMO, time
```
python -u main.py --data-dir DATA_DIR --img-dir IMG_DIR/ --save-dir CKPT_DIR --add-image-time > CKPT_DIR/log.txt
```

### COSMO, taxonomy + location + time
```
python -u main.py --data-dir DATA_DIR --img-dir IMG_DIR --save-dir CKPT_DIR --add-id-id --add-image-time --add-image-location > CKPT_DIR/log.txt
```

## Evaluation

### Evaluate a model (specify split)
```
python eval.py --ckpt-path <PATH TO COSMO CKPT> --split test --data-dir DATA_DIR --img-dir IMG_DIR
```

## Error Analysis

### Taxonomy analysis
```
cd gen_utils/
python analyze_taxonomy_model.py --data-dir DATA_DIR --img-dir IMG_DIR --ckpt-1-path <PATH TO COSMO+TAXONOMY CKPT> --ckpt-2-path <PATH TO COSMO BASE CKPT>
```

### Plot location correlation analysis
```
cd gen_utils/
python analyze_img_loc.py --data-dir DATA_DIR
```

### Plot time correlation analysis
```
cd gen_utils/
python analyze_img_time.py --data-dir DATA_DIR
```


### Under-represented Species Analysis

#### Dump predictions for baseline image-only model
```
cd gen_utils/
python dump_imageonly_pred_specie_wise.py --ckpt-path <PATH TO IMAGE-ONLY MODEL> --split test --out-dir <OUT DIR>
```

#### Dump predictions for COSMO model
```
cd gen_utils/
python dump_kge_pred_specie_wise.py --ckpt-path <PATH TO COSMO MODEL> --split test --out-dir <OUT DIR>
```

#### Compare performance for under-represented species
```
cd gen_utils/
python eval_kge_specie_wise.py --y-pred-path-1 <PATH TO PREDICTIONS JSON FILE OF BASELINE MODEL> --y-pred-path-2 <PATH TO COSMO PREDICTIONS JSON FILE>
```
