# Installation

```
pip install -r requirements.txt
```

# Data Preprocessing

## iWildCam2020-WILDS
```
bash preprocess_iwildcam.sh
```
Note: The dir. `data/iwildcam_v2.0/train/` contains images for all splits.

## Snapshot Mountain Zebra
1. Download snapshot_mountain_zebra.zip from [this link](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EWI05mXQsopNskBo78a_l_ABSZJHl0uCsdNMu72aXmNNiA?e=LOtm5Q) and uncompress it into a directory `data/snapshot_mountain_zebra/`.
2. Download images using the command `gsutil -m cp -r "gs://public-datasets-lila/snapshot-safari/MTZ/MTZ_public" data/snapshot_mountain_zebra/`
2. Run `bash preprocess_mountain_zebra.sh`


# Training

Note: The below commands will use the DistMult model by default. For ConvE, use `--kg-embed-model conve --embedding-dim 200` in args.

## Image-only model (ERM baseline)
```
python -u run_image_only_model.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR > CKPT_DIR/log.txt
```

## COSMO, no-context baseline
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR > CKPT_DIR/log.txt
```

## COSMO, image-id + taxonomy
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id > CKPT_DIR/log.txt
```

## COSMO, image-id + location
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-image-location > CKPT_DIR/log.txt
```

## COSMO, image-id + time
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-image-time > CKPT_DIR/log.txt
```

## COSMO, image-id + taxonomy + time
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id --add-image-time > CKPT_DIR/log.txt
```

## COSMO, image-id + taxonomy + location
```
python -u main.py --n_epochs 12 --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id --add-image-location > CKPT_DIR/log.txt
```

## Evaluate a model (specify split)
```
python eval.py --ckpt-path <PATH TO COSMO CKPT> --split test --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/
```

# Error Analysis

## Taxonomy analysis (Sec 5.3.1)
```
cd gen_utils/
python analyze_taxonomy_model.py --data-dir ../data/iwildcam_v2.0/ --img-dir ../data/iwildcam_v2.0/train/ --ckpt-1-path <PATH TO COSMO+TAXONOMY CKPT> --ckpt-2-path <PATH TO COSMO BASE CKPT>
```

## Plot location correlation analysis (Sec 5.3.2)
```
cd gen_utils/
python analyze_img_loc.py --data-dir ../data/iwildcam_v2.0/
```

## Plot time correlation analysis (Sec 5.3.2)
```
cd gen_utils/
python analyze_img_time.py --data-dir ../data/iwildcam_v2.0/
```


## Under-represented Species Analysis (Sec 5.3.3)

### dump predictions for baseline image-only model
```
cd gen_utils/
python dump_imageonly_pred_specie_wise.py --ckpt-path <PATH TO IMAGE-ONLY MODEL> --split test --out-dir <OUT DIR>
```

### dump predictions for COSMO model
```
cd gen_utils/
python dump_kge_pred_specie_wise.py --ckpt-path <PATH TO COSMO MODEL> --split test --out-dir <OUT DIR>
```

### compare performance for under-represented species
```
cd gen_utils/
python eval_kge_specie_wise.py --y-pred-path-1 <PATH TO PREDICTIONS JSON FILE OF BASELINE MODEL> --y-pred-path-2 <PATH TO COSMO PREDICTIONS JSON FILE>
```
