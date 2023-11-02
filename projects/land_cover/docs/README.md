# Land Cover Experiment

## Metadata

Labels: /adapt/nobackup/projects/3sl/labels/landcover/tappan
SRLite Processed: /adapt/nobackup/projects/ilab/data/srlite/products/srlite-0.9.10-07042022-mode-warp/07052022/Senegal

## Training Labels

- 1-0 trees
- 2-1 crop
- 3-2 other vegetation
- 4-3 water/shadow
- 5-4 burn
- 6-5 clouds
- 7-6 - nodata

## Container Access

```bash
module load singularity; singularity shell --nv -B /att,/lscratch,/adapt/nobackup/projects/ilab,/adapt/nobackup/people,/lscratch/jacaraba/tmp:/tmp /lscratch/jacaraba/container/tf-container/
```

## Running Scripts

Preprocessing
```bash
python scripts/preprocess.py -c configs/land_cover_test.yaml -d configs/land_cover_test.csv
```

Training

```bash
python scripts/train.py -c configs/land_cover_test.yaml -d configs/land_cover_test.csv
```

Prediction

```bash
python scripts/predict.py -c configs/land_cover_test.yaml -d configs/land_cover_test.csv
```