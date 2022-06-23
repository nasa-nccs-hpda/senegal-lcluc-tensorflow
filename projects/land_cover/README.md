# Land Cover Experiment

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