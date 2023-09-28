# Surface Reflectance Experiments

## Experiment #1: No normalization using Unfixed labels

### TOA

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/tensorflow-caney:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/NASA/2023-surface_reflectance/ard_srlite_toa/experiment1/toa_dirty_experiment1.yaml -d /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/NASA/2023-surface_reflectance/ard_srlite_toa/experiment1/toa_dirty.csv -s preprocess
```

### SRLite
/explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05

### ARD
/explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05


## Experiment #2: No normalization using Fixed labels

SRLite - no normalization - clean labels
TOA
ARD

## Experiment #3: Standardization using Unfixed labels

SRLite - standardization - dirty labels
TOA
ARD

## Experiment #4: Standardization using Fixed labels

SRLite - standardization - clean labels
TOA
ARD

## Experiment #5: Data augmentation using Fixed labels

SRLite - data augmentation - clean labels
TOA
ARD
