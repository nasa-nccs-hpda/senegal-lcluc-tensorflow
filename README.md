# senegal-lcluc-tensorflow

[![DOI](https://zenodo.org/badge/474016543.svg)](https://zenodo.org/badge/latestdoi/474016543)
![CI Workflow](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/senegal-lcluc-tensorflow/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/senegal-lcluc-tensorflow?branch=main)

Senegal LCLUC TensorFlow

## Downloading the Container

```bash
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

## Quick Start

### Test

Generate metrics and statistics using test data.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-GMU-V2/batch1/eCAS-wCAS-otcb-40/eCAS-wCAS-otcb-40.yaml -t '/explore/nobackup/projects/3sl/labels/landcover/2m_all_fixed/*.tif' -s test
```

### Validate

Generate metrics and statistics using validation data.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -v '/explore/nobackup/projects/3sl/data/Validation/3sl-validation-database-20230412-all-three-agreed.gpkg'  -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-GMU-V2/batch1/eCAS-wCAS-otcb-30/eCAS-wCAS-otcb-30.yaml  -s validate
```

## Land Use 1D CNN Workflow

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landuse_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_use/configs/landuse.yaml --gee-account 'id-sl-senegal-service-account@ee-3sl-senegal.iam.gserviceaccount.com' --gee-key '/home/$USER/gee/ee-3sl-senegal-8fa70fe1c565.json' -s setup
```

## Land Cover

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/global_standardization_256_crop_4band_short.yaml -d /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/land_cover_512_otcb_50TS_cas-wcas-short.csv -s preprocess train predict
```

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/$USER/container/tensorflow-caney python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/8bit_scale_256_crop_4band_short.yaml -d /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/land_cover_512_otcb_50TS_cas-wcas-8bit-short.csv -s preprocess train predict
```
