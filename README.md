# senegal-lcluc-tensorflow

![CI Workflow](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/senegal-lcluc-tensorflow/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/senegal-lcluc-tensorflow/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/senegal-lcluc-tensorflow?branch=main)

Senegal LCLUC TensorFlow

## Downloading the Container

```bash
singularity build --sandbox /lscratch/jacaraba/container/tf-container docker://gitlab.nccs.nasa.gov:5050/nccs-ci/nccs-containers/rapids-tensorflow/nccs-ubuntu20-rapids-tensorflow
```

singularity build --sandbox /lscratch/jacaraba/container/tf-container docker://gitlab.nccs.nasa.gov:5050/cisto-ilab/gdal-containers/tf-container:latest

## Quick Start

```bash
module load singularity; singularity shell --nv -B /att,/lscratch,/adapt/nobackup/projects/ilab,/adapt/nobackup/people,/lscratch/jacaraba/tmp:/tmp /adapt/nobackup/projects/ilab/containers/tf-container-rapids;
source activate rapids
```

```bash
module load singularity; singularity shell --nv -B /att,/lscratch,/adapt/nobackup/projects/ilab,/adapt/nobackup/people,/lscratch/jacaraba/tmp:/tmp /lscratch/jacaraba/container/tf-container-rapids;
source activate rapids
```






