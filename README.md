# senegal-lcluc-tensorflow

![Code style: PEP8](https://github.com/nasa-cisto-ai/senegal-lcluc-tensorflow/actions/workflows/lint.yml/badge.svg)

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
