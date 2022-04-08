# senegal-lcluc-tensorflow

Senegal LCLUC TensorFlow

## Downloading the Container

```bash
singularity build --sandbox /lscratch/jacaraba/container/tf-container docker://gitlab.nccs.nasa.gov:5050/nccs-ci/nccs-containers/rapids-tensorflow/nccs-ubuntu20-rapids-tensorflow
```

## Quick Start

```bash
module load singularity
singularity shell --nv -B /att,/lscratch,/adapt/nobackup/projects/ilab,/adapt/nobackup/people,/lscratch/jacaraba/tmp:/tmp /lscratch/jacaraba/container/tf-container/
```
