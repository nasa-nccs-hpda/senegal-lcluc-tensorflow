#!/bin/bash

CONFIG_REGEX=$1
DATABASE=$2
CUSTOM_PYTHON="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney"
CONTAINER="/explore/nobackup/projects/ilab/containers/above-shrubs.2023.07"

for config_file in $CONFIG_REGEX; do
    echo $config_file
    singularity exec --env PYTHONPATH=$CUSTOM_PYTHON --nv \
        -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/explore/nobackup/people $CONTAINER \
        python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
        -v $DATABASE \
        -c $config_file \
        -s validate &
done
