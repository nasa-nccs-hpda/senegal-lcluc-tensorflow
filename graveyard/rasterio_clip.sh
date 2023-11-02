#!/bin/bash

# v1
# bash rasterio_clip.sh '/adapt/nobackup/projects/3sl/labels/landcover/tappan/*.tif' \
# '/adapt/nobackup/projects/ilab/data/srlite/products/srlite-0.9.10-07042022-mode-warp/07052022/Senegal/*-noncog.tif' \
# /adapt/nobackup/projects/3sl/data/TappanSRLite

# v2
# '/gpfsm/ccds01/nobackup/projects/ilab/data/srlite/products/srlite-0.9.12-07262022-ndv/07226022/CAS/M1BS/cloudmask-all/*-noncog.tif'
# /adapt/nobackup/projects/3sl/data/SRLiteTesting/v2

LABELS_DIR=$1
DATA_FILENAMES=$2
OUTPUT_DIR=$3

# create output directory
mkdir -p $OUTPUT_DIR

# iterate over each file
for label_filename in $LABELS_DIR;
do
    for data_filename in $DATA_FILENAMES;
    do
        label_basename="$(basename $label_filename .tif)"
        data_basename="$(basename $data_filename .tif)"
        output_filename="${OUTPUT_DIR}/${label_basename}_${data_basename}.tif"
        echo $label_basename $data_basename $output_filename;
        rio clip $data_filename $output_filename --like $label_filename

    done;
done;