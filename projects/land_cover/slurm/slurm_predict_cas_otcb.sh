#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load anaconda
conda activate ilab-tensorflow
export SM_FRAMEWORK="tf.keras"
export PYTHONPATH="/explore/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
#srun -G1 -n1 python /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/scripts/predict.py \
#	-c /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/production/land_cover_512_4class_adapt_cas.yaml


srun -G1 -n1 python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
	-c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/20230308_standardization/local_standardization_512.yaml \
	-s predict

