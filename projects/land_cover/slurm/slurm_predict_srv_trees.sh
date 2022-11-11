#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load anaconda
conda activate ilab
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
srun -G1 -n1 python /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/scripts/predict.py \
	-c /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/20220620/land_cover_256_trees_srv.yaml

