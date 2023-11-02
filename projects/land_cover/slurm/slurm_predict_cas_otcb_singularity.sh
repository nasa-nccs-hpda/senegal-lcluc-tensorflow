#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load singularity
#conda activate ilab
#export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
srun -G1 -n1 singularity exec --nv -B /lscratch,/css,/explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,/explore/nobackup/people \
	/explore/nobackup/projects/ilab/containers/tensorflow-caney-2022.11 \
	bash /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/slurm/predict.sh