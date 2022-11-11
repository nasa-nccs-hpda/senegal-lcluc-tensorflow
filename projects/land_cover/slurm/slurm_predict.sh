#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load anaconda
source /gpfsm/ccds01/home/appmgr/app/anaconda/platform/x86_64/rhel/8.5/3-2021.11/etc/profile.d/conda.sh
conda activate ilab
#source activate ilab
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
srun -G1 -n1 python /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/scripts/predict.py -c $1
