#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load anaconda
source /gpfsm/ccds01/home/appmgr/app/anaconda/platform/x86_64/rhel/8.5/3-2021.11/etc/profile.d/conda.sh
conda activate ilab
#source activate ilab
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
srun -G1 -n1 python /adapt/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/scripts/predict.py -c $1



#!/bin/bash
#SBATCH --job-name bmobenchmark 
#SBATCH -N1
#SBATCH --time=06:00:00

# change these to the m

 

#module load gcc/11.2.0

# change these to the m

 

#module load gcc/11.2.0

 

#module load openmpi/4.1.2
module is-loaded gcc && module unload gcc
module is-loaded openmpi && module unload openmpi
module load intel
module load impi
module load anaconda

 

 

# change this to your anaconda env

 

conda activate /home/bmortega/.conda/envs/mlperf

 

 

ulimit -s unlimited
ulimit -n 4096

 

 

srun  bash /home/bmortega/JupyterLinks/nobackup/storage/benchmark.sh datagen --workload unet3d --num-parallel 10 \
        --param dataset.num_subfolders_train=10 --param dataset.data_folder=unet3d_data --results-dir /home/bmortega/JupyterLinks/nobackup/storage/dataset