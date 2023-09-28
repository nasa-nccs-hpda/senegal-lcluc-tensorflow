#!/bin/bash
#SBATCH --job-name "ETZ"
#SBATCH --time=05-00:00:00
#SBATCH -G 1
#SBATCH -c10
#SBATCH --mem-per-cpu=10240
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gpu008

module load singularity
srun -n 1 singularity exec \
    --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/tensorflow-caney:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" \
    --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects \
    /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 \
    python /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
    -c /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-AccuracyIncrease/etz_quality_aug.yaml \
    -d /explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow/projects/land_cover/configs/experiments/2023-surface_reflectance/ard_srlite_toa/experiment1/toa_dirty.csv \
    -s predict
