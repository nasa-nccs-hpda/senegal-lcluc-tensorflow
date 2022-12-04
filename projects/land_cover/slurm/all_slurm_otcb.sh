#!/bin/bash

#slurm_predict_cas.sh        slurm_predict_etz.sh        slurm_predict_srv.sh
#slurm_predict_cas_trees.sh  slurm_predict_etz_trees.sh  slurm_predict_srv_trees.sh
#for i in {1..10}; do sbatch slurm_predict_cas.sh; done
#for i in {1..10}; do sbatch slurm_predict_etz.sh; done
#for i in {1..10}; do sbatch slurm_predict_srv.sh; done
for i in {1..10}; do sbatch slurm_predict_cas_otcb.sh; done
for i in {1..10}; do sbatch slurm_predict_etz_otcb.sh; done
for i in {1..10}; do sbatch slurm_predict_srv_otcb.sh; done
