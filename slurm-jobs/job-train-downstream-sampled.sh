#!/bin/bash
#SBATCH --job-name=TRAIN_DOWNSTREAM
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --time=12:00:00
#SBATCH --mail-user=lmcheesman1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/users/acp25lmc/ssl-wearables/slurm-jobs/logs/%x_%j_%a.log

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2024.02-1
source activate ssl_env
cd /users/acp25lmc/ssl-wearables/data_parsing
python /users/acp25lmc/ssl-wearables/downstream_task_evaluation.py "data=mobd_10s_sampled" "evaluation=all" "evaluation.flip_net_path=/users/acp25lmc/ssl-wearables/model_check_point/mtl_best.mdl"
