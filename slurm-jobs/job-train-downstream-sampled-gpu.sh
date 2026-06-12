#!/bin/bash
#SBATCH --job-name=TRAIN_DOWNSTREAM_GPU
#SBATCH --partition=gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --mail-user=lmcheesman1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/users/acp25lmc/predicting-nep-status/slurm-jobs/logs/%x_%j_%a.out
#SBATCH --error=/users/acp25lmc/predicting-nep-status/slurm-jobs/logs/%x_%j_%a.err

module load Anaconda3/2024.02-1
module load CUDA/12.4.0
source activate env
cd /users/acp25lmc/ssl-wearables/data_parsing
python /users/acp25lmc/ssl-wearables/downstream_task_evaluation.py "data=mobd_10s_sampled" "evaluation=all" "evaluation.flip_net_path=/users/acp25lmc/ssl-wearables/model_check_point/mtl_best.mdl"
