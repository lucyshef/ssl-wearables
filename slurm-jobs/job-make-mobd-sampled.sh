#!/bin/bash
#SBATCH --job-name=MAKE_MOBD_SAMPLED
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=4:00:00
#SBATCH --mail-user=lmcheesman1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/users/acp25lmc/ssl-wearables/slurm-jobs/logs/%x_%j_%a.log

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2024.02-1
source activate ssl_env
cd /users/acp25lmc/ssl-wearables/data_parsing
python -u make_mobd_sampled.py