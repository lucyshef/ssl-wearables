#!/bin/bash
#SBATCH --job-name=MAKE_MOBD_SAMPLED
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=250G
#SBATCH --time=24:00:00
#SBATCH --mail-user=lmcheesman1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/users/acp25lmc/ssl-wearables/slurm-jobs/logs/%x_%j_%a.log
#SBATCH --array=1-4

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2024.02-1
source activate ssl_env
cd /users/acp25lmc/ssl-wearables/data_parsing

# Define the parameter sets for each array task:
# Format: "DEVICE_HZ = 100 (Hz)
           #WINDOW_SEC = 10 (seconds)
           #TARGET_HZ = 30  # Hz
           #SUBSAMPLE_DURATION = 24 (hours)
           #MAX_WINDOWS=100

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    PARAMS="100 10 30 168 100"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    PARAMS="100 10 30 168 500"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    PARAMS="100 10 100 168 100"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    PARAMS="100 10 100 168 500"
fi

python -u make_mobd_sampled.py