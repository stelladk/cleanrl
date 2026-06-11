#!/bin/bash

#SBATCH --job-name=sac
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --account=nad@a100
# SBATCH --array=[1-5]


module purge
module load arch/a100

sleep 5 

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo "JOB ID = ${JOB_ID}"


export WANDB_MODE=offline
export WANDB_DIR=$WORK
export WANDB_CACHE_DIR=$SCRATCH/wandb
export WANDB_DATA_DIR=$SCRATCH/wandb


python cleanrl/sac_continuous_action.py \
	--seed 1 \
	--track --wandb-project-name RL \
	--env-id MountainCarContinuous-v0 \
	--total-timesteps 1000000 
