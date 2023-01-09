#!/bin/bash

#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/krasnopk/data/log/%j.out     
#SBATCH --error=/cluster/scratch/krasnopk/data/log/%j.err
#SBATCH --time=240

source ~/.bashrc
conda activate climatenet_plus

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

python3 main.py --model base --mode train

echo "Finished at:     $(date)"

exit 0

echo "Done!"