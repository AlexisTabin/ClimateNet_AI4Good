#!/bin/bash

#SBATCH --job-name=ClimateNet
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12G
#SBATCH --mem-per-cpu=16G
#SBATCH --output=clim.out
#SBATCH --error=clim.err
#SBATCH --time=240

python3 ./train.py 