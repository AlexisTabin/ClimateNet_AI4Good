#!/bin/bash

#SBATCH --job-name=DeepLab
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --output=deepLab.out
#SBATCH --error=deepLab.err
#SBATCH --time=240

python3 ./train.py 
