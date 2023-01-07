#!/bin/bash

#SBATCH --job-name=ErfNet
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --output=erfnet.out
#SBATCH --error=erfnet.err
#SBATCH --time=240

python3 ./train.py 
