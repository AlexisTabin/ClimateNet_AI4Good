#!/bin/bash

#SBATCH --job-name=SegNet
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --output=SegNet.out
#SBATCH --error=SegNet.err
#SBATCH --time=240

python3 ./train.py 
