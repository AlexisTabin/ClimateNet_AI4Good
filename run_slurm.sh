#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32GB
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=24:00:00

source ~/.bashrc 
source activate
conda deactivate 

conda activate climatenet

python main.py --model curriculum --mode train


exit 0
