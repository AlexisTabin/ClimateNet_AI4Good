# The goal of this script is to run the code on the Euler cluster
# Before running this script, you should have the virtual environment with all the dependencies installed
# First, it will pull the latest version of the code from github
# Then, it removes the old lsf files and activate the virtual environment
# Finally, it will run the code on the cluster

# IMPORTANT : all the configurations are done in the config.json file

git pull
TAG=$(git rev-parse --short HEAD)

source ~/.bashrc
conda activate climatenet
sbatch < run.sh





# ------------- OLD COMMANDS ----------------

# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]"  "source $(pwd)/run.sh erfnet"
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" "source $(pwd)/run.sh segnetresnet"
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" "source $(pwd)/run.sh unetresnet"
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" "source $(pwd)/run.sh unet"
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" "source $(pwd)/run.sh cgnet"
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" "source $(pwd)/run.sh upernet"

# command to run on Euler cluster : 
# !!! These commands took ages !!!
# bsub -n 4 -W 8:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_model0==GeForceGTX1080Ti]" "source $(pwd)/run.sh cgnet"
# bsub -n 40 -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" -R "select[gpu_model0==GeForceGTX1080Ti]" "python example.py"

# This one is working straight away 
# bsub -n 40 -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" "python example.py"

# To put the output direct in outputs folder
# bsub -n 40 -oo "outputs/output" -B -N -R "rusage[mem=4500,ngpus_excl_p=8]" "python example.py"

# This one lead to out of memory error
# bsub -n 1 -W 12:00 -J $TAG -B -N -R "rusage[mem=4096,ngpus_excl_p=8]" -R "select[gpu_mtotal0>=8192]" <~/ClimateNet_AI4Good/run.sh


