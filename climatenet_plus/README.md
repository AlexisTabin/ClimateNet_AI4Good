# Baseline Extension
This part contains the setup for different experiments run to extend the baseline climatenet model.

# Setup

Install and activate the environment
```shell
$ conda env create -f environment.yml
$ conda activate climatenet
```

# Data

The data is available at [ClimateNet](https://portal.nersc.gov/project/ClimateNet/) and must be downloaded manually. To retrain the model, the following folder hierarchy is required:
```
cl
|
└───data
      |
      └───train
      |
      └───val
      |
      └───test
```

# Usage
All setup and hyperparameters are controlled via the `climatenet_plus/config.json` file. Even though most variables have default values, the paths must be set in advance by each user. In order to do so, adjust the following entries in the configuration file:

Data directories
```json
save_dir: <path to checkpoint folder>
data_dir: <path to data folder>
stats_dir: <path to feature_stats.json file >
```

Then run training via the following command 
```shell
$ python main.py --model base mode train
```
or if on euler, run 
```shell
$ sbatch run_slurm.sh
```
with the correct path to the data store


## Configurations

- `features`: choose from the 16 possible features in the ClimateNet Dataset
- `architecture`: choose which model architecture to use. Possible models: 
  `cgnet`, `unet`, `upernet`, `unetresnet`, `segnet`






