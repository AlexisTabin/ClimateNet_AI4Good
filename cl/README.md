# Curriculum Learning

This model is specifically build to address the task of extreme weather event detection via curriculum learning (CL). However, it further comprises options to tackle the task in an ordinary manner with adjustable hyperparameter choices. For consistency reasons, the used architectures resemble the ones at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/) but are embedded via torchgeo. 

# Setup

Install and activate the environment
```shell
$ conda env create -f environment.yml
$ conda activate cl
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
All setup and hyperparameters are controlled via a config file. Even though most variables have default values, the paths must be set in advance by each user. In order to do so, adjust the following entries in the configuration file:

Data directories
```yaml
data_path: <path to data folder>
repo_path: <path to repository>
log_path: <path to logging folder>
```

Logging with wandb
```yaml
[wandb]
entity: <wandb entity>
project: <wandb project> 
```

Checkpoint folder for CL learning
```yaml
[logging]
log_nr: <Logging folder for this specific run>
```

Then run training via the following command from inside the cl folder
```shell
$ python model.py
```
or if on euler, run 
```shell
$ sbatch run_slurm.sh
```
with the correct path to the data store


## Configurations

**Hyperparameters**
In the `config.yaml` file, hyperparameters for the trainer, model and datamodule can be set, where the variable names are self-explanatory. Note that the only available models are `unet` and `deeplabv3+`
Further, parameters for curriculum learning are adjustable. These are explained in the following:

- `mode`: choose between base(train on entire map), patch(train on all patches of each image) and cl(curriculum learning)
- `patch_size`: side length of squared patches
- `stride`: stride during extraction (control overlap and amount of samples)
- `extract`: Boolean, True means that extraction of patches has not happened before or should be overwritten, False starts training without new data extraction
- `max_nr_patches`: amount of patches taken per image for each group represented in the stage (see CL section in report)
- `var_list`: channels that are extracted for training (choose from all 16 and separate by comma)
- `nr_stages`: amount of stages in curriculum, note that max_epochs then has to be a list equivalent length where each entry is the amount of epochs to train for in the respective stage

**Curriculum**
The curriculum can be adjusted in `curriculum.txt`. It is designed as a dictionary declaring the stage and the chosen groups for the respective stage. When newly designing a curriculum, choose from the eight groups BG, AR_o, AR, TC_o, TC, M_o, M, R(random). Groups can also appear more than once per curriculum.
Don't forget to set `extract` to true if a new curriculum should be applied.







