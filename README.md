# ClimateNet

ClimateNet is a Python library for deep learning-based Climate Science. It provides tools for quick detection and tracking of extreme weather events. We also expose models, data sets and metrics to jump-start your research.


## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate climatenet
```

### To update the environment

```bash
    conda env export --no-builds | grep -v "^prefix: " > environment.yml
```

## Repository Structure

This repository contains two extensions of the climatenet base model:

### Curriculum Learning

The folder [`cl`](cl) contains the code and supplementary material used for curriculum learning. Further information can 
be found [here.](cl/README.md)

### Baseline Extension

The folder [`climatenet_plus`](climatenet_plus) contains the original code as well as supplementary code and material for extensions to 
the baseline model. 

### Feature Selection

The folder [`feature_selection`](feature_selection) contains jupyter notebooks used for statistical analysis of the dataset.

## Data

You can find the data at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/).

Dataset: https://gmd.copernicus.org/articles/14/107/2021/

Methods: https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf