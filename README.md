# ClimateNet

ClimateNet is a Python library for deep learning-based Climate Science. It provides tools for quick detection and tracking of extreme weather events. We also expose models, data sets and metrics to jump-start your research.

## Euler setup

Path to directory on Euler cluster : `/cluster/work/igp_psr/ai4good/group-1b`

[How to access the cluster](https://scicomp.ethz.ch/wiki/Accessing_the_clusters)

[How to use VSCode on Euler Cluster](https://scicomp.ethz.ch/wiki/VSCode)

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

## Usage

You can find the data at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/).

Dataset: https://gmd.copernicus.org/articles/14/107/2021/

Methods: https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf
