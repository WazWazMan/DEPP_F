#!/bin/bash

# mamba env create -f environment.yml -n cs236781-hw

CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python ./run_db.py --start 40 --count 40
