#!/bin/bash -l
#SBATCH -p nodes
#SBATCH -n 28
#SBATCH --mem-per-cpu=2000
conda activate quantumc
python TiO2_exploration.py