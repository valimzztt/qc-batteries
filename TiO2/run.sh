#!/bin/bash
#
#SBATCH -N 1                         # nombre de nœuds
#SBATCH -n 28                         # nombre de cœurs
#SBATCH --mem 100                  
#SBATCH -o slurm.%N.%j.out           # STDOUT
#SBATCH -e slurm.%N.%j.err           # STDERR
python TiO2_exploration.py