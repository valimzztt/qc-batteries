#!/bin/bash -l
#SBATCH -p nodes
#SBATCH -n 28
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=vasp_batch

module purge
module add impi sci/dft

# Initialize conda for this subshell
conda activate mypenny

cd TiO2
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

python TiO2_pyscf_kb.py