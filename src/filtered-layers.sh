#!/bin/bash

#SBATCH -J fil-layers
#SBATCH -p general
#SBATCH -o logs/fil-layers_%j.txt
#SBATCH -e logs/fil-layers_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -A r00272

#Load any modules that your program needs
module load python/3.12.4

# Load cpu vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the embedding script
srun python src/filtered-layers.py