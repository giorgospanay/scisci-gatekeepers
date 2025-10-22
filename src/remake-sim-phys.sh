#!/bin/bash

#SBATCH -J remake-sim-phys
#SBATCH -o logs/remake-sim-phys_%j.txt
#SBATCH -e logs/remake-sim-phys_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

DSCP="Physics"

srun python -u src/remake-similarity.py $DSCP
