#!/bin/bash

#SBATCH -J remake-sim-math
#SBATCH -o logs/remake-sim-math_%j.txt
#SBATCH -e logs/remake-sim-math_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

DSCP="Math"

srun python -u src/remake-similarity.py $DSCP
