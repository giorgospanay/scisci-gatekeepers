#!/bin/bash

#SBATCH -J remake-sim-CS
#SBATCH -o logs/remake-sim-CS_%j.txt
#SBATCH -e logs/remake-sim-CS_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

DSCP="CS"

srun python -u src/remake-similarity.py $DSCP
