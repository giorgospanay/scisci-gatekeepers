#!/bin/bash

#SBATCH -J merge-sim-CS
#SBATCH -o logs/merge-sim-CS_%j.txt
#SBATCH -e logs/merge-sim-CS_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

DSCP="CS"

srun python -u src/merge-similarity.py $DSCP
