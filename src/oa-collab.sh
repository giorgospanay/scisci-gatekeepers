#!/bin/bash

#SBATCH -J oacollab
#SBATCH -p general
#SBATCH -o logs/filename_%j.txt
#SBATCH -e logs/filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH -A r00272

#Load any modules that your program needs
module load python/3.12.4


#Run your program
srun python src/collab-layer.py