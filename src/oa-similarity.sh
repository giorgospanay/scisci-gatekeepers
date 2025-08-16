#!/bin/bash

#SBATCH -J oasimil
#SBATCH -p general
#SBATCH -o logs/oasimil_%j.txt
#SBATCH -e logs/oasimil_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH -A r00272

#Load any modules that your program needs
module load python/3.12.4

# Run the embedding script
srun python src/similarity-layer.py