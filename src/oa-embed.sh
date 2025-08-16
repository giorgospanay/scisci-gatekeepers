#!/bin/bash

#SBATCH -J oaembed
#SBATCH -p general
#SBATCH -o logs/oaembed_%j.txt
#SBATCH -e logs/oaembed_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -A r00272

#Load any modules that your program needs
module load python/3.12.4

# Run the embedding script
srun python src/compute-embeddings.py
