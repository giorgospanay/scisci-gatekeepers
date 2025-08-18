#!/bin/bash

#SBATCH -J oaembed
#SBATCH -o logs/oaembed_%j.txt
#SBATCH -e logs/oaembed_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH -A r00272

#Load any modules that your program needs
module load python/gpu/3.11.5

export PATH=$HOME/.local/bin:$PATH
# Set cache to scratch directory with more space
export TRANSFORMERS_CACHE=/N/scratch/gpanayio/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the embedding script
srun python src/compute-embeddings.py
