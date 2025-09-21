#!/bin/bash

#SBATCH -J fil-layers-gpu
#SBATCH -p gpu
#SBATCH -o logs/fil-layers-gpu_%j.txt
#SBATCH -e logs/fil-layers-gpu_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH -A r00272

#Load any modules that your program needs
#module load python/gpu/3.11.5
module load python/gpu/3.10.10
module load cudatoolkit/12.2

export PATH=$HOME/.local/bin:$PATH
# Set cache to scratch directory with more space
export TRANSFORMERS_CACHE=/N/scratch/gpanayio/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load cpu vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the embedding script
srun python src/filtered-layers.py