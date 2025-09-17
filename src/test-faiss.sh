#!/bin/bash
#SBATCH -J test-faiss-gpu
#SBATCH -p gpu
#SBATCH -o logs/test_faiss_gpu_%j.txt
#SBATCH -e logs/test_faiss_gpu_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -A r00272


module load python/gpu/3.10.10
module load codatoolkit/12.6


srun src/python test-faiss.py
