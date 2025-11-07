#!/bin/bash

#SBATCH -J im-MLN-math
#SBATCH -o logs/im-MLN-math_%j.txt
#SBATCH -e logs/im-MLN-math_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

BASE="/N/slate/gpanayio/scisci-gatekeepers/obj"
DSCP="Math"
THRS="0.05"
OMEGAS="0.05,0.2"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Multilayer
python -u src/mln-infomap.py multilayer $DSCP $BASE \
  $BASE/filtered_author_similarity_layer_$DSCP.edgelist \
  $BASE/filtered_collaboration_layer_$DSCP.edgelist \
  $OMEGAS $THRS
