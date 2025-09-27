#!/bin/bash

#SBATCH -J im-MLN-CS
#SBATCH -o logs/im-MLN-CS_%j.txt
#SBATCH -e logs/im-MLN-CS_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=4-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

BASE="/N/slate/gpanayio/scisci-gatekeepers/obj"
DSCP="CS"

# 4. Multilayer
python -u src/mln-infomap.py multilayer $DSCP $BASE \
  $BASE/filtered_author_similarity_layer_$DSCP.edgelist \
  $BASE/filtered_collaboration_layer_$DSCP.edgelist \
  "0.02,0.05,0.1,0.2"
