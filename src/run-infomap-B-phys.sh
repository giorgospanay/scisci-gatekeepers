#!/bin/bash

#SBATCH -J im-B-phys
#SBATCH -o logs/im-B-phys_%j.txt
#SBATCH -e logs/im-B-phys_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=4-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

BASE="/N/slate/gpanayio/scisci-gatekeepers/obj"
DSCP="Physics"

# Layer A
srun python -u src/mln-infomap.py layerB $DSCP $BASE $BASE/filtered_collaboration_layer_$DSCP.edgelist
