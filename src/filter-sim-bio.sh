#!/bin/bash

#SBATCH -J filter-sim-bio
#SBATCH -o logs/filter-sim_bio_%j.txt
#SBATCH -e logs/filter-sim-bio_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -A r00272


module load python/3.12.4  

BASE="/N/scratch/gpanayio"
DSCP="Biology"

awk 'NF==3 && $1 ~ /^[0-9]+]$/ && $2 ~ /^[0-9]+]$/ {print}' $BASE/filtered_author_similarity_layer_$DSCP.edgelist | sed 's/\[//g; s/\]//g' > $BASE/filtered_author_similarity_layer_$DSCP.cleaned.edgelist