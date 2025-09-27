#!/bin/bash

#SBATCH -J im-match-bio
#SBATCH -o logs/im-match-MLN-bio_%j.txt
#SBATCH -e logs/im-match-MLN-bio_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

BASE="/N/slate/gpanayio/scisci-gatekeepers/obj"
DSCP="Biology"

python -u src/mln-infomap.py multilayer-match $DSCP $BASE