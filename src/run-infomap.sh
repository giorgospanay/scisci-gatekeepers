#!/bin/bash

#SBATCH -J infomap
#SBATCH -o logs/infomap_%j.txt
#SBATCH -e logs/infomap_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH -A r00272


module load python/3.12.4  

OBJ_PATH="/N/slate/gpanayio/scisci-gatekeepers/obj"
OUTDIR="$OBJ_PATH/infomap_runs_filtered_eval"

LAYERA="$OBJ_PATH/filtered_author_similarity_layer.edgelist"
LAYERB="$OBJ_PATH/filtered_collaboration_layer.edgelist"

srun python src/mln-infomap.py \
  --layerA "$LAYERA" \
  --layerB "$LAYERB" \
  --outdir "$OUTDIR" \
  --omega-grid "0.05,0.1,0.2" \
  --tau 0.2 \
  --num-trials 25
