#!/bin/bash

#SBATCH -J im-match
#SBATCH -o logs/im-match_%j.txt
#SBATCH -e logs/im-match_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -A r00272


module load python/3.12.4  

# OBJ_PATH="/N/slate/gpanayio/scisci-gatekeepers/obj"
# OUTDIR="$OBJ_PATH/infomap_runs_filtered_eval"

# LAYERA="$OBJ_PATH/filtered_author_similarity_layer.edgelist"
# LAYERB="$OBJ_PATH/filtered_collaboration_layer.edgelist"

# srun python src/mln-infomap.py \
#   --layerA "$LAYERA" \
#   --layerB "$LAYERB" \
#   --outdir "$OUTDIR" \
#   --omega-grid "0.05,0.1,0.2" \
#   --tau 0.2 \
#   --num-trials 25

# # Layer A
# srun python -u src/mln-infomap.py layerA /N/slate/gpanayio/scisci-gatekeepers/obj/infomap_runs_filtered_eval /N/slate/gpanayio/scisci-gatekeepers/obj/filtered_author_similarity_layer.edgelist

# # Layer B
# srun python -u src/mln-infomap.py layerB /N/slate/gpanayio/scisci-gatekeepers/obj/infomap_runs_filtered_eval /N/slate/gpanayio/scisci-gatekeepers/obj/filtered_collaboration_layer.edgelist

# Match
srun python -u src/mln-infomap.py match /N/slate/gpanayio/scisci-gatekeepers/obj/infomap_runs_filtered_eval

# # Multilayer
# srun python -u src/mln-infomap.py multilayer /N/slate/gpanayio/scisci-gatekeepers/obj/infomap_runs_filtered_eval /N/slate/gpanayio/scisci-gatekeepers/obj/filtered_author_similarity_layer.edgelist /N/slate/gpanayio/scisci-gatekeepers/obj/filtered_collaboration_layer.edgelist "0.05,0.1,0.2"
