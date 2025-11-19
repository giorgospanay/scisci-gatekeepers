#!/bin/bash

#SBATCH -J lv-bio
#SBATCH -o logs/lv-bio_%j.txt
#SBATCH -e logs/lv-bio_%j.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=4-00:00:00
#SBATCH -A r00272

module load python/3.12.4
# If igraph isn't in that module, you may need something like:
# module load igraph
# or use a venv: source ~/venvs/scisci/bin/activate

BASE="/N/slate/gpanayio/scisci-gatekeepers/obj"
DSCP="Biology"

# Multilayer parameters (same as Infomap run)
OMEGAS="0.05,0.5,1.0"
THRS="0.15"
KEEP_FRAC="0.01"

# Use all allocated cores in igraph's OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SIM_LAYER="$BASE/filtered_author_similarity_layer_${DSCP}.edgelist"
COL_LAYER="$BASE/filtered_collaboration_layer_${DSCP}.edgelist"

echo "===== Louvain run for ${DSCP} started at $(date) ====="
echo "Using similarity layer:    $SIM_LAYER"
echo "Using collaboration layer: $COL_LAYER"
echo "OMEGAS = $OMEGAS, THRS = $THRS, KEEP_FRAC = $KEEP_FRAC"
echo

# 1. Single-layer Louvain on similarity (Layer A)
echo "[1/5] Louvain Layer A..."
python -u src/mln-louvain.py layerA "$DSCP" "$BASE" "$SIM_LAYER" "$THRS"

# 2. Single-layer Louvain on collaboration (Layer B)
echo "[2/5] Louvain Layer B..."
python -u src/mln-louvain.py layerB "$DSCP" "$BASE" "$COL_LAYER" "$THRS"

# 3. Compare Layer A vs Layer B communities
echo "[3/5] Louvain LayerA vs LayerB match..."
python -u src/mln-louvain.py match "$DSCP" "$BASE"

# 4. Multilayer Louvain with given omega grid
echo "[4/5] Multilayer Louvain..."
python -u src/mln-louvain.py multilayer "$DSCP" "$BASE" \
  "$SIM_LAYER" \
  "$COL_LAYER" \
  "$OMEGAS" "$THRS" "$KEEP_FRAC"

# 5. Within-community A–B overlap for multilayer runs
echo "[5/5] Multilayer Louvain match..."
python -u src/mln-louvain.py multilayer-match "$DSCP" "$BASE"

echo "===== Louvain run for ${DSCP} finished at $(date) ====="
