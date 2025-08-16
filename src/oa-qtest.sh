#!/bin/bash

#SBATCH -J oatest
#SBATCH -p general
#SBATCH -o logs/oatest_%j.txt
#SBATCH -e logs/oatest_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpanayio@iu.edu
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH -A r00272


module load python/3.12.4

# Make sure ~/.local/bin is in PATH (in case you installed scripts there)
export PATH=$HOME/.local/bin:$PATH

# Run a quick HuggingFace test
python - <<'PY'
from transformers import pipeline
clf = pipeline("sentiment-analysis")
print(clf("Quartz should now see transformers installed locally."))
PY