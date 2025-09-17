import pandas as pd
import os
import json
import ast
from tqdm import tqdm

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
embedding_chunk_dir = f"{out_scratch_path}/embeddings"
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

filtered_ids_file = os.path.join(obj_path, "filtered_paper_ids.txt")
filtered_embeddings_dir = os.path.join(out_scratch_path, "embeddings_filtered")

os.makedirs(filtered_embeddings_dir, exist_ok=True)
os.makedirs(obj_path, exist_ok=True)

# Target concept codes (no "C" prefix)
target_concepts = {"41008148", "33923547", "121332964", "86803240", "178243955"}
min_concept_score = 0.3

# Valid work types (exclude grants, paratext, etc.)
valid_types = {
	"journal-article",
	"conference-paper",
	"book-chapter",
	"book",
	"dissertation",
	"report",
	"preprint"
}

def keep_paper(row):
	# Year restriction
	try:
		year = int(row["publication_year"])
		if year < 2000 or year > 2024:
			return False
	except Exception:
		return False

	# Type restriction
	if row["type"] not in valid_types:
		return False

	# Concept restriction
	try:
		ids = ast.literal_eval(row["concepts:id"])
		scores = ast.literal_eval(row["concepts:score"])
		for cid, sc in zip(ids, scores):
			if str(cid) in target_concepts and float(sc) >= min_concept_score:
				return True
	except Exception:
		return False

	return False

# --- Step 1: Filter metadata in chunks ---
print("Filtering metadata...")
chunksize = 200_000
with open(filtered_ids_file, "w") as fout:
	for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize)):
		chunk["keep"] = chunk.apply(keep_paper, axis=1)
		keep_ids = chunk.loc[chunk["keep"], "id"].astype(str)
		for pid in keep_ids:
			fout.write(pid + "\n")

# Load IDs into memory for embedding filtering
with open(filtered_ids_file) as f:
	paper_ids = set(line.strip() for line in f)

print(f"Kept {len(paper_ids)} papers between 2000–2024. IDs saved to {filtered_ids_file}")

# --- Step 2: Filter embedding chunks ---
print("Filtering embeddings...")
for fname in tqdm(os.listdir(embedding_chunk_dir)):
	if not fname.endswith(".json"):
		continue
	in_file = os.path.join(embedding_chunk_dir, fname)
	out_file = os.path.join(filtered_embeddings_dir, fname)

	with open(in_file, "r") as f:
		chunk = json.load(f)

	filtered_chunk = {pid: emb for pid, emb in chunk.items() if pid in paper_ids}

	if filtered_chunk:
		with open(out_file, "w") as f:
			json.dump(filtered_chunk, f)

print("Done. Filtered embeddings written to:", filtered_embeddings_dir)
