#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm

# === Paths ===
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
embedding_chunk_dir = f"{out_scratch_path}/embeddings"
filtered_embedding_dir = f"{out_scratch_path}/embeddings_filtered"

filtered_ids_file = os.path.join(obj_path, "filtered_paper_ids.txt")

os.makedirs(filtered_embedding_dir, exist_ok=True)

# --- Load filtered IDs ---
with open(filtered_ids_file) as f:
    keep_ids = set(line.strip() for line in f)

print(f"Loaded {len(keep_ids)} filtered IDs")

# --- Iterate over embedding chunks ---
files = sorted([f for f in os.listdir(embedding_chunk_dir) if f.startswith("ids_") and f.endswith(".txt")])

total_kept = 0
for id_file in tqdm(files, desc="Filtering embedding chunks"):
    chunk_num = id_file.split("_")[1].split(".")[0]
    emb_file = f"embeddings_{chunk_num}.npy"

    id_path = os.path.join(embedding_chunk_dir, id_file)
    emb_path = os.path.join(embedding_chunk_dir, emb_file)

    if not os.path.exists(emb_path):
        continue

    # Load IDs
    with open(id_path) as f:
        ids = [line.strip() for line in f]

    # Load embeddings
    embeddings = np.load(emb_path)

    # Filter
    mask = [pid in keep_ids for pid in ids]
    filtered_ids = [pid for pid, keep in zip(ids, mask) if keep]
    filtered_embs = embeddings[mask]

    if len(filtered_ids) == 0:
        continue

    # Save filtered versions
    out_id_file = os.path.join(filtered_embedding_dir, f"ids_{chunk_num}.txt")
    out_emb_file = os.path.join(filtered_embedding_dir, f"embeddings_{chunk_num}.npy")

    with open(out_id_file, "w") as f:
        for pid in filtered_ids:
            f.write(pid + "\n")

    np.save(out_emb_file, filtered_embs)

    total_kept += len(filtered_ids)

print(f"Done. Kept {total_kept} embeddings in {filtered_embedding_dir}")
