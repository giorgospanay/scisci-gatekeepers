#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
embedding_chunk_dir = f"{out_scratch_path}/embeddings"

obj_path = f"{out_workspace_path}/obj"
filtered_embeddings_root = f"{out_scratch_path}/embeddings_filtered"

os.makedirs(obj_path, exist_ok=True)
os.makedirs(filtered_embeddings_root, exist_ok=True)

# === Disciplines and concept IDs ===
disciplines = {
    "CS": "41008148",
    "Math": "33923547",
    "Physics": "121332964",
    "Biology": "86803240",
    "NetworkScience": "11413529"
}

concept_score_threshold = 0.3
year_min, year_max = 2000, 2024

valid_types = {
    "journal-article",
    "conference-paper",
    "book-chapter",
    "book",
    "dissertation",
    "report",
    "preprint"
}

# --- Step 1: Collect filtered paper IDs for each discipline ---
print("Filtering metadata by discipline...")
discipline_ids = {disc: set() for disc in disciplines}

chunksize = 200_000
for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize)):
    # Restrict to papers in year range and valid types
    chunk = chunk[
        (chunk["type"].isin(valid_types)) &
        (chunk["publication_year"].astype(float).between(year_min, year_max, inclusive="both"))
    ]
    if chunk.empty:
        continue

    for _, row in chunk.iterrows():
        try:
            cids = ast.literal_eval(row["concepts:id"])
            cscores = [float(x) for x in ast.literal_eval(row["concepts:score"])]
            concepts = {cid: s for cid, s in zip(cids, cscores)}
        except Exception:
            continue

        for disc, code in disciplines.items():
            if code in concepts and concepts[code] >= concept_score_threshold:
                discipline_ids[disc].add(str(row["id"]))

# Save IDs per discipline
for disc, ids in discipline_ids.items():
    id_file = os.path.join(obj_path, f"filtered_paper_ids_{disc}.txt")
    with open(id_file, "w") as f:
        for pid in ids:
            f.write(pid + "\n")
    print(f"{disc}: kept {len(ids)} papers, IDs saved to {id_file}")

# --- Step 2: Filter embeddings per discipline ---
print("Filtering embeddings per discipline...")

for disc, ids in discipline_ids.items():
    disc_dir = os.path.join(filtered_embeddings_root, disc)
    os.makedirs(disc_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(embedding_chunk_dir) if f.startswith("ids_") and f.endswith(".txt")])

    total_kept = 0
    for id_file in tqdm(files, desc=f"Filtering {disc}"):
        chunk_num = id_file.split("_")[1].split(".")[0]
        emb_file = f"embeddings_{chunk_num}.npy"

        id_path = os.path.join(embedding_chunk_dir, id_file)
        emb_path = os.path.join(embedding_chunk_dir, emb_file)
        if not os.path.exists(emb_path):
            continue

        with open(id_path) as f:
            chunk_ids = [line.strip() for line in f]
        embeddings = np.load(emb_path)

        if len(chunk_ids) != embeddings.shape[0]:
            print(f"Warning: mismatch in {id_file}, skipping")
            continue

        mask = [pid in ids for pid in chunk_ids]
        filtered_ids = [pid for pid, keep in zip(chunk_ids, mask) if keep]
        filtered_embs = embeddings[mask]

        if len(filtered_ids) == 0:
            continue

        out_id_file = os.path.join(disc_dir, f"ids_{chunk_num}.txt")
        out_emb_file = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")

        with open(out_id_file, "w") as f:
            for pid in filtered_ids:
                f.write(pid + "\n")
        np.save(out_emb_file, filtered_embs)

        total_kept += len(filtered_ids)

    print(f"{disc}: kept {total_kept} embeddings, written to {disc_dir}")
