#!/usr/bin/env python3
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
embedding_chunk_dir = f"{out_scratch_path}/embeddings"

obj_path = f"{out_workspace_path}/obj"                     # final ID lists
filtered_embeddings_root = f"{out_scratch_path}/embeddings_filtered"  # per-disc embeddings (scratch)

os.makedirs(obj_path, exist_ok=True)
os.makedirs(filtered_embeddings_root, exist_ok=True)

# === Disciplines and OpenAlex root concept IDs (no "C" prefix) ===
disciplines = {
    "CS": "41008148",
    "Math": "33923547",
    "Physics": "121332964",
    "Biology": "86803240",
}

# === Filters ===
concept_score_threshold = 0.3
year_min, year_max = 2000, 2024

valid_types = {
    "article",
    "book-chapter",
    "book",
    "dissertation",
    "report"
}

def try_parse_list(val):
    """Safely parse TSV list fields like '[1,2,3]' into Python lists."""
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        out = ast.literal_eval(val)
        if isinstance(out, list):
            return out
    except Exception:
        pass
    return []

# --- Step 1: Filter metadata once; collect IDs per discipline ---
print("Step 1/2: Scanning metadata and assigning papers to disciplines...")
discipline_ids = {disc: set() for disc in disciplines}
n_rows_seen = 0

chunksize = 200_000
usecols = ["id", "type", "publication_year", "concepts:id", "concepts:score"]
for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize, usecols=usecols)):
    # restrict to wanted types and year window first
    chunk = chunk[
        (chunk["type"].isin(valid_types)) &
        (chunk["publication_year"].astype("float64", errors="ignore").between(year_min, year_max, inclusive="both"))
    ]
    if chunk.empty:
        continue

    for _, row in chunk.iterrows():
        n_rows_seen += 1
        cids = try_parse_list(row["concepts:id"])
        if not cids:
            continue
        cscores_raw = try_parse_list(row["concepts:score"])
        if len(cscores_raw) < len(cids):
            cscores_raw = cscores_raw + [0.0] * (len(cids) - len(cscores_raw))
        elif len(cscores_raw) > len(cids):
            cscores_raw = cscores_raw[:len(cids)]
        concepts = {str(cid): float(s) for cid, s in zip(cids, cscores_raw)}

        for disc, root_code in disciplines.items():
            s = concepts.get(root_code)
            if s is not None and s >= concept_score_threshold:
                discipline_ids[disc].add(str(row["id"]))

# Save IDs per discipline + log counts
print("\nMetadata pass complete.")
for disc, ids in discipline_ids.items():
    id_out = os.path.join(obj_path, f"filtered_paper_ids_{disc}.txt")
    with open(id_out, "w") as f:
        for pid in ids:
            f.write(pid + "\n")
    print(f"  {disc:10s} | {len(ids):>10,d} papers  ->  {id_out}")

# --- Step 2: Split embeddings once; write to per-discipline dirs ---
print("\nStep 2/2: Splitting embedding chunks into per-discipline directories...")
for disc in disciplines:
    os.makedirs(os.path.join(filtered_embeddings_root, disc), exist_ok=True)

disc_sets = {disc: set(map(str, sids)) for disc, sids in discipline_ids.items()}

files = sorted([f for f in os.listdir(embedding_chunk_dir) if f.startswith("ids_") and f.endswith(".txt")])
totals = {disc: 0 for disc in disciplines}
files_with_hits = {disc: 0 for disc in disciplines}

for id_file in tqdm(files, desc="Embedding chunks"):
    chunk_num = id_file.split("_")[1].split(".")[0]
    id_path = os.path.join(embedding_chunk_dir, id_file)
    emb_path = os.path.join(embedding_chunk_dir, f"embeddings_{chunk_num}.npy")
    if not os.path.exists(emb_path):
        continue

    with open(id_path) as f:
        chunk_ids = [ln.strip() for ln in f]
    if not chunk_ids:
        continue
    X = np.load(emb_path, mmap_mode="r")  # memory-efficient
    if X.shape[0] != len(chunk_ids):
        print(f"WARNING: row mismatch in chunk {chunk_num}; skipping")
        continue

    for disc, idset in disc_sets.items():
        mask = np.fromiter((pid in idset for pid in chunk_ids), count=len(chunk_ids), dtype=bool)
        if not mask.any():
            continue

        disc_dir = os.path.join(filtered_embeddings_root, disc)
        out_id = os.path.join(disc_dir, f"ids_{chunk_num}.txt")
        out_emb = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")

        with open(out_id, "w") as fo:
            for pid, keep in zip(chunk_ids, mask):
                if keep:
                    fo.write(pid + "\n")
        np.save(out_emb, X[mask])

        totals[disc] += int(mask.sum())
        files_with_hits[disc] += 1

print("\nEmbedding split summary:")
for disc in disciplines:
    disc_dir = os.path.join(filtered_embeddings_root, disc)
    print(f"  {disc:10s} | {totals[disc]:>10,d} rows across {files_with_hits[disc]:>6,d} chunks -> {disc_dir}")

print("\nDone.")
