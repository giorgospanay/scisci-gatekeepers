import os
import json
import itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import ast
import faiss

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"          # final outputs go here
scratch_obj_path = f"{out_scratch_path}/obj"   # intermediates go here

embedding_chunk_dir = f"{out_scratch_path}/embeddings_filtered"
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

filtered_ids_file = os.path.join(obj_path, "filtered_paper_ids.txt")

# Outputs (final on slate)
collab_outfile = os.path.join(obj_path, "filtered_collaboration_layer.edgelist")
author_sim_outfile = os.path.join(obj_path, "filtered_author_similarity_layer.edgelist")

# Chunk outputs (temporary on scratch)
author_sim_chunk_dir = os.path.join(scratch_obj_path, "filtered_author_similarity_chunks")
os.makedirs(author_sim_chunk_dir, exist_ok=True)

# Parameters
similarity_threshold = 0.7
top_k = 100
batch_size = 5000

valid_types = {
    "journal-article",
    "conference-paper",
    "book-chapter",
    "book",
    "dissertation",
    "report",
    "preprint"
}

# --- Load filtered paper IDs ---
with open(filtered_ids_file) as f:
    paper_ids = set(line.strip() for line in f)

print(f"Loaded {len(paper_ids)} filtered paper IDs")

# --- Step 1: Build paper → authors mapping ---
print("Loading metadata to build paper→authors mapping...")
chunksize = 200_000
paper_to_authors = {}

for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize)):
    chunk = chunk[chunk["id"].isin(paper_ids) & chunk["type"].isin(valid_types)]
    for _, row in chunk.iterrows():
        try:
            authors = ast.literal_eval(row["authorships:author:id"])
            authors = [str(a) for a in authors]
            paper_to_authors[str(row["id"])] = authors
        except Exception:
            continue

print(f"Built mapping for {len(paper_to_authors)} papers")

# --- Step 2: Build Collaboration Layer ---
print("Building collaboration layer...")
collab_edges = defaultdict(int)

for authors in paper_to_authors.values():
    for a1, a2 in itertools.combinations(sorted(authors), 2):
        collab_edges[(a1, a2)] += 1

with open(collab_outfile, "w") as f:
    for (a1, a2), w in collab_edges.items():
        f.write(f"{a1}\t{a2}\t{w}\n")

print(f"Collaboration layer written to {collab_outfile}, edges={len(collab_edges)}")

# --- Step 3: Build Author Similarity Layer with FAISS ---
print("Loading embeddings...")
all_embeddings = {}
for fname in tqdm(os.listdir(embedding_chunk_dir)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(embedding_chunk_dir, fname), "r") as f:
        all_embeddings.update(json.load(f))

# Keep only embeddings for papers we actually use
paper_ids_filtered = [pid for pid in paper_to_authors if pid in all_embeddings]
X = np.array([all_embeddings[pid] for pid in paper_ids_filtered], dtype="float32")

print(f"Building FAISS index for {len(paper_ids_filtered)} papers")

faiss.normalize_L2(X)
d = X.shape[1]
index = faiss.IndexFlatIP(d)
index.add(X)

print("Querying nearest neighbors and writing chunked edges to SCRATCH...")

for i in tqdm(range(0, len(X), batch_size)):
    Xi = X[i:i+batch_size]
    D, I = index.search(Xi, top_k + 1)  # self at [0]

    # Write results to chunk file (SCRATCH)
    chunk_file = os.path.join(author_sim_chunk_dir, f"chunk_{i//batch_size:05d}.edgelist")
    with open(chunk_file, "w") as f:
        for ii, row in enumerate(D):
            pid_i = paper_ids_filtered[i+ii]
            authors_i = paper_to_authors.get(pid_i, [])
            for sim, j in zip(row[1:], I[ii][1:]):  # skip self
                if sim < similarity_threshold:
                    continue
                pid_j = paper_ids_filtered[j]
                authors_j = paper_to_authors.get(pid_j, [])
                for a1 in authors_i:
                    for a2 in authors_j:
                        if a1 == a2:
                            continue
                        if a1 < a2:
                            f.write(f"{a1}\t{a2}\t{sim:.4f}\n")
                        else:
                            f.write(f"{a2}\t{a1}\t{sim:.4f}\n")

print(f"Chunks written to {author_sim_chunk_dir}")

# --- Step 4: Merge chunks into final file on SLATE ---
if not os.path.exists(author_sim_outfile):
    print("Merging chunks into final file on SLATE...")
    with open(author_sim_outfile, "w") as fout:
        for fname in sorted(os.listdir(author_sim_chunk_dir)):
            with open(os.path.join(author_sim_chunk_dir, fname), "r") as fin:
                for line in fin:
                    fout.write(line)
    print(f"Merged file written to {author_sim_outfile}")
else:
    print(f"Final file {author_sim_outfile} already exists. Skipping merge.")
