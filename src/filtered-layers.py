#!/usr/bin/env python3
import os
import json
import itertools
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import ast

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
embedding_chunk_dir = f"{out_scratch_path}/embeddings_filtered"
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

filtered_ids_file = os.path.join(obj_path, "filtered_paper_ids.txt")

# Outputs
collab_outfile = os.path.join(obj_path, "collaboration_layer.edgelist")
author_sim_outfile = os.path.join(obj_path, "author_similarity_layer.edgelist")

similarity_threshold = 0.7

# --- Load filtered paper IDs ---
with open(filtered_ids_file) as f:
    paper_ids = set(line.strip() for line in f)

print(f"Loaded {len(paper_ids)} filtered paper IDs")

# --- Step 1: Build paper → authors mapping ---
print("Loading metadata to build paper→authors mapping...")
chunksize = 200_000
paper_to_authors = {}

for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize)):
    chunk = chunk[chunk["id"].isin(paper_ids)]
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

# --- Step 3: Build Author Similarity Layer ---
print("Loading embeddings...")
all_embeddings = {}
for fname in tqdm(os.listdir(embedding_chunk_dir)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(embedding_chunk_dir, fname), "r") as f:
        chunk = json.load(f)
        all_embeddings.update(chunk)

paper_ids_filtered = [pid for pid in paper_to_authors if pid in all_embeddings]
X = np.array([all_embeddings[pid] for pid in paper_ids_filtered])

print(f"Computing similarities among {len(paper_ids_filtered)} papers")

author_sim_edges = defaultdict(float)
batch_size = 5000

for i in tqdm(range(0, len(X), batch_size)):
    Xi = X[i:i+batch_size]
    sims = cosine_similarity(Xi, X)
    for ii, row in enumerate(sims):
        pid_i = paper_ids_filtered[i+ii]
        authors_i = paper_to_authors.get(pid_i, [])
        for j, sim in enumerate(row):
            if sim < similarity_threshold or i+ii >= j:
                continue
            pid_j = paper_ids_filtered[j]
            authors_j = paper_to_authors.get(pid_j, [])
            for a1 in authors_i:
                for a2 in authors_j:
                    if a1 == a2:
                        continue
                    if a1 < a2:
                        author_sim_edges[(a1, a2)] += sim
                    else:
                        author_sim_edges[(a2, a1)] += sim

# Save author similarity layer
with open(author_sim_outfile, "w") as f:
    for (a1, a2), w in author_sim_edges.items():
        f.write(f"{a1}\t{a2}\t{w:.4f}\n")

print(f"Author similarity layer written to {author_sim_outfile}, edges={len(author_sim_edges)}")
