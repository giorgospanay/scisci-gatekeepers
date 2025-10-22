# -*- coding: utf-8 -*-
"""
Chunked conversion of paper–paper similarity layer → author–author similarity layer.

Usage:
    python make_author_similarity_layer_chunked.py <discipline>

Example:
    python make_author_similarity_layer_chunked.py Math
    python make_author_similarity_layer_chunked.py Physics

This version:
  - Streams the paper similarity file in chunks (10M edges by default)
  - Writes temporary author–author partials to disk after each chunk
  - Merges them at the end with summed weights
"""

import os, sys, csv, glob
from collections import defaultdict
import pandas as pd

# ===== Configuration =====
CHUNK_SIZE = 10_000_000  # number of paper edges per chunk
base = "/N/slate/gpanayio/scisci-gatekeepers/obj"
base_tmp = "/N/scratch/gpanayio/"
metadata_path = "/N/scratch/gpanayio/openalex-pre/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
author_col = "authorships:author:id"
id_col = "id"

# ===== Discipline argument =====
if len(sys.argv) < 2:
    print("Usage: python make_author_similarity_layer_chunked.py <discipline>")
    sys.exit(1)

disc = sys.argv[1]
paper_sim_path = os.path.join(base, f"paper_similarity_layer_{disc}.edgelist")
out_path = os.path.join(base, f"filtered_author_similarity_layer_{disc}.edgelist")
tmp_dir = os.path.join(base_tmp, f"tmp_author_edges_{disc}")
os.makedirs(tmp_dir, exist_ok=True)

print(f"Discipline: {disc}")
print(f"Paper similarity file: {paper_sim_path}")
print(f"Metadata file:          {metadata_path}")
print(f"Temporary directory:    {tmp_dir}")
print(f"Final output:           {out_path}")
print("-" * 90)

# ===== STEP 1: Build paper → authors mapping =====
paper_authors = defaultdict(set)

with open(metadata_path) as f:
    header = f.readline().rstrip("\n").split("\t")
    try:
        idx_id = header.index(id_col)
        idx_auth = header.index(author_col)
    except ValueError as e:
        raise RuntimeError(f"Expected columns '{id_col}' and '{author_col}' not found") from e

    for i, line in enumerate(f, start=2):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(idx_id, idx_auth):
            continue
        paper_id = parts[idx_id].split("/")[-1]
        auth_str = parts[idx_auth]
        for a in auth_str.split(","):
            a = a.strip()
            if a:
                paper_authors[paper_id].add(a)
        if i % 1_000_000 == 0:
            print(f"  Parsed {i:,} metadata rows...", flush=True)

print(f"Loaded author sets for {len(paper_authors):,} papers.")
print("-" * 90)

# ===== STEP 2: Stream paper–paper edges in chunks =====
auth_pairs = defaultdict(float)
edge_count = 0
chunk_idx = 0
missing = 0

def flush_chunk():
    """Write the current accumulated author edges to a temporary file and clear memory."""
    global chunk_idx, auth_pairs
    if not auth_pairs:
        return
    tmp_path = os.path.join(tmp_dir, f"author_edges_chunk{chunk_idx:04d}.csv")
    with open(tmp_path, "w", newline="") as fout:
        w = csv.writer(fout, delimiter=" ")
        for (a, b), wgt in auth_pairs.items():
            w.writerow([a, b, wgt])
    print(f"  ✔️ Wrote {len(auth_pairs):,} author edges to {tmp_path}", flush=True)
    auth_pairs.clear()
    chunk_idx += 1

with open(paper_sim_path) as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        p1, p2, w = parts[0], parts[1], float(parts[2])
        A1, A2 = paper_authors.get(p1), paper_authors.get(p2)
        if not A1 or not A2:
            missing += 1
            continue
        norm = len(A1) * len(A2)
        if norm == 0:
            continue
        contrib = w / norm
        for a in A1:
            for b in A2:
                if a == b:
                    continue
                auth_pairs[(a, b)] += contrib
        edge_count += 1
        if edge_count % 1_000_000 == 0:
            print(f"Processed {edge_count:,} paper edges...", flush=True)
        if edge_count % CHUNK_SIZE == 0:
            flush_chunk()

# Flush any remaining edges
flush_chunk()

print(f"Finished streaming {edge_count:,} paper edges.")
print(f"Skipped {missing:,} edges with missing paper–author info.")
print("-" * 90)

# ===== STEP 3: Merge all temporary chunks =====
print("Merging temporary author edge chunks...")

chunk_files = sorted(glob.glob(os.path.join(tmp_dir, "author_edges_chunk*.csv")))
dfs = []
for fp in chunk_files:
    print(f"  Reading {os.path.basename(fp)}")
    df = pd.read_csv(fp, sep=" ", names=["a", "b", "w"], dtype={"a": str, "b": str, "w": float})
    dfs.append(df)
print(f"Concatenating {len(dfs)} chunks...")
merged = pd.concat(dfs, ignore_index=True)
print("Grouping and summing weights...")
merged = merged.groupby(["a", "b"], as_index=False)["w"].sum()

print(f"Final unique author–author edges: {len(merged):,}")
print(f"Writing to {out_path} ...")
merged.to_csv(out_path, sep=" ", header=False, index=False)

print("\n✅ Done. You can now remove the temporary directory if desired:")
print(f"  rm -r {tmp_dir}")
