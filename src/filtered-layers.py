#!/usr/bin/env python3
import os
import itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import ast
import faiss
import time

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"          # final outputs (SLATE)
scratch_obj_path = f"{out_scratch_path}/obj"   # intermediates (SCRATCH)

metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
filtered_embeddings_root = f"{out_scratch_path}/embeddings_filtered"

# === Disciplines ===
disciplines = ["CS", "Math", "Physics", "Biology", "NetworkScience"]

# === Parameters ===
similarity_threshold = 0.7
top_k = 100
batch_size = 5000

# FAISS IVF params
nlist = 10000
nprobe = 50
use_gpu = True   # set False if running on CPU

valid_types = {
    "journal-article",
    "conference-paper",
    "book-chapter",
    "book",
    "dissertation",
    "report",
    "preprint"
}

# --- Helper: build paper → authors mapping ---
def build_paper_to_authors(id_file, metadata_path):
    with open(id_file) as f:
        paper_ids = set(line.strip() for line in f)

    print(f"Loaded {len(paper_ids)} paper IDs from {id_file}")

    chunksize = 200_000
    paper_to_authors = {}

    for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize)):
        chunk = chunk[chunk["id"].isin(paper_ids) & chunk["type"].isin(valid_types)]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            try:
                authors = ast.literal_eval(row["authorships:author:id"])
                authors = [str(a) for a in authors]
                paper_to_authors[str(row["id"])] = authors
            except Exception:
                continue

    print(f"Built mapping for {len(paper_to_authors)} papers")
    return paper_to_authors

# --- Helper: build collaboration layer ---
def build_collaboration_layer(paper_to_authors, outfile):
    if os.path.exists(outfile):
        print(f"Collaboration layer already exists: {outfile}, skipping...")
        return

    print("Building collaboration layer...")
    collab_edges = defaultdict(int)
    for authors in paper_to_authors.values():
        for a1, a2 in itertools.combinations(sorted(authors), 2):
            collab_edges[(a1, a2)] += 1

    with open(outfile, "w") as f:
        for (a1, a2), w in collab_edges.items():
            f.write(f"{a1}\t{a2}\t{w}\n")

    print(f"Collaboration layer written: {outfile}, edges={len(collab_edges)}")

# --- Helper: build similarity layer ---
def build_similarity_layer(paper_to_authors, embedding_dir, outfile, chunk_dir):
    if os.path.exists(outfile):
        print(f"Similarity layer already exists: {outfile}, skipping...")
        return

    os.makedirs(chunk_dir, exist_ok=True)

    print("Loading embeddings...")
    all_ids = []
    all_embeddings = []

    for fname in os.listdir(embedding_dir):
        if fname.startswith("ids_") and fname.endswith(".txt"):
            chunk_num = fname.split("_")[1].split(".")[0]
            id_path = os.path.join(embedding_dir, fname)
            emb_path = os.path.join(embedding_dir, f"embeddings_{chunk_num}.npy")

            if not os.path.exists(emb_path):
                continue

            with open(id_path) as f:
                ids = [line.strip() for line in f]
            embeddings = np.load(emb_path)

            if len(ids) != embeddings.shape[0]:
                print(f"Warning: mismatch in {fname}, skipping")
                continue

            all_ids.extend(ids)
            all_embeddings.append(embeddings)

    if len(all_ids) == 0:
        print("No embeddings found, skipping similarity layer")
        return

    X = np.vstack(all_embeddings).astype("float32")
    print(f"Loaded embeddings: {len(all_ids)} papers, shape={X.shape}")

    faiss.normalize_L2(X)
    d = X.shape[1]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    print("Training FAISS IVF index...")
    index.train(X)
    index.add(X)
    index.nprobe = nprobe

    if use_gpu:
        print("Moving FAISS index to GPU...")
        index = faiss.index_cpu_to_all_gpus(index)
        print(f"Using {faiss.get_num_gpus()} GPUs")

    num_batches = (len(X) + batch_size - 1) // batch_size
    start_time = time.time()

    for b, i in enumerate(range(0, len(X), batch_size), start=1):
        Xi = X[i:i+batch_size]
        D, I = index.search(Xi, top_k + 1)

        chunk_file = os.path.join(chunk_dir, f"chunk_{i//batch_size:05d}.edgelist")
        with open(chunk_file, "w") as f:
            for ii, row in enumerate(D):
                pid_i = all_ids[i+ii]
                authors_i = paper_to_authors.get(pid_i, [])
                for sim, j in zip(row[1:], I[ii][1:]):
                    if sim < similarity_threshold:
                        continue
                    pid_j = all_ids[j]
                    authors_j = paper_to_authors.get(pid_j, [])
                    for a1 in authors_i:
                        for a2 in authors_j:
                            if a1 == a2:
                                continue
                            if a1 < a2:
                                f.write(f"{a1}\t{a2}\t{sim:.4f}\n")
                            else:
                                f.write(f"{a2}\t{a1}\t{sim:.4f}\n")

        elapsed = time.time() - start_time
        avg_per_batch = elapsed / b
        remaining = avg_per_batch * (num_batches - b)
        print(f"Batch {b}/{num_batches} done "
              f"({b/num_batches:.1%}). "
              f"Elapsed: {elapsed/3600:.2f}h, ETA: {remaining/3600:.2f}h")

    print(f"Chunks written to {chunk_dir}")

    # Merge chunks into final file
    print("Merging chunks into final file...")
    with open(outfile, "w") as fout:
        for fname in sorted(os.listdir(chunk_dir)):
            with open(os.path.join(chunk_dir, fname), "r") as fin:
                for line in fin:
                    fout.write(line)
    print(f"Merged file written: {outfile}")

# --- Main loop ---
for disc in disciplines:
    print("\n==============================")
    print(f"Processing {disc}")
    print("==============================")

    id_file = os.path.join(obj_path, f"filtered_paper_ids_{disc}.txt")
    embedding_dir = os.path.join(filtered_embeddings_root, disc)
    collab_outfile = os.path.join(obj_path, f"filtered_collaboration_layer_{disc}.edgelist")
    author_sim_outfile = os.path.join(obj_path, f"filtered_author_similarity_layer_{disc}.edgelist")
    chunk_dir = os.path.join(scratch_obj_path, f"filtered_author_similarity_chunks_{disc}")

    if not os.path.exists(id_file):
        print(f"No ID file for {disc}, skipping...")
        continue

    paper_to_authors = build_paper_to_authors(id_file, metadata_path)

    build_collaboration_layer(paper_to_authors, collab_outfile)
    build_similarity_layer(paper_to_authors, embedding_dir, author_sim_outfile, chunk_dir)

print("All disciplines processed.")
