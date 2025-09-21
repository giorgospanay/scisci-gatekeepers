#!/usr/bin/env python3
import os
import ast
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# === Paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
obj_path = f"{out_workspace_path}/obj"
filtered_embeddings_root = f"{out_scratch_path}/embeddings_filtered"
output_dir = f"{out_workspace_path}/obj"

os.makedirs(output_dir, exist_ok=True)

# === Disciplines to process ===
disciplines = ["CS", "Math", "Physics", "Biology"]

# === FAISS parameters ===
embedding_dim = 768
nlist = 4096
nprobe = 32
top_k = 100
batch_size = 50000

def load_ids_embeddings(id_file, emb_file):
    with open(id_file) as f:
        ids = [ln.strip() for ln in f]
    X = np.load(emb_file, mmap_mode="r")
    if X.shape[0] != len(ids):
        raise ValueError(f"Mismatch: {id_file} has {len(ids)} IDs but {emb_file} has {X.shape[0]} rows")
    return ids, X

# --- Collaboration layer ---
def build_collaboration_layer(disc):
    out_file = os.path.join(output_dir, f"filtered_collaboration_layer_{disc}.edgelist")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        print(f"{disc}: collaboration layer already exists at {out_file}, skipping.")
        return

    print(f"Building collaboration layer for {disc}...")
    id_file = os.path.join(obj_path, f"filtered_paper_ids_{disc}.txt")
    if not os.path.exists(id_file):
        print(f"No paper IDs for {disc}, skipping collaboration layer.")
        return

    with open(id_file) as f:
        paper_ids = set(ln.strip() for ln in f)

    edges = {}
    chunksize = 200000
    usecols = ["id", "authorships:author:id"]
    for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str, chunksize=chunksize, usecols=usecols), desc=f"Metadata {disc}"):
        chunk = chunk[chunk["id"].isin(paper_ids)]
        for _, row in chunk.iterrows():
            authors = ast.literal_eval(row["authorships:author:id"]) if row["authorships:author:id"] else []
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a, b = sorted((authors[i], authors[j]))
                    edges[(a, b)] = edges.get((a, b), 0) + 1

    with open(out_file, "w") as fout:
        for (a, b), w in edges.items():
            fout.write(f"{a}\t{b}\t{w}\n")

    print(f"Finished {disc}: collaboration layer written to {out_file}")

# --- Similarity layer ---
def build_similarity_layer(disc):
    out_file = os.path.join(output_dir, f"filtered_author_similarity_layer_{disc}.edgelist")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        print(f"{disc}: similarity layer already exists at {out_file}, skipping.")
        return

    print(f"Building similarity layer for {disc}...")
    disc_dir = os.path.join(filtered_embeddings_root, disc)
    if not os.path.exists(disc_dir):
        print(f"No embeddings for {disc}, skipping similarity layer.")
        return

    files = sorted([f for f in os.listdir(disc_dir) if f.startswith("ids_") and f.endswith(".txt")])
    if not files:
        print(f"No ID files for {disc}, skipping similarity layer.")
        return

    # Training sample
    sample_X = []
    for id_file in files[:50]:
        chunk_num = id_file.split("_")[1].split(".")[0]
        id_path = os.path.join(disc_dir, id_file)
        emb_path = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")
        ids, X = load_ids_embeddings(id_path, emb_path)
        sample_X.append(X[:20000] if len(X) > 20000 else X)
    sample_X = np.vstack(sample_X).astype("float32")
    faiss.normalize_L2(sample_X)

    quantizer = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(sample_X)

    if faiss.get_num_gpus() > 0:
        print("Using GPU FAISS")
        index = faiss.index_cpu_to_all_gpus(index)

    # Add embeddings
    all_ids = []
    for id_file in tqdm(files, desc=f"Adding {disc}"):
        chunk_num = id_file.split("_")[1].split(".")[0]
        id_path = os.path.join(disc_dir, id_file)
        emb_path = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")
        ids, X = load_ids_embeddings(id_path, emb_path)
        X = X.astype("float32")
        faiss.normalize_L2(X)
        index.add(X)
        all_ids.extend(ids)

    # Query in batches
    with open(out_file, "w") as fout:
        for start in tqdm(range(0, len(all_ids), batch_size), desc=f"Querying {disc}"):
            end = min(start + batch_size, len(all_ids))
            batch_ids = all_ids[start:end]

            X_batch = []
            for id_file in files:
                chunk_num = id_file.split("_")[1].split(".")[0]
                id_path = os.path.join(disc_dir, id_file)
                emb_path = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")
                ids, X = load_ids_embeddings(id_path, emb_path)
                idmap = {pid: i for i, pid in enumerate(ids)}
                for pid in batch_ids:
                    if pid in idmap:
                        X_batch.append(X[idmap[pid]])
            if not X_batch:
                continue
            X_batch = np.vstack(X_batch).astype("float32")
            faiss.normalize_L2(X_batch)

            D, I = index.search(X_batch, top_k)
            for i, src in enumerate(batch_ids):
                for tgt_idx, sim in zip(I[i], D[i]):
                    if tgt_idx < 0 or src == all_ids[tgt_idx]:
                        continue
                    fout.write(f"{src}\t{all_ids[tgt_idx]}\t{sim:.4f}\n")

    print(f"Finished {disc}: similarity layer written to {out_file}")

# --- Main loop ---
for disc in disciplines:
    print("\n" + "="*50)
    print(f"Starting discipline: {disc}")
    print("="*50)
    build_collaboration_layer(disc)
    build_similarity_layer(disc)
    print(f"Completed {disc}")
    print("="*50)

print("All disciplines processed.")
