#!/usr/bin/env python3
import os
import ast
import math
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# =========================
# Paths (adjust if needed)
# =========================
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
obj_path = f"{out_workspace_path}/obj"
filtered_embeddings_root = f"{out_scratch_path}/embeddings_filtered"
output_dir = obj_path  # final edgelists live on SLATE

os.makedirs(output_dir, exist_ok=True)

# =========================
# Configuration
# =========================
disciplines = ["CS", "Math", "Physics", "Biology"]

embedding_dim = 768
nlist_default = 4096   # minimum clusters for IVF
nprobe = 32
top_k = 100

# Training and adding
target_train_cap = 500_000
add_minibatch = 10_000

# Metadata
meta_chunksize = 200_000
meta_usecols = ["id", "authorships:author:id"]

# =========================
# Helpers
# =========================
def load_ids_embeddings(id_file, emb_file):
	with open(id_file) as f:
		ids = [ln.strip() for ln in f]
	X = np.load(emb_file, mmap_mode="r")
	if X.shape[0] != len(ids):
		raise ValueError(f"Mismatch: {id_file} has {len(ids)} IDs but {emb_file} has {X.shape[0]} rows")
	return ids, X

def list_discipline_chunks(disc_dir):
	files = sorted([f for f in os.listdir(disc_dir) if f.startswith("ids_") and f.endswith(".txt")])
	info = []
	for id_file in files:
		chunk_num = id_file.split("_")[1].split(".")[0]
		id_path = os.path.join(disc_dir, id_file)
		emb_path = os.path.join(disc_dir, f"embeddings_{chunk_num}.npy")
		if not os.path.exists(emb_path):
			continue
		with open(id_path) as f:
			n = sum(1 for _ in f)
		info.append((chunk_num, id_path, emb_path, n))
	return info

# =========================
# Collaboration layer
# =========================
def build_collaboration_layer(disc):
	out_file = os.path.join(output_dir, f"filtered_collaboration_layer_{disc}.edgelist")
	if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
		print(f"{disc}: collaboration layer already exists at {out_file}, skipping.")
		return

	id_file = os.path.join(obj_path, f"filtered_paper_ids_{disc}.txt")
	if not os.path.exists(id_file):
		print(f"{disc}: no paper ID list found at {id_file}, skipping collaboration layer.")
		return

	print(f"Building collaboration layer for {disc}...")
	with open(id_file) as f:
		paper_ids = set(ln.strip() for ln in f)

	edges = {}
	for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str,
								  chunksize=meta_chunksize, usecols=meta_usecols),
					  desc=f"Metadata {disc}"):
		chunk = chunk[chunk["id"].isin(paper_ids)]
		if chunk.empty:
			continue
		for _, row in chunk.iterrows():
			try:
				authors = ast.literal_eval(row["authorships:author:id"]) if row["authorships:author:id"] else []
			except Exception:
				authors = []
			L = len(authors)
			for i in range(L):
				ai = authors[i]
				for j in range(i+1, L):
					aj = authors[j]
					a, b = (ai, aj) if ai <= aj else (aj, ai)
					edges[(a, b)] = edges.get((a, b), 0) + 1

	with open(out_file, "w") as fout:
		for (a, b), w in edges.items():
			fout.write(f"{a}\t{b}\t{w}\n")

	print(f"Finished {disc}: collaboration layer written to {out_file}")

# =========================
# Similarity layer
# =========================
def build_similarity_layer(disc):
	out_file = os.path.join(output_dir, f"filtered_author_similarity_layer_{disc}.edgelist")
	if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
		print(f"{disc}: similarity layer already exists at {out_file}, skipping.")
		return

	print(f"Building similarity layer for {disc}...")
	disc_dir = os.path.join(filtered_embeddings_root, disc)
	if not os.path.isdir(disc_dir):
		print(f"{disc}: no embeddings directory at {disc_dir}, skipping.")
		return

	chunks = list_discipline_chunks(disc_dir)
	if not chunks:
		print(f"{disc}: no embedding chunks found, skipping.")
		return

	n_total = sum(n for (_, _, _, n) in chunks)
	print(f"{disc}: total vectors = {n_total:,}")

	# ---- Training sample ----
	print("Preparing training sample...")
	target_train = min(target_train_cap, n_total)
	sample_accum = []
	remaining = target_train
	for (_, id_path, emb_path, n) in chunks:
		if remaining <= 0:
			break
		ids, X = load_ids_embeddings(id_path, emb_path)
		take = min(n, remaining)
		sample_accum.append(X[:take])
		remaining -= take
	sample_X = np.vstack(sample_accum).astype("float32")
	faiss.normalize_L2(sample_X)
	n_train = sample_X.shape[0]
	print(f"{disc}: training sample size = {n_train:,}")

	# ---- Choose index type ----
	if n_total < 1_000_000:
		print(f"{disc}: small dataset (<1M). Using Flat (CPU).")
		index = faiss.IndexFlatIP(embedding_dim)
	else:
		# Ensure nlist >= 4096
		nlist = max(4096, int(math.sqrt(n_total)))
		if n_train < nlist:
			nlist = max(4096, min(nlist, n_train))
		m = 64      # subquantizers
		nbits = 8   # 256 centroids per subquantizer
		print(f"{disc}: using IVF-PQ (CPU) with nlist={nlist}, m={m}, nbits={nbits}, nprobe={nprobe}")
		quantizer = faiss.IndexFlatIP(embedding_dim)
		index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
		index.train(sample_X)
		index.nprobe = nprobe

	# ---- Add embeddings ----
	print("Adding embeddings to index (CPU only, mini-batches)...")
	all_ids = []
	for (chunk_num, id_path, emb_path, n) in tqdm(chunks, desc=f"Adding {disc}"):
		ids, X = load_ids_embeddings(id_path, emb_path)
		X = X.astype("float32")
		faiss.normalize_L2(X)
		for s in range(0, len(X), add_minibatch):
			e = min(s + add_minibatch, len(X))
			index.add(X[s:e])
		all_ids.extend(ids)

	print(f"{disc}: index built with {index.ntotal:,} vectors")

	# ---- Query ----
	print(f"{disc}: writing edges to {out_file} ...")
	with open(out_file, "w") as fout:
		for (chunk_num, id_path, emb_path, n) in tqdm(chunks, desc=f"Querying {disc}"):
			ids, X = load_ids_embeddings(id_path, emb_path)
			if len(ids) == 0:
				continue
			X = X.astype("float32")
			faiss.normalize_L2(X)
			D, I = index.search(X, top_k)
			for i_src, src_id in enumerate(ids):
				for tgt_idx, sim in zip(I[i_src], D[i_src]):
					if tgt_idx < 0:
						continue
					tgt_id = all_ids[tgt_idx]
					if tgt_id == src_id:
						continue
					fout.write(f"{src_id}\t{tgt_id}\t{sim:.4f}\n")

	print(f"Finished {disc}: similarity layer written to {out_file}")

# =========================
# Main
# =========================
if __name__ == "__main__":
	for disc in disciplines:
		print("\n" + "="*50)
		print(f"Starting discipline: {disc}")
		print("="*50)
		build_collaboration_layer(disc)
		build_similarity_layer(disc)
		print(f"Completed {disc}")
		print("="*50)

	print("All disciplines processed.")
