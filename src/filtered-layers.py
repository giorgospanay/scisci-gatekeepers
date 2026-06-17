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
disciplines = ["CS", "Math", "Physics", "Biology", "Sociology", "Economics", "Linguistics"]

embedding_dim = 768
nlist_default = 4096   # minimum clusters for IVF
nprobe = 32
top_k = 100

# Training and adding
target_train_cap = 500_000
add_minibatch = 10_000

# Similarity accumulation — flush to disk periodically to bound memory
similarity_threshold = 0.7
flush_every_n_pairs = 20_000_000   # flush author_pair dict to disk once it grows this large

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
# Paper → authors lookup
# (loaded once, shared across all disciplines)
# =========================
def load_paper_authors(metadata_path, chunksize=200_000):
	print("Loading paper→author mapping from metadata...")
	paper_authors = {}
	usecols = ["id", "authorships:author:id"]
	for chunk in tqdm(pd.read_csv(metadata_path, sep="\t", dtype=str,
								  chunksize=chunksize, usecols=usecols),
					  desc="Metadata (paper→authors)"):
		chunk = chunk.dropna(subset=["authorships:author:id"])
		for _, row in chunk.iterrows():
			try:
				authors = ast.literal_eval(row["authorships:author:id"])
			except Exception:
				authors = [a.strip() for a in row["authorships:author:id"].split(",") if a.strip()]
			if authors:
				paper_authors[row["id"]] = authors
	print(f"Loaded author lists for {len(paper_authors):,} papers.")
	return paper_authors

# =========================
# Similarity layer
# (author–author, projected directly from FAISS results)
# =========================
def build_similarity_layer(disc, paper_authors):
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

	# ---- Checkpoint paths for the built index ----
	index_ckpt_path = os.path.join(out_scratch_path, f"faiss_index_{disc}.bin")
	ids_ckpt_path = os.path.join(out_scratch_path, f"faiss_index_{disc}_ids.txt")

	if os.path.exists(index_ckpt_path) and os.path.exists(ids_ckpt_path):
		print(f"{disc}: found existing index checkpoint, loading instead of rebuilding...")
		index = faiss.read_index(index_ckpt_path)
		with open(ids_ckpt_path) as f:
			all_ids = [ln.rstrip("\n") for ln in f]
		print(f"{disc}: loaded index with {index.ntotal:,} vectors, {len(all_ids):,} ids")
	else:
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
			nlist = max(4096, int(math.sqrt(n_total)))
			if n_train < nlist:
				nlist = max(4096, min(nlist, n_train))
			m = 64
			nbits = 8
			print(f"{disc}: using IVF-PQ (CPU) with nlist={nlist}, m={m}, nbits={nbits}, nprobe={nprobe}")
			quantizer = faiss.IndexFlatIP(embedding_dim)
			index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
			index.train(sample_X)
			index.nprobe = nprobe

		# ---- Add embeddings ----
		print("Adding embeddings to index...")
		all_ids = []
		for (chunk_num, id_path, emb_path, n) in tqdm(chunks, desc=f"Adding {disc}"):
			ids, X = load_ids_embeddings(id_path, emb_path)
			X = X.astype("float32")
			faiss.normalize_L2(X)
			for s in range(0, len(X), add_minibatch):
				index.add(X[s:min(s + add_minibatch, len(X))])
			all_ids.extend(ids)

		print(f"{disc}: index built with {index.ntotal:,} vectors")

		# ---- Save checkpoint so a crash during querying doesn't force a rebuild ----
		print(f"{disc}: saving index checkpoint to {index_ckpt_path}")
		faiss.write_index(index, index_ckpt_path)
		with open(ids_ckpt_path, "w") as f:
			for pid in all_ids:
				f.write(pid + "\n")

	# ---- Query and project directly to author–author edges ----
	# For each paper pair (src, tgt) with sim >= threshold:
	#   distribute sim / (|authors_src| * |authors_tgt|) to each author pair.
	# To bound memory, the in-memory dict is periodically flushed to a
	# temporary chunk file on scratch, then all chunks are merged at the end
	# via external sort + awk (same averaging logic as the old merge step,
	# just run automatically here).
	print(f"{disc}: querying and projecting to author–author edges...")
	from collections import defaultdict
	import subprocess

	tmp_dir = os.path.join(out_scratch_path, f"tmp_author_edges_{disc}")
	os.makedirs(tmp_dir, exist_ok=True)

	def flush_to_disk(pair_sums, pair_counts):
		if not pair_sums:
			return
		tmp_path = os.path.join(tmp_dir, f"author_edges_chunk{flush_to_disk.counter:04d}.csv")
		with open(tmp_path, "w") as fout:
			for (a, b), total in pair_sums.items():
				fout.write(f"{a} {b} {total} {pair_counts[(a, b)]}\n")
		flush_to_disk.counter += 1
		print(f"  flushed {len(pair_sums):,} pairs -> {tmp_path}", flush=True)
	flush_to_disk.counter = 0

	author_pair_sums = defaultdict(float)
	author_pair_counts = defaultdict(int)

	for (chunk_num, id_path, emb_path, n) in tqdm(chunks, desc=f"Querying {disc}"):
		ids, X = load_ids_embeddings(id_path, emb_path)
		if not ids:
			continue
		X = X.astype("float32")
		faiss.normalize_L2(X)
		D, I = index.search(X, top_k)

		for i_src, src_id in enumerate(ids):
			authors_src = paper_authors.get(src_id)
			if not authors_src:
				continue
			for rank in range(top_k):
				tgt_idx = I[i_src, rank]
				sim = D[i_src, rank]
				if tgt_idx < 0 or sim < similarity_threshold:
					continue
				tgt_id = all_ids[tgt_idx]
				if tgt_id == src_id:
					continue
				authors_tgt = paper_authors.get(tgt_id)
				if not authors_tgt:
					continue
				norm = len(authors_src) * len(authors_tgt)
				contrib = sim / norm
				for a in authors_src:
					for b in authors_tgt:
						if a == b:
							continue
						key = (a, b) if a <= b else (b, a)
						author_pair_sums[key] += contrib
						author_pair_counts[key] += 1

		# Flush periodically to bound memory
		if len(author_pair_sums) >= flush_every_n_pairs:
			flush_to_disk(author_pair_sums, author_pair_counts)
			author_pair_sums.clear()
			author_pair_counts.clear()

	# Final flush of whatever remains
	flush_to_disk(author_pair_sums, author_pair_counts)
	author_pair_sums.clear()
	author_pair_counts.clear()

	# ---- Merge all chunk files on disk, averaging duplicate pairs ----
	chunk_files = sorted(
		os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("author_edges_chunk")
	)
	print(f"{disc}: merging {len(chunk_files)} chunk files into {out_file} ...")

	cpus = os.getenv("SLURM_CPUS_PER_TASK", "1")
	nmem = 100
	cat_cmd = "cat " + " ".join(chunk_files)
	sort_cmd = f"sort -S {nmem}G -T {out_scratch_path} --parallel={cpus} -k1,1 -k2,2"
	# Each line is: a b sum count -- combine duplicate (a,b) by summing sum and count, then divide
	awk_cmd = (
		"awk 'BEGIN{OFS=\"\\t\"} "
		"{key=$1\" \"$2; if(key==prev){sum+=$3; cnt+=$4}"
		"else{if(NR>1)print prev_a,prev_b,sum/cnt;"
		"split(key,arr,\" \"); prev_a=arr[1]; prev_b=arr[2]; sum=$3; cnt=$4;} prev=key;} "
		"END{if(NR>0)print prev_a,prev_b,sum/cnt;}'"
	)
	cmd = f"{cat_cmd} | {sort_cmd} | {awk_cmd} > {out_file}"
	ret = subprocess.call(cmd, shell=True)
	if ret != 0:
		raise RuntimeError(f"{disc}: merge command failed with exit code {ret}")

	print(f"{disc}: cleaning up temporary chunk files in {tmp_dir}")
	for f in chunk_files:
		os.remove(f)
	os.rmdir(tmp_dir)

	print(f"{disc}: cleaning up index checkpoint")
	if os.path.exists(index_ckpt_path):
		os.remove(index_ckpt_path)
	if os.path.exists(ids_ckpt_path):
		os.remove(ids_ckpt_path)

	print(f"Finished {disc}: similarity layer written to {out_file}")

# =========================
# Main
# =========================
if __name__ == "__main__":
	# Load paper→authors mapping once — shared across all disciplines
	paper_authors = load_paper_authors(metadata_path)

	for disc in disciplines:
		print("\n" + "="*50)
		print(f"Starting discipline: {disc}")
		print("="*50)
		build_collaboration_layer(disc)
		build_similarity_layer(disc, paper_authors)
		print(f"Completed {disc}")
		print("="*50)

	print("All disciplines processed.")