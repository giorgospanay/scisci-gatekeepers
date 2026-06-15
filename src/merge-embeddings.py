import numpy as np
import os
import glob
from tqdm import tqdm

# === Paths (from your config) ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
chunk_dir = f"{out_scratch_path}/embeddings"
merged_embeddings_path = f"{out_scratch_path}/merged_embeddings.npy"
merged_ids_path = f"{out_scratch_path}/merged_ids.txt"

# === Locate chunk files ===
embedding_files = sorted(glob.glob(os.path.join(chunk_dir, "embeddings_*.npy")))
id_files = sorted(glob.glob(os.path.join(chunk_dir, "ids_*.txt")))

assert len(embedding_files) == len(id_files), "Mismatch between .npy and .txt chunk files."

print(f"Found {len(embedding_files)} embedding chunks to merge...")

# === Pre-compute total row count without loading data into RAM ===
print("Counting total rows (mmap peek)...")
total_rows = 0
for file in tqdm(embedding_files, desc="Counting rows"):
    emb = np.load(file, mmap_mode="r")
    total_rows += emb.shape[0]

dim = np.load(embedding_files[0], mmap_mode="r").shape[1]
print(f"Total rows: {total_rows:,}, dim: {dim}")

# === Write merged embeddings incrementally via memmap ===
# This never holds more than one chunk in RAM at a time.
os.makedirs(out_scratch_path, exist_ok=True)
merged = np.memmap(merged_embeddings_path, dtype="float32", mode="w+", shape=(total_rows, dim))

offset = 0
for file in tqdm(embedding_files, desc="Merging embeddings"):
    chunk = np.load(file, mmap_mode="r")
    n = chunk.shape[0]
    merged[offset : offset + n] = chunk
    offset += n

del merged  # flushes to disk
print(f"Merged embeddings saved to: {merged_embeddings_path}")

# === Merge ID files ===
with open(merged_ids_path, "w") as fout:
    for file in tqdm(id_files, desc="Merging IDs"):
        with open(file, "r") as fin:
            for line in fin:
                fout.write(line)

print(f"Merged IDs saved to: {merged_ids_path}")
print("Done.")