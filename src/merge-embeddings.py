import numpy as np
import json
import os
from tqdm import tqdm
import glob

# === Config ===
chunk_dir = "out/embeddings"
merged_embeddings_path = "out/merged_embeddings.npy"
merged_metadata_path = "out/merged_metadata.jsonl"

# === Find all chunk files ===
embedding_files = sorted(glob.glob(os.path.join(chunk_dir, "embeddings_*.npy")))
metadata_files = sorted(glob.glob(os.path.join(chunk_dir, "metadata_*.jsonl")))

assert len(embedding_files) == len(metadata_files), "Mismatch between embedding and metadata files."

print(f"🧩 Found {len(embedding_files)} chunks to merge...")

# === Merge embeddings ===
all_embeddings = []
for file in tqdm(embedding_files, desc="Merging embeddings"):
    emb = np.load(file)
    all_embeddings.append(emb)

print("🔗 Concatenating all embeddings...")
merged = np.concatenate(all_embeddings,_
