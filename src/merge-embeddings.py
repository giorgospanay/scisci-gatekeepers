import numpy as np
import os
import glob
from tqdm import tqdm

# === Paths (from your config) ===
raw_workspace_path = "/N/slate/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
chunk_dir = f"{out_scratch_path}/embeddings"
merged_embeddings_path = f"{obj_path}/merged_embeddings.npy"
merged_ids_path = f"{obj_path}/merged_ids.txt"

# === Locate chunk files ===
embedding_files = sorted(glob.glob(os.path.join(chunk_dir, "embeddings_*.npy")))
id_files = sorted(glob.glob(os.path.join(chunk_dir, "ids_*.txt")))

assert len(embedding_files) == len(id_files), "Mismatch between .npy and .txt chunk files."

print(f"Found {len(embedding_files)} embedding chunks to merge...")

# === Merge embeddings ===
all_embeddings = []
for file in tqdm(embedding_files, desc="Merging embeddings"):
    emb = np.load(file)
    all_embeddings.append(emb)

print("🔗 Concatenating all embeddings...")
merged = np.concatenate(all_embeddings, axis=0)
os.makedirs(obj_path, exist_ok=True)
np.save(merged_embeddings_path, merged)
print(f"Merged embeddings saved to: {merged_embeddings_path}")

# === Merge ID files ===
with open(merged_ids_path, "w") as fout:
    for file in tqdm(id_files, desc="Merging IDs"):
        with open(file, "r") as fin:
            for line in fin:
                fout.write(line)

print(f"Merged IDs saved to: {merged_ids_path}")
