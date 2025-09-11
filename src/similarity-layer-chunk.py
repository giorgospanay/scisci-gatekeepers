# compute_author_similarity_chunkwise.py

import numpy as np
import pandas as pd
import json
import os
import itertools
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# === Configurable paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

obj_path = f"{out_workspace_path}/obj"
embedding_chunk_dir = f"{out_scratch_path}/embeddings"
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
similarity_output_dir = f"{obj_path}/author_similarity_chunks"

os.makedirs(similarity_output_dir, exist_ok=True)

similarity_threshold = 0.7

# === Load paper metadata (author IDs) ===
print("Loading full paper metadata...")
df_meta = pd.read_csv(metadata_path, sep="\t", usecols=["id", "authorships:author:id"], dtype=str)
df_meta = df_meta.dropna(subset=["authorships:author:id"])

paper_to_authors = {
    row["id"]: [a.strip() for a in row["authorships:author:id"].split(",") if a.strip()]
    for _, row in df_meta.iterrows()
}

# === List of chunks ===
chunk_files = sorted([f for f in os.listdir(embedding_chunk_dir) if f.startswith("embeddings_") and f.endswith(".npy")])
chunk_ids = [f.replace("embeddings_", "").replace(".npy", "") for f in chunk_files]

print(f"Found {len(chunk_ids)} embedding chunks.")

# === Process chunk pairs ===
for i, j in tqdm(itertools.combinations_with_replacement(chunk_ids, 2), total=(len(chunk_ids)*(len(chunk_ids)+1))//2):
    outfile = os.path.join(similarity_output_dir, f"author_sim_{i}_{j}.jsonl")
    if os.path.exists(outfile):
        continue

    # Load embeddings
    emb_i = np.load(os.path.join(embedding_chunk_dir, f"embeddings_{i}.npy"))
    emb_j = np.load(os.path.join(embedding_chunk_dir, f"embeddings_{j}.npy"))

    # Load paper IDs
    with open(os.path.join(embedding_chunk_dir, f"ids_{i}.txt")) as f:
        ids_i = [line.strip() for line in f]
    with open(os.path.join(embedding_chunk_dir, f"ids_{j}.txt")) as f:
        ids_j = [line.strip() for line in f]

    # Map paper idx to authors
    authors_i = [paper_to_authors.get(pid, []) for pid in ids_i]
    authors_j = [paper_to_authors.get(pid, []) for pid in ids_j]

    # Compute similarities
    sim_matrix = cosine_similarity(emb_i, emb_j)

    pair_sims = defaultdict(list)

    for x in range(len(ids_i)):
        for y in range(len(ids_j)):
            sim = sim_matrix[x, y]
            if sim < similarity_threshold:
                continue
            for a1 in authors_i[x]:
                for a2 in authors_j[y]:
                    if a1 == a2:
                        continue
                    key = tuple(sorted((a1, a2)))
                    pair_sims[key].append(sim)

    # Write partial result
    with open(outfile, "w") as f_out:
        for (a1, a2), sims in pair_sims.items():
            json.dump({"a1": a1, "a2": a2, "avg_sim": float(np.mean(sims))}, f_out)
            f_out.write("\n")

print("Done with all chunk pair similarities.")
