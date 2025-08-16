import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import itertools
from collections import defaultdict

# Workspace path
raw_workspace_path="/N/slate/gpanayio/openalex-pre"
out_workspace_path="/N/slate/gpanayio/scisci-gatekeepers"

# Globals
raw_path=f"{raw_workspace_path}/data"
obj_path=f"{out_workspace_path}/obj"

# Data path
tsv_file = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"


# === Config ===
embedding_path = f"{obj_path}/paper_embeddings.npy"
metadata_path = f"{obj_path}/paper_metadata.jsonl"
output_edge_path = f"{obj_path}/author_similarity.edgelist"


print("Loading embeddings...")
embeddings = np.load(embedding_path)
paper_authors = []

print("Loading metadata...")
with open(metadata_path) as f:
    for line in f:
        paper_authors.append(json.loads(line)["authors"])

assert len(embeddings) == len(paper_authors)

print("Computing cosine similarities...")
similarities = cosine_similarity(embeddings)

# Build author-author similarity weights
author_pair_weights = defaultdict(list)

print("Aggregating author pair similarities...")
for i in tqdm(range(len(embeddings))):
    authors_i = set(paper_authors[i])
    for j in range(i + 1, len(embeddings)):
        authors_j = set(paper_authors[j])
        sim = similarities[i, j]
        for a1, a2 in itertools.product(authors_i, authors_j):
            if a1 == a2:
                continue
            key = tuple(sorted((a1, a2)))
            author_pair_weights[key].append(sim)

print("Saving weighted edgelist...")
with open(output_edge_path, "w") as f_out:
    for (a1, a2), sim_list in author_pair_weights.items():
        avg_sim = np.mean(sim_list)
        f_out.write(f"{a1},{a2},{avg_sim:.4f}\n")

print("Done.")
