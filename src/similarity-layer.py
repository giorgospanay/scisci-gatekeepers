import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from collections import defaultdict
from tqdm import tqdm


# Workspace path
raw_workspace_path="/N/slate/gpanayio/openalex-pre"
out_workspace_path="/N/slate/gpanayio/scisci-gatekeepers"

# Globals
raw_path=f"{raw_workspace_path}/data"
obj_path=f"{out_workspace_path}/obj"

# Data path
tsv_file = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"


# Load
embedding_file = f"{obj_path}/paper_embeddings.npy"
metadata_file = f"{obj_path}/paper_metadata.jsonl"
output_edges = f"{obj_path}/author_similarity.edgelist"

embeddings = np.load(embedding_file)
paper_authors = []

with open(metadata_file) as f:
    for line in f:
        paper_authors.append(json.loads(line)["authors"])

N = len(embeddings)
assert N == len(paper_authors), "Mismatch in number of embeddings and metadata"

# Compute pairwise similarities
similarities = cosine_similarity(embeddings)

# Build author-author similarity edges
author_pair_weights = defaultdict(list)

for i in tqdm(range(N)):
    authors_i = set(paper_authors[i])
    for j in range(i + 1, N):
        authors_j = set(paper_authors[j])
        sim = similarities[i, j]
        for a1, a2 in itertools.product(authors_i, authors_j):
            if a1 == a2:
                continue
            key = tuple(sorted((a1, a2)))
            author_pair_weights[key].append(sim)

# Aggregate and save
with open(output_edges, "w") as f_out:
    for (a1, a2), sim_list in author_pair_weights.items():
        avg_sim = np.mean(sim_list)
        f_out.write(f"{a1},{a2},{avg_sim:.4f}\n")
