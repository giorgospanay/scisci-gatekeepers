import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from collections import defaultdict

# === Updated workspace paths ===
raw_workspace_path = "/N/scratch/gpanayio/openalex-pre"
out_workspace_path = "/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path = "/N/scratch/gpanayio"

# === Derived paths ===
raw_path = f"{raw_workspace_path}/data"
obj_path = f"{out_workspace_path}/obj"

embedding_path = f"{obj_path}/merged_embeddings.npy"
id_path = f"{obj_path}/merged_ids.txt"
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
output_path = f"{obj_path}/author_similarity.edgelist"



## IMPORTANT GLOBAL: THRESHOLD FOR EDGE
# for now: have edges only above 0.5 sim. Still pretty large network so idk
sim_threshold=0.5



# === Load embeddings and paper IDs ===
print("Loading embeddings and paper IDs...")
embeddings = np.load(embedding_path)
with open(id_path, "r") as f:
    paper_ids = [line.strip() for line in f]

assert embeddings.shape[0] == len(paper_ids)

# === Build paper_id → index map ===
paper_index = {pid: i for i, pid in enumerate(paper_ids)}

# === Load author–paper mapping from metadata ===
print("Loading author–paper mapping...")
df = pd.read_csv(metadata_path, sep="\t", usecols=["id", "authorships:author:id"], dtype=str)
df = df.dropna(subset=["authorships:author:id"])
df = df[df["id"].isin(paper_index)]

# === Build author → paper index mapping ===
author_papers = defaultdict(list)
for _, row in df.iterrows():
    paper_id = row["id"]
    authors = [a.strip() for a in row["authorships:author:id"].split(",") if a.strip()]
    for author in authors:
        author_papers[author].append(paper_index[paper_id])

print(f"Found {len(author_papers)} authors with associated papers.")

# === Compute average pairwise similarities between authors ===
print("Computing average author similarities...")
rows = []
authors = sorted(author_papers.keys())

for a1, a2 in tqdm(itertools.combinations(authors, 2), total=(len(authors) * (len(authors)-1)) // 2):
    idx1 = author_papers[a1]
    idx2 = author_papers[a2]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    sims = cosine_similarity(emb1, emb2)
    avg_sim = sims.mean()

    if avg_sim > sim_threshold:  # Optional: threshold
        rows.append((a1, a2, avg_sim))

# === Save weighted edgelist ===
print("Saving author similarity edgelist...")
with open(output_path, "w") as f:
    for a1, a2, sim in rows:
        f.write(f"{a1},{a2},{sim:.4f}\n")

print(f"Done. Saved to {output_path}")
