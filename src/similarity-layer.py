import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from collections import defaultdict

# === Paths ===
obj_path = "/N/slate/gpanayio/scisci-gatekeepers/obj"
raw_path = "/N/slate/gpanayio/openalex-pre/data"

embedding_path = f"{obj_path}/merged_embeddings.npy"
id_path = f"{obj_path}/merged_ids.txt"
metadata_path = f"{raw_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

output_path = f"{obj_path}/author_similarity_layer.csv"

# === Load embeddings + paper IDs ===
print("Loading embeddings and paper IDs...")
embeddings = np.load(embedding_path)
with open(id_path) as f:
    paper_ids = [line.strip() for line in f]

assert len(paper_ids) == embeddings.shape[0]

# === Build paper_id → index map ===
paper_index = {pid: i for i, pid in enumerate(paper_ids)}

# === Load author data from metadata ===
print("Loading author–paper mapping...")
df = pd.read_csv(metadata_path, sep="\t", usecols=["id", "authorships:author:id"], dtype=str)
df = df.dropna(subset=["authorships:author:id"])
df = df[df["id"].isin(paper_index)]

# === Build author → paper index list ===
author_papers = defaultdict(list)

for _, row in df.iterrows():
    paper_id = row["id"]
    authors = [a.strip() for a in row["authorships:author:id"].split(",") if a.strip()]
    for author in authors:
        author_papers[author].append(paper_index[paper_id])

print(f"Found {len(author_papers)} authors.")

# === Compute pairwise author similarities ===
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

    if avg_sim > 0.7:  # ⚠️ Optional: threshold to save space
        rows.append((a1, a2, avg_sim))

# === Save to CSV ===
df_out = pd.DataFrame(rows, columns=["author1", "author2", "similarity"])
df_out.to_csv(output_path, index=False)
print(f"Author similarity layer saved to: {output_path}")
