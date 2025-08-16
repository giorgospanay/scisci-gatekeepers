import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
import json

# Load SPECTER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter").to(device)
model.eval()


# Workspace path
raw_workspace_path="/N/slate/gpanayio/openalex-pre"
out_workspace_path="/N/slate/gpanayio/scisci-gatekeepers"

# Globals
raw_path=f"{raw_workspace_path}/data"
obj_path=f"{out_workspace_path}/obj"


# === Config ===
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
abstracts_path = f"{raw_workspace_path}/works_core+abstract.tsv"
output_emb_path = f"{obj_path}/paper_embeddings.npy"
output_meta_path = f"{obj_path}/paper_metadata.jsonl"
chunksize = 5000

# === Load SPECTER ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter").to(device)
model.eval()

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# === Load abstracts and metadata ===
print("Loading metadata...")
meta_cols = ["id", "authorships:author:id"]
metadata = pd.read_csv(metadata_path, sep="\t", usecols=meta_cols, dtype=str)
abstracts = pd.read_csv(abstracts_path, sep="\t", usecols=["id", "abstract"], dtype=str)

print("Merging...")
df = pd.merge(metadata, abstracts, on="id").dropna(subset=["abstract", "authorships:author:id"])

# === Compute and save ===
all_embeddings = []
with open(output_meta_path, "w") as meta_out:
    for i in tqdm(range(0, len(df), chunksize)):
        chunk = df.iloc[i:i+chunksize]
        abstracts = chunk["abstract"].tolist()
        author_lists = chunk["authorships:author:id"].apply(
            lambda x: [a.strip() for a in x.split(",") if a.strip()]
        ).tolist()

        embs = embed(abstracts)
        all_embeddings.append(embs)

        for authors in author_lists:
            json.dump({"authors": authors}, meta_out)
            meta_out.write("\n")

# Save embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(output_emb_path, all_embeddings)

print("Done.")
