import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
import json

from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Load SPECTER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter").to(device)
model.eval()


# Workspace path
raw_workspace_path="/N/slate/gpanayio/openalex-pre"
out_workspace_path="/N/slate/gpanayio/scisci-gatekeepers"
out_scratch_path="/N/scratch/gpanayio/"

# Globals
raw_path=f"{raw_workspace_path}/data"
obj_path=f"{out_workspace_path}/obj"
temp_path=f"{out_scratch_path}"


# === Config ===
metadata_path = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
abstracts_path = f"{raw_workspace_path}/works_core+abstract.tsv"
output_emb_path = f"{obj_path}/paper_embeddings.npy"
output_meta_path = f"{obj_path}/paper_metadata.jsonl"
output_dir = f"{temp_path}/embeddings"
chunksize = 5000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model + tokenizer + adapter ===
print("Loading SPECTER2 base and adapter...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model = model.to(device)
model.eval()


# === Load model and adapter ===
print("Loading SPECTER2 + adapter...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model.to(device)
model.eval()

# === Embedding function ===
def embed(papers):
    texts = [p['title'] + tokenizer.sep_token + (p.get('abstract') or '') for p in papers]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt",
                       return_token_type_ids=False, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

# === Load and join data ===
print("Loading data...")
meta_cols = ["id", "title", "authorships:author:id"]
metadata = pd.read_csv(metadata_path, sep="\t", usecols=meta_cols, dtype=str)
abstracts = pd.read_csv(abstracts_path, sep="\t", usecols=["id", "abstract"], dtype=str)
df = pd.merge(metadata, abstracts, on="id").dropna(subset=["title", "abstract", "authorships:author:id"])
print(f"Total papers: {len(df)}")

# === Make output dir ===
os.makedirs(output_dir, exist_ok=True)

# === Process in chunks ===
total_chunks = (len(df) + chunksize - 1) // chunksize

for i in tqdm(range(total_chunks), desc="Embedding chunks"):
    emb_file = os.path.join(output_dir, f"embeddings_{i:04d}.npy")
    meta_file = os.path.join(output_dir, f"metadata_{i:04d}.jsonl")

    # Skip if already processed
    if os.path.exists(emb_file) and os.path.exists(meta_file):
        continue

    chunk = df.iloc[i * chunksize: (i + 1) * chunksize]
    if chunk.empty:
        continue

    papers = chunk[["title", "abstract"]].to_dict(orient="records")
    author_lists = chunk["authorships:author:id"].apply(
        lambda x: [a.strip() for a in x.split(",") if a.strip()]
    ).tolist()

    print(f"Processing chunk {i+1}/{total_chunks}...")

    embeddings = embed(papers)
    np.save(emb_file, embeddings)

    with open(meta_file, "w") as f_out:
        for authors in author_lists:
            json.dump({"authors": authors}, f_out)
            f_out.write("\n")

print("All chunks processed.")


# # === Embedding function ===
# def embed(papers):
#     # Concatenate title + [SEP] + abstract
#     text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
#     inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt",
#                        return_token_type_ids=False, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         output = model(**inputs)
#         embeddings = output.last_hidden_state[:, 0, :]
#     return embeddings.cpu().numpy()

# # === Load and join metadata + abstracts ===
# print("Loading and merging data...")
# meta_cols = ["id", "title", "authorships:author:id"]
# metadata = pd.read_csv(metadata_path, sep="\t", usecols=meta_cols, dtype=str)
# abstracts = pd.read_csv(abstracts_path, sep="\t", usecols=["id", "abstract"], dtype=str)
# df = pd.merge(metadata, abstracts, on="id").dropna(subset=["title", "abstract", "authorships:author:id"])

# print(f"Total valid papers: {len(df)}")

# # === Compute and save embeddings ===
# os.makedirs(os.path.dirname(output_emb_path), exist_ok=True)
# all_embeddings = []

# print("Computing embeddings and writing metadata...")
# with open(output_meta_path, "w") as meta_out:
#     for i in tqdm(range(0, len(df), chunksize)):
#         chunk = df.iloc[i:i + chunksize]
#         papers = chunk[["title", "abstract"]].to_dict(orient="records")
#         author_lists = chunk["authorships:author:id"].apply(
#             lambda x: [a.strip() for a in x.split(",") if a.strip()]
#         ).tolist()

#         embs = embed(papers)
#         all_embeddings.append(embs)

#         for authors in author_lists:
#             json.dump({"authors": authors}, meta_out)
#             meta_out.write("\n")

# print("Saving embeddings...")
# all_embeddings = np.concatenate(all_embeddings, axis=0)
# np.save(output_emb_path, all_embeddings)

# print("Done!")
# print(f" - Embeddings saved to: {output_emb_path}")
# print(f" - Metadata saved to:   {output_meta_path}")