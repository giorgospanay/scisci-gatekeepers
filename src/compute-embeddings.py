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

# Data path
tsv_file = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"


# Parameters
input_file = "<YOUR_INPUT_FILE>.csv"
output_emb_file = f"{obj_path}/paper_embeddings.npy"
output_meta_file = f"{obj_path}/paper_metadata.jsonl"
chunksize = 5000

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.cpu().numpy()

all_embeddings = []
with open(output_meta_file, "w") as meta_out:
    for chunk in tqdm(pd.read_csv(input_file, usecols=["abstract", "authorships:author:id"], chunksize=chunksize)):
        chunk = chunk.dropna(subset=["abstract", "authorships:author:id"])
        abstracts = chunk["abstract"].tolist()
        author_lists = chunk["authorships:author:id"].apply(lambda x: [a.strip() for a in x.split(",") if a.strip()]).tolist()
        
        # Compute and save embeddings
        emb = embed(abstracts)
        all_embeddings.append(emb)

        # Save metadata
        for authors in author_lists:
            json.dump({"authors": authors}, meta_out)
            meta_out.write("\n")

# Save all embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(output_emb_file, all_embeddings)
