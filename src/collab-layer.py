import pandas as pd
import itertools
import pickle


# Workspace path
raw_workspace_path="/N/slate/gpanayio/openalex-pre"
out_workspace_path="/N/slate/gpanayio/scisci-gatekeepers"

# Globals
raw_path=f"{raw_workspace_path}/data"
obj_path=f"{out_workspace_path}/obj"

# Data path
tsv_file = f"{raw_workspace_path}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

author_to_node = {}
node_id_counter = 0
output_file = f"{obj_path}/coauthorship.edgelist"

with open(output_file, "w") as f_out:
    chunksize = 10_000
    for chunk in pd.read_csv(tsv_file, sep="\t", usecols=["authorships:author:id"], chunksize=chunksize, dtype=str):
        for authors_str in chunk["authorships:author:id"].fillna(""):
            authors = authors_str.split(",")
            authors = [a.strip() for a in authors if a.strip()]
            if len(authors) < 2:
                continue

            node_ids = []
            for author in authors:
                if author not in author_to_node:
                    author_to_node[author] = node_id_counter
                    node_id_counter += 1
                node_ids.append(author_to_node[author])

            # Write undirected edges (only once per pair)
            for u, v in itertools.combinations(sorted(set(node_ids)), 2):
                f_out.write(f"{u} {v}\n")
