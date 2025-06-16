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

output_edges = f"{obj_path}/coauthorship.edgelist"

with open(output_edges, "w") as f_out:
	chunksize = 10_000
	for chunk in pd.read_csv(tsv_file, sep="\t", usecols=["authorships:author:id"], chunksize=chunksize, dtype=str):
		for authors_str in chunk["authorships:author:id"].fillna(""):
			authors = authors_str.split(",")
			authors = [a.strip() for a in authors if a.strip()]
			if len(authors) < 2:
				continue

			auth_set=set(authors)

			# Write undirected edges (only once per pair)
			for u, v in itertools.combinations(auth_set,2):
				f_out.write(f"{u} {v}\n")