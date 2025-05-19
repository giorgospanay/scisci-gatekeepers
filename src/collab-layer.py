import pandas as pd
import networkit as nk
import itertools
import pickle


# Globals
raw_path="../data"
obj_path="../obj"

# Data path
tsv_file = "/N/slate/gpanayio/openalex-pre/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"



# Initialize graph and author mapping
author_to_node = {}
node_id_counter = 0
G = nk.graph.Graph(weighted=False, directed=False)

# Stream through the dataset
chunksize = 10_000  # adjust based on your system


for chunk in pd.read_csv(tsv_file, sep="\t", usecols=["authorships:author:id"], chunksize=chunksize, dtype=str):
    for authors_str in chunk["authorships:author:id"].fillna(""):
        authors = authors_str.split(",") # comma separated
        authors = [a.strip() for a in authors if a.strip()]  # remove empty/whitespace entries

        if len(authors) < 2:
            continue  # skip solo authors

        # Map authors to node IDs
        node_ids = []
        for author in authors:
            if author not in author_to_node:
                author_to_node[author] = node_id_counter
                G.addNode()  # Ensure the graph has the node
                node_id_counter += 1
            node_ids.append(author_to_node[author])

        # Add edges for all unique coauthor pairs (no weights, undirected)
        for u, v in itertools.combinations(sorted(set(node_ids)), 2):
            if not G.hasEdge(u, v):
                G.addEdge(u, v)

# Save G to pickle
with open(f"{obj_path}/collab-layer.nk","wb") as f_out:
    pickle.dump(G,f_out)

print(f"Graph built with {G.numberOfNodes()} nodes and {G.numberOfEdges()} edges.")
