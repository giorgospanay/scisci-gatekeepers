import pandas as pd
import networkx as nx
import pickle
from itertools import combinations


# Globals
raw_path="../data"
obj_path="../obj"


def clean_ids(row):
	raw_str=row
	if len(raw_str)>0:
		raw_tok=raw_str.split("/")
		return raw_tok[len(raw_tok)-1]
	return "NA"



# Read authorship data, drop other cols. Replace missing data with NA:s
df=pd.read_csv(f"{raw_path}/works_authorships.csv",usecols=["work_id","author_id","institution_id"])
df=df.fillna("/NA")
print("File read. Somehow.")


# Process ids to after last slash
df["work_id"]=df["work_id"].apply(clean_ids)
df["author_id"]=df["author_id"].apply(clean_ids)
df["institution_id"]=df["institution_id"].apply(clean_ids)
print("IDs cleaned.")


# Group by work_id and create edges between co-authors
G = nx.Graph()
for work_id, group in df.groupby("work_id"):
	authors = group["author_id"].unique()

	# Find unique author combinations. Generate edges between them
	for author1, author2 in combinations(authors, 2):
		# Skip self edges
		if author1==author2: continue

		if G.has_edge(author1, author2):
			G[author1][author2]['weight'] += 1
		else:
			G.add_edge(author1, author2, weight=1)


# Save network to file
with open(f"{obj_path}/collab.nx","wb") as f_out:
	pickle.dump(G,f_out)
print("Network saved to file.")


print(f"N={G.number_of_nodes()}, M={G.number_of_edges()}")