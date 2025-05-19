import requests
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
from itertools import combinations

# --- CONFIG ---
CONCEPT_IDS = [
    "C121332964",    # Physics
    "C41008148",     # Computer Science
    "C2778806576",   # Complex Network
    "C2778393361"    # Social Network Analysis
]

# Globals
raw_path="../data"
obj_path="../obj"

OUTPUT_NODES = "coauthor_nodes.csv"
OUTPUT_EDGES = "coauthor_edges.csv"

BASE_URL = "https://api.openalex.org/works"
G = nx.Graph()

def get_all_openalex_works(concept_ids, from_year, to_year):
    works = []
    cursor = "*"
    concept_filter = "|".join(concept_ids)
    total_count = 0

    while cursor:
        url = (
            f"{BASE_URL}?filter=from_publication_date:{from_year}-01-01,"
            f"to_publication_date:{to_year}-12-31,"
            f"concepts.id:{concept_filter}&per-page=200&cursor={cursor}"
        )
        print(f"Fetching page: cursor={cursor} | Total works so far: {total_count}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()
        results = data.get("results", [])
        works.extend(results)
        total_count += len(results)

        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(1)  # Avoid rate limiting

    return works

def build_graph_and_data(works):
    for work in works:
        authorships = work.get("authorships", [])
        authors = []
        for a in authorships:
            name = a["author"].get("display_name", "Unknown")
            institution = (
                a.get("institutions", [{}])[0].get("id", "Unknown")
                if a.get("institutions")
                else "Unknown"
            )
            authors.append(name)

            if name not in G:
                G.add_node(name, institution_id=institution)

        for a1, a2 in combinations(set(authors), 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)

def export_to_csv(graph, node_file, edge_file):
    node_data = [
        {"author": n, "institution_id": graph.nodes[n].get("institution_id", "Unknown")}
        for n in graph.nodes
    ]
    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv(node_file, index=False)

    edge_data = [
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in graph.edges(data=True)
    ]
    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv(edge_file, index=False)

# --- MAIN ---
# Import year from arguments

if len(sys.argv)>1:
	from_year=int(sys.argv[1])
else:
	from_year=2025 # Set as default

print("Querying OpenAlex...")
works = get_all_openalex_works(CONCEPT_IDS, from_year, from_year)

# Save to pickle
with open(f"{obj_path}/works_{from_year}.obj","wb") as f_out:
	pickle.dump(works,f_out)
print(f"Saved to pickle obj.")

print("Building co-authorship graph...")
build_graph_and_data(works)

# Save G to pickle
with open(f"{obj_path}/citation_{from_year}.nx","wb") as f_out:
	pickle.dump(G,f_out)

print("Saving graph to CSV files...")
export_to_csv(G, f"{obj_path}/{OUTPUT_NODES}", f"{obj_path}/{OUTPUT_EDGES}")
