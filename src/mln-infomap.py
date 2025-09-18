#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infomap overlapping community detection on:
  1. Layer A only (weighted)
  2. Layer B only (unweighted)
  3. Multilayer (with inter-layer coupling ω)

Outputs:
  - layerA_coms.csv, layerB_coms.csv
  - matched_pairs.csv (A↔B community Jaccard matches)
  - multilayer_coms.csv (per ω)
  - bridge_actors.csv (per ω)
  - scores.csv (ONMI & Omega comparisons)
  - author_diagnostics.csv (per-author memberships and flags)

Requires:
  pip install infomap cdlib
"""

import os, sys, csv, argparse
from collections import defaultdict
from infomap import Infomap
from cdlib import evaluation, NodeClustering

# -----------------
# Utils
# -----------------
def read_edgelist(path, weighted=True):
	with open(path) as f:
		for line in f:
			if not line.strip() or line.startswith("#"):
				continue
			parts = line.split()
			if weighted and len(parts) >= 3:
				u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
			else:
				u, v, w = int(parts[0]), int(parts[1]), 1.0
			yield u, v, w

def extract_overlap(im):
	com2nodes = defaultdict(set)
	for phys_id, module_path in im.get_multilevel_modules(states=True).items():
		cid = module_path[-1]
		com2nodes[cid].add(phys_id)
	return dict(com2nodes)

def invert(coms):
	node2coms = defaultdict(set)
	for cid, nodes in coms.items():
		for n in nodes:
			node2coms[n].add(cid)
	return dict(node2coms)

def save_coms_csv(path, coms):
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["community_id", "nodes"])
		for cid, nodes in sorted(coms.items()):
			w.writerow([cid, "|".join(map(str, sorted(nodes)))])

def jaccard(A, B):
	return len(A & B) / len(A | B) if (A or B) else 0.0

def match_by_jaccard(A_coms, B_coms, tau=0.2):
	pairs = []
	usedB = set()
	for a_id, Aset in A_coms.items():
		best_b, best_j = None, -1.0
		for b_id, Bset in B_coms.items():
			if b_id in usedB:
				continue
			j = jaccard(Aset, Bset)
			if j > best_j:
				best_j, best_b = j, b_id
		if best_b is not None and best_j >= tau:
			inter = len(A_coms[a_id] & B_coms[best_b])
			uni = len(A_coms[a_id] | B_coms[best_b])
			pairs.append((a_id, best_b, best_j, inter, uni))
			usedB.add(best_b)
	return pairs

def write_matches_csv(path, pairs):
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["A_id","B_id","Jaccard","Intersection_size","Union_size"])
		for a,b,j,i,u in pairs:
			w.writerow([a,b,f"{j:.6f}",i,u])

def rank_bridge_actors(multilayer_coms, topk=100):
	actor_to_count = defaultdict(int)
	for cid, nodes in multilayer_coms.items():
		for v in nodes:
			actor_to_count[v] += 1
	results = {}
	for cid, nodes in multilayer_coms.items():
		scored = [(v, actor_to_count[v]) for v in nodes]
		scored.sort(key=lambda x: x[1], reverse=True)
		results[cid] = [v for v,c in scored[:topk] if c>1]
	return results

def save_bridge_csv(path, bridge_map):
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["community_id","bridge_actors"])
		for cid, actors in sorted(bridge_map.items()):
			if actors:
				w.writerow([cid,"|".join(map(str,actors))])

def to_cdlib(coms, all_nodes):
	communities = [list(nodes) for nodes in coms.values()]
	return NodeClustering(communities, list(all_nodes), overlap=True)

# -----------------
# Infomap runners
# -----------------
def run_single_layer(path, weighted=True, undirected=True, num_trials=50, seed=42):
	im = Infomap(silent=True, num_trials=num_trials, two_level=True,
				 undirected=undirected, seed=seed)
	for u,v,w in read_edgelist(path, weighted=weighted):
		im.add_link(u,v,w)
	im.run()
	return extract_overlap(im)

def run_multilayer(pathA, pathB, omega=0.1, num_trials=50, seed=42,
				   weightedA=True, weightedB=False):
	im = Infomap(silent=True, num_trials=num_trials, two_level=True,
				 directed=True, seed=seed)
	actorsA, actorsB = set(), set()
	for u,v,w in read_edgelist(pathA, weighted=weightedA):
		im.add_multilayer_intra_link(0,u,v,w)
		actorsA.update((u,v))
	for u,v,w in read_edgelist(pathB, weighted=weightedB):
		im.add_multilayer_intra_link(1,u,v,w)
		actorsB.update((u,v))
	for a in (actorsA & actorsB):
		im.add_multilayer_inter_link(0,a,1,omega)
	im.run()
	return extract_overlap(im)

# -----------------
# Main
# -----------------
def main(layerA, layerB, outdir, omega_list,
		 weightedA=True, weightedB=False, tau=0.2, num_trials=50, seed=42):

	os.makedirs(outdir, exist_ok=True)

	# Layer A
	print("Running Layer A...")
	comsA = run_single_layer(layerA, weighted=weightedA, num_trials=num_trials, seed=seed)
	save_coms_csv(os.path.join(outdir,"layerA_coms.csv"), comsA)

	# Layer B
	print("Running Layer B...")
	comsB = run_single_layer(layerB, weighted=weightedB, num_trials=num_trials, seed=seed)
	save_coms_csv(os.path.join(outdir,"layerB_coms.csv"), comsB)

	# Match
	print("Matching A<->B...")
	pairs = match_by_jaccard(comsA, comsB, tau=tau)
	write_matches_csv(os.path.join(outdir,"matched_pairs.csv"), pairs)

	# Eval A vs B
	print("Computing ONMI & Omega...")
	all_nodes = set().union(*comsA.values(), *comsB.values())
	A_cd = to_cdlib(comsA, all_nodes)
	B_cd = to_cdlib(comsB, all_nodes)
	onmi_AB = evaluation.overlapping_normalized_mutual_information_MGH(A_cd,B_cd).score
	omega_AB = evaluation.omega_index(A_cd,B_cd).score

	scores = [("A_vs_B", onmi_AB, omega_AB)]

	# Multilayer runs
	for w in omega_list:
		subdir = os.path.join(outdir, f"multilayer_omega_{w:g}")
		os.makedirs(subdir, exist_ok=True)
		print(f"Running multilayer ω={w}...")
		comsM = run_multilayer(layerA, layerB, omega=w,
							   weightedA=weightedA, weightedB=weightedB,
							   num_trials=num_trials, seed=seed)
		save_coms_csv(os.path.join(subdir,"multilayer_coms.csv"), comsM)
		bridges = rank_bridge_actors(comsM, topk=200)
		save_bridge_csv(os.path.join(subdir,"bridge_actors.csv"), bridges)

		# Eval vs A and vs B
		all_nodesM = set().union(*comsM.values())
		M_cd = to_cdlib(comsM, all_nodesM)
		onmi_AM = evaluation.overlapping_normalized_mutual_information_MGH(A_cd,M_cd).score
		omega_AM = evaluation.omega_index(A_cd,M_cd).score
		onmi_BM = evaluation.overlapping_normalized_mutual_information_MGH(B_cd,M_cd).score
		omega_BM = evaluation.omega_index(B_cd,M_cd).score
		scores.append((f"A_vs_M(ω={w})", onmi_AM, omega_AM))
		scores.append((f"B_vs_M(ω={w})", onmi_BM, omega_BM))

		# Author diagnostics
		node2A, node2B, node2M = invert(comsA), invert(comsB), invert(comsM)
		diag_path = os.path.join(subdir,"author_diagnostics.csv")
		with open(diag_path,"w",newline="") as f:
			wri = csv.writer(f)
			wri.writerow(["author_id","layerA_coms","layerB_coms","multilayer_coms","flags"])
			for n in sorted(set(node2A)|set(node2B)|set(node2M)):
				flags=[]
				Aset=node2A.get(n,set()); Bset=node2B.get(n,set()); Mset=node2M.get(n,set())
				if Aset and Bset and Aset.isdisjoint(Bset):
					flags.append("mismatch")
				if len(Mset)>1:
					flags.append("bridge")
				wri.writerow([n,
							  "|".join(map(str,sorted(Aset))) if Aset else "",
							  "|".join(map(str,sorted(Bset))) if Bset else "",
							  "|".join(map(str,sorted(Mset))) if Mset else "",
							  ";".join(flags)])

	# Write scores
	with open(os.path.join(outdir,"scores.csv"),"w",newline="") as f:
		w = csv.writer(f)
		w.writerow(["comparison","ONMI","Omega"])
		for row in scores:
			w.writerow(row)

	print("Done.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--layerA", required=True)
	parser.add_argument("--layerB", required=True)
	parser.add_argument("--outdir", required=True)
	parser.add_argument("--omega-grid", default="0.1", help="Comma-separated omegas")
	parser.add_argument("--tau", type=float, default=0.2)
	parser.add_argument("--num-trials", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--weightedA", action="store_true", default=True)
	parser.add_argument("--weightedB", action="store_true", default=False)
	args = parser.parse_args()

	omega_list = [float(x) for x in args.omega_grid.split(",")]
	main(args.layerA,args.layerB,args.outdir,omega_list,
		 weightedA=args.weightedA, weightedB=args.weightedB,
		 tau=args.tau, num_trials=args.num_trials, seed=args.seed)
