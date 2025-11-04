#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infomap overlapping community detection, discipline-aware.

Usage:
	python -u mln-infomap.py <mode> <discipline> <out_base> [layerA] [layerB] [omega-grid]

Modes:
	layerA           Run Infomap on Layer A (similarity)
	layerB           Run Infomap on Layer B (collaboration)
	match            Compare Layer A vs B
	multilayer       Run multilayer (needs both layers)
	multilayer-match Compare multilayer runs vs A and B

Arguments:
	discipline   Short name (e.g., "physics", "biology")
	out_base     Base output directory (e.g., /N/slate/.../obj)
	layerA       Path to similarity edgelist
	layerB       Path to collaboration edgelist
	omega-grid   Comma-separated (default: "0.1")

Examples:
	python -u mln-infomap.py layerA Physics /N/slate/.../obj filtered_author_similarity_layer_Physics.edgelist
	python -u mln-infomap.py multilayer Physics /N/slate/.../obj filtered_author_similarity_layer_Physics.edgelist filtered_collaboration_layer_Physics.edgelist "0.05,0.1,0.2"
"""

import os, sys, csv, math, random
from collections import defaultdict
from infomap import Infomap
from cdlib import evaluation, NodeClustering

# ========= ID Mapper =========
class IdMapper:
	def __init__(self): 
		self.forward={}
		self.reverse=[]
	def get(self, orig):
		if orig not in self.forward:
			nid=len(self.reverse)
			self.forward[orig]=nid
			self.reverse.append(orig)
		return self.forward[orig]
	def remap_edge(self,u,v,w): 
		return self.get(u),self.get(v),w
	def save(self,path):
		with open(path,"w",newline="") as f:
			w=csv.writer(f)
			w.writerow(["internal_id","original_id"])
			for nid,orig in enumerate(self.reverse): 
				w.writerow([nid,orig])
	@classmethod
	def load(cls,path):
		mapper=cls()
		with open(path) as f:
			next(f)
			for line in f:
				nid,orig=line.strip().split(",",1)
				nid=int(nid)
				mapper.forward[orig]=nid
				while len(mapper.reverse)<=nid: 
					mapper.reverse.append(None)
				mapper.reverse[nid]=orig
		return mapper

# ========= Utils =========
def read_edgelist(path, weighted=True, report_every=1_000_000, threshold=None):
	count=0
	with open(path) as f:
		for line in f:
			if not line.strip() or line.startswith("#"): continue
			parts=line.split()

			if weighted and len(parts)>=3:
				u,v,w=parts[0],parts[1],float(parts[2])
				if threshold is not None and w<threshold: continue
			else:
				u,v,w=parts[0],parts[1],1.0
			yield u,v,w
			count+=1
			if count%report_every==0: print(f"  ...read {count:,} edges from {os.path.basename(path)}",flush=True)
	print(f"  Finished reading {count:,} edges from {os.path.basename(path)}",flush=True)

def extract_overlap(im,idmap):
	# com2nodes=defaultdict(set)
	# for pid,modpath in im.get_multilevel_modules(states=True).items():
	# 	cid=modpath[-1]
	# 	orig=idmap.reverse[int(pid)]
	# 	com2nodes[cid].add(orig)
	# return dict(com2nodes)
	
	com2nodes = defaultdict(set)
	skipped = 0
	for pid, modpath in im.get_multilevel_modules(states=True).items():
		cid = modpath[-1]
		pid_int = int(pid)
		if pid_int >= len(idmap.reverse):
			skipped += 1
			continue
		orig = idmap.reverse[pid_int]
		com2nodes[cid].add(orig)
	if skipped > 0:
		print(f"Skipped {skipped:,} unmapped state-node IDs (multilayer states).")
	return dict(com2nodes)


def save_coms_csv(path,coms):
	with open(path,"w",newline="") as f:
		w=csv.writer(f); w.writerow(["community_id","nodes"])
		for cid,nodes in sorted(coms.items()):
			w.writerow([cid,"|".join(map(str,sorted(nodes)))])

def to_cdlib(coms): 
	return NodeClustering([list(nodes) for nodes in coms.values()],graph=None,overlap=True)

def pass_stats_over_weights(path, is_weighted=True, sample_every=1, cap_samples=5_000_000):
	n=0; s=0.0; wmax=0.0; sample=[]
	step=max(1,sample_every)
	for i,line in enumerate(open(path)):
		if not line.strip() or line.startswith("#"): continue
		parts=line.split()
		if is_weighted and len(parts)>=3:
			w=float(parts[2])
		else:
			w=1.0
		n+=1; s+=w; 
		if w>wmax: wmax=w
		if (i%step)==0:
			sample.append(w)
			if len(sample)>cap_samples:
				sample=random.sample(sample,len(sample)//2)
	sample.sort()
	def q(p):
		if not sample: return 0.0
		idx=int(p*(len(sample)-1))
		return sample[idx]
	return {"n":n,"sum":s,"max":wmax,"p95":q(0.95),"p99":q(0.99)}

# ========= Weight transforms =========
def sim_sharpen_temp(w, tau=0.02):
	w = max(0.0, min(1.0, w))  # clamp within [0,1]
	return math.exp(-(1.0-w)/max(tau,1e-6))

def sim_sharpen_gamma(w, alpha=3.0):
	return 1.0-(1.0-w)**alpha

def collab_normalize_log1p(w, p99):
	return min(1.0, math.log1p(w)/math.log1p(max(p99,1.0)))

# ========= Runners =========
def run_single_layer(path,idmap,weighted=True,num_trials=1,seed=42,
					 threshold=None,sim_sharpen=False):
	im=Infomap(silent=False,num_trials=num_trials,two_level=True,seed=seed)
	for u,v,w in read_edgelist(path,weighted=weighted,threshold=threshold):
		if sim_sharpen:
			# Add safeguard for duplicated edges
			w = max(0.0, min(1.0, w))  
			w_t = sim_sharpen_temp(w, tau=0.02)
		else:
			w_t=w
		ui,vi,wi=idmap.remap_edge(u,v,w_t)
		im.add_link(ui,vi,wi)
	im.run()
	return extract_overlap(im,idmap)

def run_multilayer(pathA,pathB,idmap,omega=0.1,num_trials=1,seed=42,
				   threshold=None):
	# --- stats pass for collab ---
	statsA=pass_stats_over_weights(pathA,is_weighted=True)
	statsB=pass_stats_over_weights(pathB,is_weighted=True)
	p99_B=statsB["p99"] if statsB["p99"]>0 else 1.0

	# approximate balancing by raw sums
	sA=1.0
	sB=(statsA["sum"] if statsA["sum"]>0 else 1.0)/(statsB["sum"] if statsB["sum"]>0 else 1.0)

	im=Infomap(silent=False,num_trials=num_trials,two_level=True,directed=False,seed=seed)

	actorsA=set()
	actorsB=set()

	# similarity layer
	for u,v,w in read_edgelist(pathA,weighted=True,threshold=threshold):
		# Clamp w to [0,1] to prevent math overflow
		if not isinstance(w, (int, float)):
			continue
		w = max(0.0, min(1.0, w))
		w_t = sim_sharpen_temp(w, tau=0.02) * sA
		ui,vi,wi = idmap.remap_edge(u,v,w_t)
		im.add_multilayer_intra_link(0,ui,vi,wi)
		actorsA.update((u,v))

	# collaboration layer
	for u,v,w in read_edgelist(pathB,weighted=True):
		# Ensure w >= 0 to avoid log domain error
		if not isinstance(w, (int, float)):
			continue
		w = max(0.0, w)
		w_t = collab_normalize_log1p(w, p99_B) * sB
		ui,vi,wi = idmap.remap_edge(u,v,w_t)
		im.add_multilayer_intra_link(1,ui,vi,wi)
		actorsB.update((u,v))

	# interlayer coupling (fixed)
	for a in actorsA & actorsB:
		ai=idmap.get(a)
		im.add_multilayer_inter_link(0,ai,1,weight=omega)

	print(f"Built multilayer: |A|={len(actorsA):,} |B|={len(actorsB):,} |A∩B|={len(actorsA & actorsB):,}", flush=True)
	im.run()
	print("Infomap finished.", flush=True)
	return extract_overlap(im,idmap)

# ========= Main =========
if __name__=="__main__":

	# added new cmd argument: ...discipline (threshold - optional)

	if len(sys.argv) < 3:
		print(__doc__)
		sys.exit(1)

	mode = sys.argv[1]
	disc = sys.argv[2]
	out_base = sys.argv[3]
	outdir = os.path.join(out_base, f"infomap_runs_{disc}")
	os.makedirs(outdir, exist_ok=True)

	# Optional threshold argument (after all required ones)
	# e.g. python mln-infomap.py multilayer CS /N/... sim.edgelist collab.edgelist "0.1" 0.01
	threshold = None
	try:
		# Look for a numeric argument at the end
		if len(sys.argv) > 4 and sys.argv[-1].replace('.', '', 1).isdigit():
			threshold = float(sys.argv[-1])
	except Exception:
		threshold = None

	if threshold is not None:
		print(f"Applying weight threshold: {threshold}", flush=True)


	mode=sys.argv[1]
	disc=sys.argv[2]
	out_base=sys.argv[3]
	outdir=os.path.join(out_base,f"infomap_runs_{disc}")
	os.makedirs(outdir,exist_ok=True)

	if mode=="layerA":
		pathA=sys.argv[4]; 
		print(f"Running Layer A ({disc}) with sharpening (tau=0.02)",flush=True)
		idmap=IdMapper()
		coms=run_single_layer(pathA,idmap,weighted=True,sim_sharpen=True,threshold=threshold)
		save_coms_csv(os.path.join(outdir,"layerA_coms.csv"),coms)
		idmap.save(os.path.join(outdir,"id_mapping.csv"))

	elif mode=="layerB":
		pathB=sys.argv[4]
		print(f"Running Layer B ({disc}) with log1p normalization",flush=True)
		idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
		coms=run_single_layer(pathB,idmap,weighted=True,sim_sharpen=False,threshold=threshold)
		save_coms_csv(os.path.join(outdir,"layerB_coms.csv"),coms)
		idmap.save(os.path.join(outdir,"id_mapping.csv"))

	elif mode=="match":
		# ----- Parameters -----
		MIN_SIZE = 25          		# threshold for "large" communities
		OVERLAP_THRESHOLD = 0.5 	# threshold for high overlap

		import statistics

		print(f"Analyzing Layer A vs B ({disc})", flush=True)

		# --- Load community files ---
		def load_coms(path):
			coms = {}
			with open(path) as f:
				next(f)  # skip header
				for line in f:
					cid, nodes = line.strip().split(",", 1)
					coms[int(cid)] = set(nodes.split("|")) if nodes else set()
			return coms

		comsA = load_coms(os.path.join(outdir, "layerA_coms.csv"))
		comsB = load_coms(os.path.join(outdir, "layerB_coms.csv"))

		# --- (a) Basic stats ---
		def stats(coms):
			sizes = [len(v) for v in coms.values()]
			ncoms = len(sizes)
			avg = statistics.mean(sizes) if sizes else 0
			sd = statistics.stdev(sizes) if len(sizes) > 1 else 0
			large = sum(1 for s in sizes if s >= MIN_SIZE)
			return ncoms, avg, sd, large

		statsA = stats(comsA)
		statsB = stats(comsB)

		print(f"Layer A: {statsA[0]:,} communities "
			  f"(avg size={statsA[1]:.2f}, sd={statsA[2]:.2f}, >={MIN_SIZE}: {statsA[3]})")
		print(f"Layer B: {statsB[0]:,} communities "
			  f"(avg size={statsB[1]:.2f}, sd={statsB[2]:.2f}, >={MIN_SIZE}: {statsB[3]})",
			  flush=True)

		with open(os.path.join(outdir, "layer_stats.csv"), "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["layer", "n_communities", "avg_size", "sd_size", f"n_size>={MIN_SIZE}"])
			w.writerow(["A", *statsA])
			w.writerow(["B", *statsB])

		# --- (b) Overlap search ---
		results = []
		for cidB, nodesB in comsB.items():
			# Only check communities with a large enough size.
			if len(nodesB) < MIN_SIZE:
				continue
			best_cidA, best_overlap = None, 0.0
			for cidA, nodesA in comsA.items():
				inter = len(nodesA & nodesB)
				if inter == 0:
					continue
				jacc = inter / len(nodesA | nodesB)
				if jacc > best_overlap:
					best_overlap = jacc
					best_cidA = cidA
			if best_cidA is not None:
				results.append((cidB, best_cidA, best_overlap))

		# Save overlap summary
		with open(os.path.join(outdir, "community_overlap_summary.csv"), "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["layerB_comm", "best_layerA_comm", "overlap_pct"])
			for r in results:
				w.writerow([r[0], r[1], f"{r[2]:.4f}"])

		print(f"Analyzed {len(results)} large Layer-B communities", flush=True)

		# --- (c) Save high-overlap differences ---
		out_missing = os.path.join(outdir, "high_overlap_differences.csv")
		with open(out_missing, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["layerB_comm", "layerA_comm", "missing_in_A", "missing_in_B"])
			for cidB, cidA, overlap in results:
				if overlap >= OVERLAP_THRESHOLD:
					setA, setB = comsA[cidA], comsB[cidB]
					missing_in_A = sorted(setB - setA)
					missing_in_B = sorted(setA - setB)
					w.writerow([
						cidB, cidA,
						"|".join(map(str, missing_in_A)),
						"|".join(map(str, missing_in_B))
					])
		print("Saved high-overlap differences.", flush=True)


	elif mode=="multilayer":
		pathA=sys.argv[4]
		pathB=sys.argv[5]
		omega_str=sys.argv[6]
		omegas=[float(x) for x in omega_str.split(",")]
		print(f"Running multilayer ({disc}), omegas={omegas}",flush=True)
		idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
		for w in omegas:
			print(f"  omega={w}",flush=True)
			subdir=os.path.join(outdir,f"multilayer_omega_{w:g}")
			os.makedirs(subdir,exist_ok=True)
			comsM=run_multilayer(pathA,pathB,idmap,omega=w,threshold=threshold)
			save_coms_csv(os.path.join(subdir,"multilayer_coms.csv"),comsM)

	elif mode == "multilayer-match":
		print(f"Analyzing within-community A–B overlap for multilayer runs ({disc})", flush=True)

		HIGH_THRESHOLD = 0.85
		LOWER_THAN_PERFECT = 0.999999

		for subdir in [d for d in os.listdir(outdir) if d.startswith("multilayer_omega_")]:
			run_dir = os.path.join(outdir, subdir)
			print(f"  Processing {subdir}", flush=True)

			multi_path = os.path.join(run_dir, "multilayer_coms.csv")
			if not os.path.exists(multi_path):
				print(f"    Skipping {subdir}: multilayer_coms.csv not found.")
				continue

			# --- Load multilayer communities ---
			coms = {}
			with open(multi_path) as f:
				next(f)
				for line in f:
					cid, nodes = line.strip().split(",", 1)
					coms[int(cid)] = set(nodes.split("|")) if nodes else set()

			results = []
			detail_dir = os.path.join(run_dir, "intralayer_overlap_details")
			os.makedirs(detail_dir, exist_ok=True)

			# --- Analyze each community ---
			for cid, members in coms.items():
				layerA = set()
				layerB = set()
				for m in members:
					# detect which layer (0: or 1:) based on your naming convention
					if ":" in m:
						layer_id, nodeid = m.split(":", 1)
						if layer_id == "0":
							layerA.add(nodeid)
						elif layer_id == "1":
							layerB.add(nodeid)
					else:
						# fallback: skip malformed IDs
						continue

				if not layerA and not layerB:
					continue

				union = layerA | layerB
				inter = layerA & layerB
				jacc = len(inter) / len(union) if union else 0.0

				results.append((cid, len(layerA), len(layerB), len(inter), len(union), jacc))

				if HIGH_THRESHOLD < jacc < LOWER_THAN_PERFECT:
					uniqA = sorted(layerA - layerB)
					uniqB = sorted(layerB - layerA)
					with open(os.path.join(detail_dir, f"comm_{cid}.txt"), "w") as f:
						f.write(f"Community {cid}\n")
						f.write(f"Jaccard overlap = {jacc:.4f}\n")
						f.write(f"Layer A nodes: {len(layerA)}\n")
						f.write(f"Layer B nodes: {len(layerB)}\n")
						f.write(f"Shared authors: {len(inter)}\n\n")
						f.write(f"Unique to A ({len(uniqA)}):\n")
						f.write("\n".join(uniqA))
						f.write("\n\nUnique to B ({len(uniqB)}):\n")
						f.write("\n".join(uniqB))

			# --- Save summary CSV ---
			out_csv = os.path.join(run_dir, "intralayer_overlap_summary.csv")
			with open(out_csv, "w", newline="") as f:
				w = csv.writer(f)
				w.writerow(["community_id", "n_layerA", "n_layerB", "shared", "union", "jaccard"])
				w.writerows(results)

			print(f"  Saved {out_csv}", flush=True)

		print("All multilayer within-community overlaps complete.", flush=True)


	else:
		sys.exit(f"Unknown mode: {mode}")
