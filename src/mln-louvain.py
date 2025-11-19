#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Louvain (modularity) community detection, discipline-aware.

Usage:
	python -u mln-louvain.py <mode> <discipline> <out_base> [layerA] [layerB] [omega-grid]

Modes:
	layerA           Run Louvain on Layer A (similarity)
	layerB           Run Louvain on Layer B (collaboration)
	match            Compare Layer A vs B (Jaccard overlap, same as Infomap script)
	multilayer       Run multilayer Louvain (two-layer supra-graph with coupling ω)
	multilayer-match Compare multilayer runs vs A and B (same logic as Infomap script)

Arguments:
	discipline   Short name (e.g., "physics", "biology")
	out_base     Base output directory (e.g., /N/slate/.../obj)
	layerA       Path to similarity edgelist
	layerB       Path to collaboration edgelist
	omega-grid   Comma-separated (default: "0.1")

Examples:
	python -u mln-louvain.py layerA Physics /N/slate/.../obj filtered_author_similarity_layer_Physics.edgelist
	python -u mln-louvain.py multilayer Physics /N/slate/.../obj filtered_author_similarity_layer_Physics.edgelist filtered_collaboration_layer_Physics.edgelist "0.05,0.1,0.2"

Notes:
	* Requires python-igraph (`import igraph as ig`).
	* Graphs are stored fully in memory; for extremely large networks this
	  may still be heavy, though igraph is reasonably efficient.
"""

import os, sys, csv, math, random, gc
from collections import defaultdict

import igraph as ig  # Louvain via community_multilevel


# ========= ID Mapper =========
class IdMapper:
	"""
	Map arbitrary external IDs (strings) to compact 0..N-1 integers.
	This is shared across all modes/layers.
	"""
	def __init__(self):
		self.forward = {}   # original_id -> internal_id
		self.reverse = []   # internal_id -> original_id

	def get(self, orig):
		if orig not in self.forward:
			nid = len(self.reverse)
			self.forward[orig] = nid
			self.reverse.append(orig)
		return self.forward[orig]

	def remap_edge(self, u, v, w):
		return self.get(u), self.get(v), w

	def save(self, path):
		with open(path, "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["internal_id", "original_id"])
			for nid, orig in enumerate(self.reverse):
				w.writerow([nid, orig])

	@classmethod
	def load(cls, path):
		mapper = cls()
		with open(path) as f:
			next(f)
			for line in f:
				nid, orig = line.strip().split(",", 1)
				nid = int(nid)
				mapper.forward[orig] = nid
				while len(mapper.reverse) <= nid:
					mapper.reverse.append(None)
				mapper.reverse[nid] = orig
		return mapper


# ========= Utils =========
def read_edgelist(path, weighted=True, report_every=1_000_000, threshold=None):
	count = 0
	with open(path) as f:
		for line in f:
			if not line.strip() or line.startswith("#"):
				continue
			parts = line.split()

			if weighted and len(parts) >= 3:
				u, v, w = parts[0], parts[1], float(parts[2])
				if threshold is not None and w < threshold:
					continue
			else:
				u, v, w = parts[0], parts[1], 1.0

			yield u, v, w
			count += 1
			if count % report_every == 0:
				print(f"  ...read {count:,} edges from {os.path.basename(path)}", flush=True)
	print(f"  Finished reading {count:,} edges from {os.path.basename(path)}", flush=True)


def save_coms_csv(path, coms):
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["community_id", "nodes"])
		for cid, nodes in sorted(coms.items()):
			w.writerow([cid, "|".join(map(str, sorted(nodes)))])


def pass_stats_over_weights(path, is_weighted=True, sample_every=1, cap_samples=5_000_000):
	n = 0
	s = 0.0
	wmax = 0.0
	sample = []
	step = max(1, sample_every)

	for i, line in enumerate(open(path)):
		if not line.strip() or line.startswith("#"):
			continue
		parts = line.split()
		if is_weighted and len(parts) >= 3:
			w = float(parts[2])
		else:
			w = 1.0
		n += 1
		s += w
		if w > wmax:
			wmax = w
		if (i % step) == 0:
			sample.append(w)
			if len(sample) > cap_samples:
				# downsample
				sample = random.sample(sample, len(sample) // 2)
	sample.sort()

	def q(p):
		if not sample:
			return 0.0
		idx = int(p * (len(sample) - 1))
		return sample[idx]

	return {"n": n, "sum": s, "max": wmax, "p95": q(0.95), "p99": q(0.99)}


# ========= Weight transforms (same as Infomap script) =========
def sim_sharpen_temp(w, tau=0.02):
	w = max(0.0, min(1.0, w))  # clamp within [0,1]
	return math.exp(-(1.0 - w) / max(tau, 1e-6))


def sim_sharpen_gamma(w, alpha=3.0):
	return 1.0 - (1.0 - w) ** alpha


def collab_normalize_log1p(w, p99):
	return min(1.0, math.log1p(w) / math.log1p(max(p99, 1.0)))


# ========= Louvain runners =========
def run_single_layer_louvain(path, idmap, weighted=True,
							 threshold=None, sim_sharpen=False):
	"""
	Build a single-layer igraph.Graph, run Louvain, and return communities
	as {community_id: set(original_ids)}.
	"""
	print(f"Building igraph graph from {path} ...", flush=True)
	edges = []
	weights = []

	for u, v, w in read_edgelist(path, weighted=weighted, threshold=threshold):
		if sim_sharpen:
			w = max(0.0, min(1.0, w))
			w_t = sim_sharpen_temp(w, tau=0.02)
		else:
			w_t = w

		ui, vi, _ = idmap.remap_edge(u, v, w_t)
		edges.append((ui, vi))
		weights.append(w_t)

	n_nodes = len(idmap.reverse)
	print(f"  Creating graph with {n_nodes:,} nodes and {len(edges):,} edges", flush=True)
	g = ig.Graph(n=n_nodes, edges=edges, directed=False)
	g.es["weight"] = weights

	print("Running Louvain (community_multilevel)...", flush=True)
	# igraph's Louvain implementation
	part = g.community_multilevel(weights=g.es["weight"])
	membership = part.membership

	print("Extracting communities...", flush=True)
	com2nodes = defaultdict(set)
	for internal_id, cid in enumerate(membership):
		if internal_id >= len(idmap.reverse):
			continue
		orig = idmap.reverse[internal_id]
		com2nodes[cid].add(orig)

	print(f"Extracted {len(com2nodes)} communities.", flush=True)

	# Free some memory
	del g, edges, weights, membership, part
	gc.collect()

	return dict(com2nodes)


def run_multilayer_louvain(pathA, pathB, idmap, omega=0.1,
						   threshold=None, inter_keep_frac=1.0,
						   rescale_interlayer=False):
	"""
	Construct a two-layer supra-graph:

		layer 0: similarity (A) with sharpened weights
		layer 1: collaboration (B) with log1p-normalized weights

	Nodes in the supra-graph are (layer, actor), encoded as:
		node_id = 2 * actor_internal_id + layer_id (0 or 1)

	Interlayer edges connect (0, actor) <-> (1, actor) for a subset of shared actors
	with weight ω (optionally rescaled if inter_keep_frac < 1).

	Then run Louvain on the supra-graph and extract communities as sets of
	"layer:original_id" strings (e.g., "0:123456", "1:987654").
	"""
	# Stats for collab
	print("Pass over weights for A and B...", flush=True)
	statsA = pass_stats_over_weights(pathA, is_weighted=True)
	statsB = pass_stats_over_weights(pathB, is_weighted=True)
	p99_B = statsB["p99"] if statsB["p99"] > 0 else 1.0

	sA = 1.0
	sB = (statsA["sum"] if statsA["sum"] > 0 else 1.0) / (statsB["sum"] if statsB["sum"] > 0 else 1.0)

	# For supra-graph
	edges = []
	weights = []

	actorsA = set()  # internal actor IDs present in similarity layer
	actorsB = set()  # same for collaboration

	keptA = keptB = 0

	# similarity layer (layer 0)
	if threshold is not None:
		print(f"Filtering similarity edges below {threshold}", flush=True)
	for u, v, w in read_edgelist(pathA, weighted=True, threshold=threshold):
		if not isinstance(w, (int, float)):
			continue
		w = max(0.0, min(1.0, w))
		w_t = sim_sharpen_temp(w, tau=0.02) * sA

		ui, vi, _ = idmap.remap_edge(u, v, w_t)
		# supra nodes for layer 0
		nu = 2 * ui
		nv = 2 * vi

		edges.append((nu, nv))
		weights.append(w_t)

		actorsA.update((ui, vi))
		keptA += 1

	# collaboration layer (layer 1)
	if threshold is not None:
		print(f"Filtering collaboration edges below {threshold}", flush=True)
	for u, v, w in read_edgelist(pathB, weighted=True, threshold=threshold):
		if not isinstance(w, (int, float)):
			continue
		w = max(0.0, w)
		w_t = collab_normalize_log1p(w, p99_B) * sB

		ui, vi, _ = idmap.remap_edge(u, v, w_t)
		# supra nodes for layer 1
		nu = 2 * ui + 1
		nv = 2 * vi + 1

		edges.append((nu, nv))
		weights.append(w_t)

		actorsB.update((ui, vi))
		keptB += 1

	shared = actorsA & actorsB
	print(f"Built intra-layers: |A edges|={keptA:,}, |B edges|={keptB:,}, "
		  f"|A nodes|={len(actorsA):,}, |B nodes|={len(actorsB):,}, "
		  f"|A∩B|={len(shared):,}", flush=True)

	# --- sparse interlayer coupling ---
	keep_frac = float(inter_keep_frac) if inter_keep_frac is not None else 1.0
	keep_frac = max(0.0, min(1.0, keep_frac))

	if keep_frac <= 0.0 or not shared:
		print("Interlayer coupling disabled or no shared actors.", flush=True)
	else:
		n_shared = len(shared)
		n_keep = int(n_shared * keep_frac) if keep_frac < 1.0 else n_shared
		if n_keep == 0 and n_shared > 0:
			n_keep = 1

		print(f"Creating interlayer links: sampling {n_keep:,} of {n_shared:,} "
			  f"shared actors (keep_frac={keep_frac:.4f}, rescale={rescale_interlayer})", flush=True)

		if keep_frac < 1.0:
			sample = random.sample(list(shared), n_keep)
		else:
			sample = shared

		# rescale ω if requested
		ω = omega / keep_frac if (rescale_interlayer and keep_frac > 0) else omega

		added = 0
		for ai in sample:
			nu = 2 * ai       # layer 0 node
			nv = 2 * ai + 1   # layer 1 node
			edges.append((nu, nv))
			weights.append(ω)
			added += 1

		print(f"Added {added:,} interlayer links with weight {ω}", flush=True)

	# Build supra-graph
	if edges:
		max_node_id = max(max(u, v) for u, v in edges)
		n_nodes = max_node_id + 1
	else:
		n_nodes = 0

	print(f"Creating supra-graph with {n_nodes:,} nodes and {len(edges):,} edges", flush=True)
	g = ig.Graph(n=n_nodes, edges=edges, directed=False)
	g.es["weight"] = weights

	print("Running Louvain on multilayer supra-graph...", flush=True)
	part = g.community_multilevel(weights=g.es["weight"])
	membership = part.membership

	# Extract communities as layer-tagged original IDs
	print("Extracting multilayer communities...", flush=True)
	com2nodes = defaultdict(set)

	for supra_id, cid in enumerate(membership):
		actor_int = supra_id // 2
		layer_id = supra_id % 2  # 0 or 1

		if actor_int >= len(idmap.reverse):
			continue

		orig = idmap.reverse[actor_int]
		label = f"{layer_id}:{orig}"
		com2nodes[cid].add(label)

	print(f"Extracted {len(com2nodes)} communities from multilayer run.", flush=True)

	del g, edges, weights, membership, part, actorsA, actorsB, shared
	gc.collect()

	return dict(com2nodes)


# ========= Main =========
if __name__ == "__main__":

	if len(sys.argv) < 3:
		print(__doc__)
		sys.exit(1)

	mode = sys.argv[1]
	disc = sys.argv[2]
	out_base = sys.argv[3]
	outdir = os.path.join(out_base, f"louvain_runs_{disc}")
	os.makedirs(outdir, exist_ok=True)

	# Also load infomap mappings to reuse
	infomap_outdir = os.path.join(out_base, f"infomap_runs_{disc}")
	infomap_idmap_path = os.path.join(infomap_outdir, "id_mapping.csv")

	def parse_float(s):
		try:
			return float(s)
		except Exception:
			return None

	# ---------- layerA ----------
	if mode == "layerA":
		if len(sys.argv) < 5:
			sys.exit("Usage: mln-louvain.py layerA <discipline> <out_base> <layerA_path> [threshold]")

		pathA = sys.argv[4]
		threshold = parse_float(sys.argv[5]) if len(sys.argv) >= 6 else None

		if threshold is not None:
			print(f"Applying weight threshold: {threshold}", flush=True)

		print(f"Running Louvain on Layer A ({disc}) with sharpening (τ=0.02)", flush=True)
			
		# Reuse Infomap mapping
		if not os.path.exists(infomap_idmap_path):
			sys.exit(f"Infomap id_mapping.csv not found at {infomap_idmap_path}. "
					 f"Run the Infomap layerA/layerB pipeline first.")
		idmap = IdMapper.load(infomap_idmap_path)

		coms = run_single_layer_louvain(pathA, idmap, weighted=True,
										sim_sharpen=True, threshold=threshold)
		save_coms_csv(os.path.join(outdir, "layerA_coms.csv"), coms)
		idmap.save(os.path.join(outdir, "id_mapping.csv"))

	# ---------- layerB ----------
	elif mode == "layerB":
		if len(sys.argv) < 5:
			sys.exit("Usage: mln-louvain.py layerB <discipline> <out_base> <layerB_path> [threshold]")

		pathB = sys.argv[4]
		threshold = parse_float(sys.argv[5]) if len(sys.argv) >= 6 else None

		if threshold is not None:
			print(f"Applying weight threshold: {threshold}", flush=True)

		print(f"Running Louvain on Layer B ({disc}) with log1p normalization", flush=True)
			
		# Reuse InfoMap id mapper
		idmap = IdMapper.load(infomap_idmap_path)

		coms = run_single_layer_louvain(pathB, idmap, weighted=True,
										sim_sharpen=False, threshold=threshold)
		save_coms_csv(os.path.join(outdir, "layerB_coms.csv"), coms)
		idmap.save(os.path.join(outdir, "id_mapping.csv"))

	# ---------- match ----------
	elif mode == "match":
		MIN_SIZE = 25           # threshold for "large" communities
		OVERLAP_THRESHOLD = 0.5 # threshold for high overlap

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

		with open(os.path.join(outdir, "community_overlap_summary.csv"), "w", newline=True) as f:
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

	# ---------- multilayer ----------
	elif mode == "multilayer":
		if len(sys.argv) < 7:
			sys.exit(
				"Usage: mln-louvain.py multilayer <discipline> <out_base> "
				"<layerA_path> <layerB_path> <omega_grid> [threshold] [keep_frac]"
			)

		pathA = sys.argv[4]
		pathB = sys.argv[5]
		omega_str = sys.argv[6]
		omegas = [float(x) for x in omega_str.split(",")]

		threshold = parse_float(sys.argv[7]) if len(sys.argv) >= 8 else None
		keep_frac = parse_float(sys.argv[8]) if len(sys.argv) >= 9 else 1.0
		if keep_frac is None:
			keep_frac = 1.0

		if threshold is not None:
			print(f"Applying weight threshold: {threshold}", flush=True)
		print(f"Running multilayer Louvain ({disc}), omegas={omegas}, keep_frac={keep_frac}", flush=True)

		# Reuse InfoMap id mapper
		idmap = IdMapper.load(infomap_idmap_path)
		
		for w in omegas:
			print(f"  ω={w}", flush=True)
			subdir = os.path.join(outdir, f"multilayer_omega_{w:g}")
			os.makedirs(subdir, exist_ok=True)

			comsM = run_multilayer_louvain(
				pathA, pathB, idmap,
				omega=w,
				threshold=threshold,
				inter_keep_frac=keep_frac,
				rescale_interlayer=False  # set True if you want ω / keep_frac
			)

			save_coms_csv(os.path.join(subdir, "multilayer_coms.csv"), comsM)

			# extra safety: free communities and force GC between ω-runs
			del comsM
			gc.collect()

	# ---------- multilayer-match ----------
	elif mode == "multilayer-match":
		print(f"Analyzing within-community A–B overlap for multilayer Louvain runs ({disc})", flush=True)

		HIGH_THRESHOLD = 0.85
		LOWER_THAN_PERFECT = 0.999999

		for subdir in [d for d in os.listdir(outdir) if d.startswith("multilayer_omega_")]:
			run_dir = os.path.join(outdir, subdir)
			print(f"  Processing {subdir}", flush=True)

			multi_path = os.path.join(run_dir, "multilayer_coms.csv")
			if not os.path.exists(multi_path):
				print(f"    Skipping {subdir}: multilayer_coms.csv not found.")
				continue

			coms = {}
			with open(multi_path) as f:
				next(f)
				for line in f:
					cid, nodes = line.strip().split(",", 1)
					coms[int(cid)] = set(nodes.split("|")) if nodes else set()

			results = []
			detail_dir = os.path.join(run_dir, "intralayer_overlap_details")
			os.makedirs(detail_dir, exist_ok=True)

			for cid, members in coms.items():
				layerA = set()
				layerB = set()
				for m in members:
					if ":" in m:
						layer_id, nodeid = m.split(":", 1)
						if layer_id == "0":
							layerA.add(nodeid)
						elif layer_id == "1":
							layerB.add(nodeid)
					else:
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

			out_csv = os.path.join(run_dir, "intralayer_overlap_summary.csv")
			with open(out_csv, "w", newline="") as f:
				w = csv.writer(f)
				w.writerow(["community_id", "n_layerA", "n_layerB", "shared", "union", "jaccard"])
				w.writerows(results)

			print(f"  Saved {out_csv}", flush=True)

		print("All multilayer within-community overlaps complete.", flush=True)

	else:
		sys.exit(f"Unknown mode: {mode}")
