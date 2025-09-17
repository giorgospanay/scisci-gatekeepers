#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run overlapping community detection using the SAME algorithm (Infomap) for:
  1) Single-layer (each layer separately, overlapping)
  2) Multilayer (both layers with inter-layer coupling ω), overlapping

Outputs CSVs to your workspace tree.

Requires:
  pip install infomap

Default file paths are your provided ones, but you can override via CLI.
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from infomap import Infomap

# === Your defaults (change here or use CLI flags) ===
RAW_WORKSPACE_PATH = "/N/scratch/gpanayio/openalex-pre"
OUT_WORKSPACE_PATH = "/N/slate/gpanayio/scisci-gatekeepers"
OUT_SCRATCH_PATH = "/N/scratch/gpanayio"

OBJ_PATH = f"{OUT_WORKSPACE_PATH}/obj"
EMBEDDING_CHUNK_DIR = f"{OUT_SCRATCH_PATH}/embeddings_filtered"
METADATA_PATH = f"{RAW_WORKSPACE_PATH}/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"

FILTERED_IDS_FILE = os.path.join(OBJ_PATH, "filtered_paper_ids.txt")

# Layer files you mentioned
COLLAB_EDGE_FILE_DEFAULT = os.path.join(OBJ_PATH, "collaboration_layer.edgelist")
AUTHOR_SIM_EDGE_FILE_DEFAULT = os.path.join(OBJ_PATH, "author_similarity_layer.edgelist")


# -----------------
# Utility functions
# -----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def read_edgelist(path, weighted=True):
    """
    Streams an edgelist. Lines are "u v [w]".
    Returns (u:int, v:int, w:float). If unweighted, w=1.0.
    Ignores blank lines and lines starting with '#'.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if weighted and len(parts) >= 3:
                u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            else:
                u, v, w = int(parts[0]), int(parts[1]), 1.0
            yield u, v, w

def extract_overlap(im) -> dict:
    """
    Returns overlapping communities as: dict[int community_id] -> set[int node_id]
    Uses states=True to include overlaps.
    """
    com2nodes = defaultdict(set)
    for phys_id, module_path in im.get_multilevel_modules(states=True).items():
        cid = module_path[-1]  # finest level
        com2nodes[cid].add(phys_id)
    return dict(com2nodes)

def save_coms_csv(path, coms: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["community_id", "nodes"])  # nodes as pipe-delimited
        for cid, nodes in sorted(coms.items()):
            w.writerow([cid, "|".join(map(str, sorted(nodes)))])

def jaccard(A, B):
    return len(A & B) / len(A | B) if (A or B) else 0.0

def match_by_jaccard(A_coms: dict, B_coms: dict, tau: float = 0.2):
    """
    Greedy A->B matching by Jaccard ≥ tau.
    Returns list of (A_id, B_id, jaccard, intersection_size, union_size)
    """
    matches = []
    usedB = set()
    # score each A against best B
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
            matches.append((a_id, best_b, best_j, inter, uni))
            usedB.add(best_b)
    return matches

def write_matches_csv(path, pairs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A_id", "B_id", "Jaccard", "Intersection_size", "Union_size"])
        for a, b, j, i, u in pairs:
            w.writerow([a, b, f"{j:.6f}", i, u])

def rank_bridge_actors(multilayer_coms: dict, topk: int = 100):
    """
    Identify actors that sit in overlaps of multilayer modules.
    Score = number of multilayer modules the actor appears in.
    Returns dict[community_id] -> list[actor ids ranked]
    """
    # build actor -> membership count
    actor_to_count = defaultdict(int)
    for cid, nodes in multilayer_coms.items():
        for v in nodes:
            actor_to_count[v] += 1

    results = {}
    for cid, nodes in multilayer_coms.items():
        scored = [(v, actor_to_count[v]) for v in nodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        results[cid] = [v for v, c in scored[:topk] if c > 1]  # only true overlaps
    return results

def save_bridge_csv(path, bridge_map: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["community_id", "bridge_actors"])  # bridge_actors pipe-delimited
        for cid, actors in sorted(bridge_map.items()):
            if actors:
                w.writerow([cid, "|".join(map(str, actors))])


# -----------------
# Infomap runners
# -----------------
def run_single_layer(path, weighted=True, undirected=True, num_trials=50, seed=42):
    im = Infomap(
        silent=True,
        num_trials=num_trials,
        two_level=True,
        undirected=undirected,
        seed=seed,
    )
    for u, v, w in read_edgelist(path, weighted=weighted):
        im.add_link(u, v, w)
    im.run()
    return extract_overlap(im)

def run_multilayer(pathA, pathB, omega=0.1, num_trials=50, seed=42,
                   weightedA=True, weightedB=False):
    """
    Build a 2-layer multiplex:
      - layer 0 = A (weightedA)
      - layer 1 = B (weightedB)
      - inter-layer coupling between (node,0) and (node,1) with weight omega
    """
    im = Infomap(
        silent=True,
        num_trials=num_trials,
        two_level=True,
        directed=True,  # required by Infomap's multilayer API
        seed=seed,
    )

    actorsA, actorsB = set(), set()

    for u, v, w in read_edgelist(pathA, weighted=weightedA):
        im.add_multilayer_intra_link(0, u, v, w)
        actorsA.update((u, v))

    for u, v, w in read_edgelist(pathB, weighted=weightedB):
        im.add_multilayer_intra_link(1, u, v, w)
        actorsB.update((u, v))

    # inter-layer links for shared actors
    for a in (actorsA & actorsB):
        im.add_multilayer_inter_link(0, a, 1, omega)

    im.run()
    return extract_overlap(im)


# -----------------
# CLI
# -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Overlapping community detection with Infomap (single-layer and multilayer).")
    p.add_argument("--layerA", default=AUTHOR_SIM_EDGE_FILE_DEFAULT,
                   help="Path to Layer A edgelist (default: author_similarity_layer.edgelist)")
    p.add_argument("--layerB", default=COLLAB_EDGE_FILE_DEFAULT,
                   help="Path to Layer B edgelist (default: collaboration_layer.edgelist)")
    p.add_argument("--weightedA", action="store_true", default=True,
                   help="Treat Layer A as weighted (default: True)")
    p.add_argument("--unweightedA", action="store_true", default=False,
                   help="Force Layer A unweighted (overrides --weightedA)")
    p.add_argument("--weightedB", action="store_true", default=False,
                   help="Treat Layer B as weighted (default: False)")
    p.add_argument("--omega", type=float, default=0.1, help="Inter-layer coupling weight (default: 0.1)")
    p.add_argument("--omega-grid", type=str, default="",
                   help="Comma-separated list of omegas to sweep (e.g., 0.05,0.1,0.2). If set, overrides --omega and runs all.")
    p.add_argument("--tau", type=float, default=0.2, help="Jaccard threshold for matching A↔B (default: 0.2)")
    p.add_argument("--num-trials", type=int, default=50, help="Infomap num_trials (default: 50)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--outdir", default=os.path.join(OBJ_PATH, "infomap_runs"),
                   help="Output directory (default: <obj>/infomap_runs)")
    return p.parse_args()

def main():
    args = parse_args()

    weightedA = args.weightedA and not args.unweightedA
    weightedB = args.weightedB

    ensure_dir(args.outdir)

    # Run single-layer A
    print(f"[1/5] Running single-layer on A: {args.layerA} (weighted={weightedA})")
    comsA = run_single_layer(args.layerA, weighted=weightedA,
                             undirected=True, num_trials=args.num_trials, seed=args.seed)
    outA = os.path.join(args.outdir, "layerA_coms.csv")
    save_coms_csv(outA, comsA)
    print(f"      -> {outA} ({len(comsA)} communities)")

    # Run single-layer B
    print(f"[2/5] Running single-layer on B: {args.layerB} (weighted={weightedB})")
    comsB = run_single_layer(args.layerB, weighted=weightedB,
                             undirected=True, num_trials=args.num_trials, seed=args.seed)
    outB = os.path.join(args.outdir, "layerB_coms.csv")
    save_coms_csv(outB, comsB)
    print(f"      -> {outB} ({len(comsB)} communities)")

    # Match A <-> B
    print(f"[3/5] Matching A<->B with Jaccard tau={args.tau}")
    pairs = match_by_jaccard(comsA, comsB, tau=args.tau)
    outPairs = os.path.join(args.outdir, "matched_pairs.csv")
    write_matches_csv(outPairs, pairs)
    print(f"      -> {outPairs} ({len(pairs)} matched pairs)")

    # Omega sweep
    omegas = [args.omega]
    if args.omega_grid.strip():
        omegas = [float(x) for x in args.omega_grid.split(",") if x.strip()]

    print(f"[4/5] Multilayer runs (omegas: {omegas})")
    for w in omegas:
        subdir = ensure_dir(os.path.join(args.outdir, f"multilayer_omega_{w:g}"))
        print(f"      - ω={w}: building multilayer and running Infomap...")
        comsM = run_multilayer(args.layerA, args.layerB, omega=w,
                               num_trials=args.num_trials, seed=args.seed,
                               weightedA=weightedA, weightedB=weightedB)

        outM = os.path.join(subdir, "multilayer_coms.csv")
        save_coms_csv(outM, comsM)
        print(f"        -> {outM} ({len(comsM)} communities)")

        # Bridge actors
        bridges = rank_bridge_actors(comsM, topk=200)
        outBridge = os.path.join(subdir, "bridge_actors.csv")
        save_bridge_csv(outBridge, bridges)
        print(f"        -> {outBridge}")

    print(f"[5/5] Done. Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
