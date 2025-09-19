#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infomap overlapping community detection with explicit sys.argv parsing.

Usage:
    python -u mln-infomap.py mode outdir [layerA] [layerB] [omega-grid]

Modes:
    layerA     Run Infomap on Layer A only
    layerB     Run Infomap on Layer B only
    match      Compare Layer A vs B (requires both coms files)
    multilayer Run multilayer analysis (requires layerA, layerB, omega-grid)

Example:
    python -u mln-infomap.py layerA obj/infomap_runs_filtered_eval obj/layerA.edgelist
    python -u mln-infomap.py layerB obj/infomap_runs_filtered_eval obj/layerB.edgelist
    python -u mln-infomap.py match obj/infomap_runs_filtered_eval
    python -u mln-infomap.py multilayer obj/infomap_runs_filtered_eval obj/layerA.edgelist obj/layerB.edgelist "0.05,0.1"
"""

import os, sys, csv
from collections import defaultdict
from infomap import Infomap
from cdlib import evaluation, NodeClustering

# ========= ID Mapper =========
class IdMapper:
    def __init__(self):
        self.forward = {}
        self.reverse = []

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
            w = csv.writer(f); w.writerow(["internal_id","original_id"])
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
def read_edgelist(path, weighted=True, report_every=1_000_000):
    count=0
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts=line.split()
            if weighted and len(parts)>=3:
                u,v,w=parts[0],parts[1],float(parts[2])
            else:
                u,v,w=parts[0],parts[1],1.0
            yield u,v,w
            count+=1
            if count % report_every==0:
                print(f"  ...read {count:,} edges from {os.path.basename(path)}", flush=True)
    print(f"  Finished reading {count:,} edges from {os.path.basename(path)}", flush=True)

def extract_overlap(im,idmap):
    com2nodes=defaultdict(set)
    for pid,modpath in im.get_multilevel_modules(states=True).items():
        cid=modpath[-1]
        orig=idmap.reverse[int(pid)]
        com2nodes[cid].add(orig)
    return dict(com2nodes)

def save_coms_csv(path, coms):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["community_id","nodes"])
        for cid,nodes in sorted(coms.items()):
            w.writerow([cid,"|".join(map(str,sorted(nodes)))])

def to_cdlib(coms):
    return NodeClustering([list(nodes) for nodes in coms.values()], graph=None, overlap=True)

def invert(coms):
    node2coms=defaultdict(set)
    for cid,nodes in coms.items():
        for n in nodes: node2coms[n].add(cid)
    return dict(node2coms)

# ========= Runners =========
def run_single_layer(path,idmap,weighted=True,num_trials=25,seed=42):
    im=Infomap(silent=True,num_trials=num_trials,two_level=True,seed=seed)
    for u,v,w in read_edgelist(path,weighted=weighted):
        ui,vi,wi=idmap.remap_edge(u,v,w); im.add_link(ui,vi,wi)
    im.run(); return extract_overlap(im,idmap)

def run_multilayer(pathA,pathB,idmap,omega=0.1,num_trials=25,seed=42,weightedA=True,weightedB=False):
    im=Infomap(silent=True,num_trials=num_trials,two_level=True,directed=True,seed=seed)
    actorsA,actorsB=set(),set()
    for u,v,w in read_edgelist(pathA,weighted=weightedA):
        ui,vi,wi=idmap.remap_edge(u,v,w)
        im.add_multilayer_intra_link(0,ui,vi,wi); actorsA.update((u,v))
    for u,v,w in read_edgelist(pathB,weighted=weightedB):
        ui,vi,wi=idmap.remap_edge(u,v,w)
        im.add_multilayer_intra_link(1,ui,vi,wi); actorsB.update((u,v))
    for a in actorsA & actorsB:
        ai=idmap.get(a); im.add_multilayer_inter_link(0,ai,1,omega)
    im.run(); return extract_overlap(im,idmap)

# ========= Main =========
if __name__=="__main__":
    if len(sys.argv)<3:
        print(__doc__); sys.exit(1)

    mode=sys.argv[1]
    outdir=sys.argv[2]
    os.makedirs(outdir,exist_ok=True)

    if mode=="layerA":
        if len(sys.argv)<4: sys.exit("Need path to LayerA edgelist")
        pathA=sys.argv[3]
        print("Running Layer A...",flush=True)
        idmap=IdMapper()
        coms=run_single_layer(pathA,idmap,weighted=True)
        save_coms_csv(os.path.join(outdir,"layerA_coms.csv"),coms)
        idmap.save(os.path.join(outdir,"id_mapping.csv"))
        print("Layer A done.",flush=True)

    elif mode=="layerB":
        if len(sys.argv)<4: sys.exit("Need path to LayerB edgelist")
        pathB=sys.argv[3]
        print("Running Layer B...",flush=True)
        idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
        coms=run_single_layer(pathB,idmap,weighted=False)
        save_coms_csv(os.path.join(outdir,"layerB_coms.csv"),coms)
        idmap.save(os.path.join(outdir,"id_mapping.csv"))
        print("Layer B done.",flush=True)

    elif mode=="match":
        print("Matching A <-> B...",flush=True)
        # read communities
        def load_coms(path):
            coms={}
            with open(path) as f:
                next(f)
                for line in f:
                    cid,nodes=line.strip().split(",",1)
                    coms[int(cid)]=set(nodes.split("|")) if nodes else set()
            return coms
        comsA=load_coms(os.path.join(outdir,"layerA_coms.csv"))
        comsB=load_coms(os.path.join(outdir,"layerB_coms.csv"))
        A_cd,B_cd=to_cdlib(comsA),to_cdlib(comsB)
        onmi=evaluation.overlapping_normalized_mutual_information_MGH(A_cd,B_cd).score
        omega=evaluation.omega_index(A_cd,B_cd).score
        with open(os.path.join(outdir,"scores.csv"),"w",newline="") as f:
            w=csv.writer(f); w.writerow(["comparison","ONMI","Omega"])
            w.writerow(["A_vs_B",onmi,omega])
        print("Match done.",flush=True)

    elif mode=="multilayer":
        if len(sys.argv)<6: sys.exit("Need LayerA, LayerB paths and omega-grid string")
        pathA=sys.argv[3]; pathB=sys.argv[4]; omega_str=sys.argv[5]
        omegas=[float(x) for x in omega_str.split(",")]
        idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
        # load comsA/B
        def load_coms(path):
            coms={}; 
            with open(path) as f:
                next(f)
                for line in f:
                    cid,nodes=line.strip().split(",",1)
                    coms[int(cid)]=set(nodes.split("|")) if nodes else set()
            return coms
        comsA=load_coms(os.path.join(outdir,"layerA_coms.csv"))
        comsB=load_coms(os.path.join(outdir,"layerB_coms.csv"))
        scores=[]
        for w in omegas:
            print(f"Running multilayer ω={w}...",flush=True)
            subdir=os.path.join(outdir,f"multilayer_omega_{w:g}")
            os.makedirs(subdir,exist_ok=True)
            comsM=run_multilayer(pathA,pathB,idmap,omega=w)
            save_coms_csv(os.path.join(subdir,"multilayer_coms.csv"),comsM)
            A_cd,B_cd,M_cd=to_cdlib(comsA),to_cdlib(comsB),to_cdlib(comsM)
            onmi_AM=evaluation.overlapping_normalized_mutual_information_MGH(A_cd,M_cd).score
            omega_AM=evaluation.omega_index(A_cd,M_cd).score
            onmi_BM=evaluation.overlapping_normalized_mutual_information_MGH(B_cd,M_cd).score
            omega_BM=evaluation.omega_index(B_cd,M_cd).score
            scores.append((f"A_vs_M({w})",onmi_AM,omega_AM))
            scores.append((f"B_vs_M({w})",onmi_BM,omega_BM))
        with open(os.path.join(outdir,"scores.csv"),"a",newline="") as f:
            w=csv.writer(f)
            for row in scores: w.writerow(row)
        print("Multilayer done.",flush=True)

    else:
        sys.exit(f"Unknown mode: {mode}")
