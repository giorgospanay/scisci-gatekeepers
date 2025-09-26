#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infomap overlapping community detection, discipline-aware.

Usage:
    python -u mln-infomap.py <mode> <discipline> <out_base> [layerA] [layerB] [omega-grid]

Modes:
    layerA     Run Infomap on Layer A (similarity)
    layerB     Run Infomap on Layer B (collaboration)
    match      Compare Layer A vs B
    multilayer Run multilayer (needs both layers)

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

import os, sys, csv, math
from collections import defaultdict
from infomap import Infomap
from cdlib import evaluation, NodeClustering

# ========= ID Mapper =========
class IdMapper:
    def __init__(self): self.forward={}; self.reverse=[]
    def get(self, orig):
        if orig not in self.forward:
            nid=len(self.reverse); self.forward[orig]=nid; self.reverse.append(orig)
        return self.forward[orig]
    def remap_edge(self,u,v,w): return self.get(u),self.get(v),w
    def save(self,path):
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["internal_id","original_id"])
            for nid,orig in enumerate(self.reverse): w.writerow([nid,orig])
    @classmethod
    def load(cls,path):
        mapper=cls()
        with open(path) as f:
            next(f)
            for line in f:
                nid,orig=line.strip().split(",",1); nid=int(nid)
                mapper.forward[orig]=nid
                while len(mapper.reverse)<=nid: mapper.reverse.append(None)
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
    com2nodes=defaultdict(set)
    for pid,modpath in im.get_multilevel_modules(states=True).items():
        cid=modpath[-1]; orig=idmap.reverse[int(pid)]; com2nodes[cid].add(orig)
    return dict(com2nodes)

def save_coms_csv(path,coms):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["community_id","nodes"])
        for cid,nodes in sorted(coms.items()):
            w.writerow([cid,"|".join(map(str,sorted(nodes)))])

def to_cdlib(coms): 
    return NodeClustering([list(nodes) for nodes in coms.values()],graph=None,overlap=True)

def pass_stats_over_weights(path, is_weighted=True, sample_every=1, cap_samples=5_000_000):
    import random
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
        idx=int(p*(len(sample)-1)); return sample[idx]
    return {"n":n,"sum":s,"max":wmax,"p95":q(0.95),"p99":q(0.99)}

# ========= Weight transforms =========
def sim_sharpen_temp(w, tau=0.02):
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
            w_t=sim_sharpen_temp(w,tau=0.02)
        else:
            w_t=w
        ui,vi,wi=idmap.remap_edge(u,v,w_t)
        im.add_link(ui,vi,wi)
    im.run(); return extract_overlap(im,idmap)

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

    actorsA=set(); actorsB=set()

    # similarity layer
    for u,v,w in read_edgelist(pathA,weighted=True,threshold=threshold):
        w_t=sim_sharpen_temp(w,tau=0.02)*sA
        ui,vi,wi=idmap.remap_edge(u,v,w_t)
        im.add_multilayer_intra_link(0,ui,vi,wi)
        actorsA.update((u,v))

    # collaboration layer
    for u,v,w in read_edgelist(pathB,weighted=True):
        w_t=collab_normalize_log1p(w,p99_B)*sB
        ui,vi,wi=idmap.remap_edge(u,v,w_t)
        im.add_multilayer_intra_link(1,ui,vi,wi)
        actorsB.update((u,v))

    # interlayer coupling
    for a in actorsA & actorsB:
        ai=idmap.get(a); im.add_multilayer_inter_link(0,ai,1,omega)

    im.run(); return extract_overlap(im,idmap)

# ========= Main =========
if __name__=="__main__":
    if len(sys.argv)<3: print(__doc__); sys.exit(1)

    mode=sys.argv[1]; disc=sys.argv[2]; out_base=sys.argv[3]
    outdir=os.path.join(out_base,f"infomap_runs_{disc}")
    os.makedirs(outdir,exist_ok=True)

    if mode=="layerA":
        pathA=sys.argv[4]; 
        print(f"Running Layer A ({disc}) with sharpening (τ=0.02)",flush=True)
        idmap=IdMapper(); 
        coms=run_single_layer(pathA,idmap,weighted=True,sim_sharpen=True)
        save_coms_csv(os.path.join(outdir,"layerA_coms.csv"),coms)
        idmap.save(os.path.join(outdir,"id_mapping.csv"))

    elif mode=="layerB":
        pathB=sys.argv[4]
        print(f"Running Layer B ({disc}) with log1p normalization",flush=True)
        idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
        coms=run_single_layer(pathB,idmap,weighted=True,sim_sharpen=False)
        save_coms_csv(os.path.join(outdir,"layerB_coms.csv"),coms)
        idmap.save(os.path.join(outdir,"id_mapping.csv"))

    elif mode=="match":
        print(f"Matching Layer A vs B ({disc})",flush=True)
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

    elif mode=="multilayer":
        pathA=sys.argv[4]; pathB=sys.argv[5]; omega_str=sys.argv[6]
        omegas=[float(x) for x in omega_str.split(",")]
        print(f"Running multilayer ({disc}), omegas={omegas}",flush=True)
        idmap=IdMapper.load(os.path.join(outdir,"id_mapping.csv"))
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
        scores=[]
        for w in omegas:
            print(f"  ω={w}",flush=True)
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
            w=csv.writer(f); 
            for row in scores: w.writerow(row)

    else:
        sys.exit(f"Unknown mode: {mode}")
