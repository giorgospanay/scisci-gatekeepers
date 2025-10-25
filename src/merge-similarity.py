# -*- coding: utf-8 -*-
"""
Merge temporary author–author edge chunks into one averaged edgelist.

Usage:
    python merge_author_chunks_avg.py <discipline>

Example:
    python merge_author_chunks_avg.py Math
"""

import os, sys, subprocess

# ===== Input argument =====
if len(sys.argv) < 2:
    print("Usage: python merge_author_chunks_avg.py <discipline>")
    sys.exit(1)

disc = sys.argv[1]
base = "/N/slate/gpanayio/scisci-gatekeepers/obj"
base_scratch = "/N/scratch/gpanayio/scisci-gatekeepers/obj"

tmp_dir = os.path.join(base_scratch, f"tmp_author_edges_{disc}")
out_path = os.path.join(base, f"filtered_author_similarity_layer_{disc}.edgelist")

print(f"=== Merging author edge chunks for {disc} ===")
print(f"Input directory: {tmp_dir}")
print(f"Output file:     {out_path}")
print("-" * 80)

# ===== Verify chunks =====
if not os.path.isdir(tmp_dir):
    sys.exit(f"Directory not found: {tmp_dir}")

chunk_files = sorted(
    os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("chunk_")
)
if not chunk_files:
    sys.exit("No chunk files found to merge.")

print(f"Found {len(chunk_files)} chunk files to merge.")

# ===== External merge command (average aggregation) =====
cat_cmd = "cat " + " ".join(chunk_files)
sort_cmd = f"sort -S 20G -k1,1 -k2,2"
awk_cmd = (
    "awk 'BEGIN{OFS=\" \"} "
    "{key=$1\" \"$2; if(key==prev){sum+=$3; n++}"
    "else{if(NR>1)print prev_a,prev_b,sum/n;"
    "split(key,a,\" \"); prev_a=a[1]; prev_b=a[2]; sum=$3; n=1;} prev=key;} "
    "END{if(NR>0)print prev_a,prev_b,sum/n;}'"
)

cmd = f"{cat_cmd} | {sort_cmd} | {awk_cmd} > {out_path}"

print(f"Running merge command:\n  {cmd}\n")

ret = subprocess.call(cmd, shell=True)
if ret != 0:
    sys.exit(f"Merge failed with exit code {ret}")
else:
    print(f"Successfully merged {len(chunk_files)} chunks.")
    print(f"Final averaged author–author layer written to:\n  {out_path}")

print("\nYou may delete the temporary chunks if desired:")
print(f"  rm -r {tmp_dir}")
print("All done")
