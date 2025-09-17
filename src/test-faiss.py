import numpy as np
import faiss

print("FAISS version:", faiss.__version__)
print("NumPy version:", np.__version__)
print("GPUs visible to FAISS:", faiss.get_num_gpus())

# Generate some random 768-dim vectors (like SPECTER embeddings)
d = 768   # dimension
nb = 1000 # database size
nq = 5    # queries

np.random.seed(1234)
xb = np.random.random((nb, d)).astype("float32")
xq = np.random.random((nq, d)).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# Build index on CPU
index = faiss.IndexFlatIP(d)
index.add(xb)

# Move to GPU
index = faiss.index_cpu_to_all_gpus(index)

# Search top-5 nearest neighbors for queries
D, I = index.search(xq, 5)

print("Query results (indices):")
print(I)
print("Query results (similarities):")
print(D)
