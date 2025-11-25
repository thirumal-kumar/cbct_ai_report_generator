import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

SOURCE_DIR = "rag_db"
OUTPUT_DIR = "retrieval_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[BUILD] Loading text documents...")

docs = []
for fname in sorted(os.listdir(SOURCE_DIR)):
    path = os.path.join(SOURCE_DIR, fname)
    if not fname.endswith(".txt"):
        continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    docs.append({"id": fname.replace(".txt", ""), "source": path, "text": txt})

print(f"[BUILD] Loaded {len(docs)} text documents.")

corpus = [d["text"] for d in docs]

# ----------- TF-IDF -----------
print("[BUILD] Building sparse TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
X = vectorizer.fit_transform(corpus)

with open(os.path.join(OUTPUT_DIR, "sparse_tfidf.pkl"), "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "matrix": X, "docs": docs}, f)

print("[BUILD] Saved sparse_tfidf.pkl")

# ----------- Dense Embeddings -----------
print("[BUILD] Loading ST model: all-mpnet-base-v2")
model = SentenceTransformer("all-mpnet-base-v2")

print("[BUILD] Creating embeddings...")
emb = model.encode(corpus, batch_size=64, show_progress_bar=True)

np.save(os.path.join(OUTPUT_DIR, "dense_embeddings.npy"), emb)

dense_metadata = [
    {"id": d["id"], "source": d["source"], "text": d["text"]}
    for d in docs
]

with open(os.path.join(OUTPUT_DIR, "dense_metadata.json"), "w") as f:
    json.dump(dense_metadata, f, indent=2)

print("[BUILD] Saved dense embeddings + metadata.")

# ----------- FAISS Index -----------
print("[BUILD] Creating FAISS HNSW index...")

d = emb.shape[1]
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 40
index.add(emb)

faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_hnsw.index"))

print("[BUILD] Saved faiss_hnsw.index")

print("\n[BUILD] Hybrid Index Build Complete.")
