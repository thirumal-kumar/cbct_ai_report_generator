# build_pdf_index_from_atlas.py
# Usage: python build_pdf_index_from_atlas.py
import os
import json
import math
import pickle
from pathlib import Path
from tqdm import tqdm

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# CONFIG
PDF_PATH = "/home/thirumal/Desktop/cbct_ai_app/Cone Beam CT of the Head and Neck An Anatomical Atlas.pdf"  # <--- atlas path
OUTPUT_DIR = "retrieval_data"
CHUNK_SIZE = 800         # approx characters
CHUNK_OVERLAP = 120
EMBED_MODEL = "all-mpnet-base-v2"
BATCH_SIZE = 64
TFIDF_MAX_FEATURES = 50000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for p in range(doc.page_count):
        text = doc.load_page(p).get_text("text")
        pages.append({"page": p+1, "text": text})
    doc.close()
    return pages

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text or len(text.strip()) == 0:
        return []
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_chunks_from_pages(pages):
    docs = []
    for p in pages:
        page_num = p["page"]
        text = p["text"] or ""
        page_chunks = chunk_text(text)
        for i, c in enumerate(page_chunks):
            docs.append({
                "id": f"atlas_p{page_num}_c{i}",
                "page": page_num,
                "chunk_index": i,
                "text": c
            })
    return docs

def build_sparse(docs, outdir):
    corpus = [d["text"] for d in docs]
    print("[BUILD] Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    X = vectorizer.fit_transform(corpus)
    with open(Path(outdir)/"sparse_tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": X, "docs": docs}, f)
    print("[BUILD] Saved sparse_tfidf.pkl")

def build_dense(docs, outdir):
    corpus = [d["text"] for d in docs]
    print(f"[BUILD] Loading embed model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("[BUILD] Creating embeddings...")
    emb = model.encode(corpus, batch_size=BATCH_SIZE, show_progress_bar=True)
    emb = np.array(emb).astype("float32")
    np.save(Path(outdir)/"dense_embeddings.npy", emb)
    dense_metadata = [{"id": d["id"], "page": d["page"], "chunk_index": d["chunk_index"], "text": d["text"]} for d in docs]
    with open(Path(outdir)/"dense_metadata.json", "w") as f:
        json.dump(dense_metadata, f, indent=2)
    print("[BUILD] Saved dense embeddings + metadata.")

    # FAISS HNSW
    d = emb.shape[1]
    print(f"[BUILD] Creating FAISS HNSW index (dim={d})...")
    index = faiss.IndexHNSWFlat(d, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(emb)
    faiss.write_index(index, str(Path(outdir) / "faiss_hnsw.index"))
    print("[BUILD] Saved faiss_hnsw.index")

if __name__ == "__main__":
    print("[BUILD] Extracting PDF pages...")
    pages = extract_pdf_pages(PDF_PATH)
    print(f"[BUILD] Extracted {len(pages)} pages.")
    docs = build_chunks_from_pages(pages)
    print(f"[BUILD] Built {len(docs)} text chunks.")
    build_sparse(docs, OUTPUT_DIR)
    build_dense(docs, OUTPUT_DIR)
    print("\n[BUILD] Atlas-only hybrid index build complete.")
