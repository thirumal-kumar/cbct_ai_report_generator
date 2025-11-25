#!/usr/bin/env python3
"""
build_master_rag.py
Build a merged RAG (sparse TF-IDF + dense MPNet embeddings + FAISS HNSW)
Outputs written to retrieval_data/merged_*
- merged_sparse.pkl       : {'vectorizer': vectorizer, 'matrix': X, 'docs': docs}
- merged_dense_fp16.npy   : embeddings (float16) shape (N, 768)
- merged_faiss.index      : faiss HNSW index
- merged_metadata.json    : list of metadata dicts per chunk
Usage:
    python build_master_rag.py --input-dirs ./rag_db ./pdfs --outdir retrieval_data
"""
import argparse
import os
import json
import math
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
import hashlib

# PDF/text extraction
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# ---------------- helpers ----------------
def extract_text_from_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. pip install PyPDF2")
    txt = []
    try:
        reader = PdfReader(str(path))
        for p in reader.pages:
            try:
                text = p.extract_text() or ""
            except Exception:
                text = ""
            txt.append(text)
    except Exception as e:
        print("[WARN] PDF read failed:", path, e)
    return "\n".join(txt)

def read_txt_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def chunk_text(text, max_chars=1200, overlap=200):
    if not text:
        return []
    text = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # fallback: split long paragraphs
    chunks = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(len(p), start + max_chars)
                chunk = p[start:end]
                chunks.append(chunk)
                if end == len(p): break
                start = end - overlap
    return chunks

def safe_id(s):
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return h

# ---------------- main ----------------
def main(input_dirs, outdir, max_features=50000, embed_model="all-mpnet-base-v2", batch_size=64):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    docs = []  # each doc = {id, source, text, chunk_id, chunk_text}
    print("[BUILD] Scanning input dirs:", input_dirs)
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            print("[WARN] path not found:", p)
            continue
        for f in sorted(p.rglob("*")):
            if f.is_dir(): continue
            lower = f.name.lower()
            if lower.endswith(".pdf"):
                text = extract_text_from_pdf(f)
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    docs.append({
                        "id": safe_id(str(f) + str(i)),
                        "source": str(f),
                        "orig_file": str(f),
                        "chunk_index": i,
                        "text": c
                    })
            elif lower.endswith(".txt"):
                text = read_txt_file(f)
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    docs.append({
                        "id": safe_id(str(f) + str(i)),
                        "source": str(f),
                        "orig_file": str(f),
                        "chunk_index": i,
                        "text": c
                    })
    print(f"[BUILD] Extracted {len(docs)} text chunks from inputs.")

    if len(docs) == 0:
        print("[BUILD] No chunks found. Exiting.")
        return

    corpus = [d["text"] for d in docs]

    # ------------- TF-IDF -------------
    print("[BUILD] Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    sparse_path = outdir / "merged_sparse.pkl"
    with open(sparse_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": X, "docs": docs}, f)
    print("[BUILD] Saved sparse ->", sparse_path)

    # ------------- Dense embeddings (ST) -------------
    print("[BUILD] Loading sentence-transformers model:", embed_model)
    model = SentenceTransformer(embed_model)
    print("[BUILD] Computing dense embeddings (float32) in batches...")
    emb = model.encode(corpus, batch_size=batch_size, show_progress_bar=True)
    emb = np.array(emb, dtype=np.float32)
    # convert to fp16 for storage
    emb_fp16 = emb.astype(np.float16)
    dense_path = outdir / "merged_dense_fp16.npy"
    np.save(str(dense_path), emb_fp16)
    print("[BUILD] Saved dense_fp16 ->", dense_path)

    # metadata
    meta = [{"id": d["id"], "source": d["source"], "chunk_index": d["chunk_index"], "text": (d["text"][:800] if d["text"] else "")} for d in docs]
    with open(outdir / "merged_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[BUILD] Saved metadata ->", outdir / "merged_metadata.json")

    # ------------- FAISS HNSW -------------
    print("[BUILD] Building FAISS HNSW index (dim=%d) ..." % emb.shape[1])
    d = emb.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 40
    index.add(emb)  # add float32 embeddings
    faiss_index_path = outdir / "merged_faiss.index"
    faiss.write_index(index, str(faiss_index_path))
    print("[BUILD] Saved faiss index ->", faiss_index_path)

    print("[BUILD] Done. Summary:")
    print(" - chunks:", len(docs))
    print(" - tfidf:", sparse_path)
    print(" - dense (fp16):", dense_path)
    print(" - faiss:", faiss_index_path)
    print(" - meta:", outdir / "merged_metadata.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", default=["./rag_db", "."],
                        help="Folders to scan for PDFs and TXT files")
    parser.add_argument("--outdir", default="retrieval_data", help="output dir")
    parser.add_argument("--embed-model", default="all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args.input_dirs, args.outdir, embed_model=args.embed_model, batch_size=args.batch_size)
