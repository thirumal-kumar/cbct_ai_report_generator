#!/usr/bin/env python3
"""
build_multi_source_index.py

Build a merged hybrid retrieval index from:
 - PDFs in a specified directory (OCR where needed)
 - existing RAG text files in rag_db/

Outputs (retrieval_data/):
 - atlas_sparse.pkl         (tfidf vectorizer + sparse matrix + docs list)
 - atlas_dense.npy          (dense embeddings npy)
 - atlas_metadata.json      (list of {id, source, text})
 - atlas_faiss.index        (faiss HNSW index)
 - merged_sparse.pkl        (merged with rag_db)
 - merged_dense.npy
 - merged_metadata.json
 - merged_faiss.index

Notes:
 - Uses pdfplumber to extract text; falls back to rendering page with PyMuPDF and pytesseract OCR.
 - Chunking uses simple sliding window for robust retrieval.
"""

import os
import re
import json
import time
import math
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pdfplumber
import fitz  # pymupdf
import pytesseract

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

# ----------------- config -----------------
PDF_DIR = Path(".")                    # where your PDFs are (or pass another path)
RAG_DIR = Path("rag_db")               # existing text snippets
OUT_DIR = Path("retrieval_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_PREFIX = "atlas"                 # produced atlas-only files
MERGED_PREFIX = "merged"               # combined with rag_db output

EMBED_MODEL = "all-mpnet-base-v2"      # sentence-transformers model
BATCH_SIZE = 64
FAISS_M = 32                           # HNSW param
CHUNK_SIZE = 400                       # tokens ~ words rough; this is words
CHUNK_OVERLAP = 80

# file globs to ingest (PDFs)
PDF_GLOBS = ["*.pdf"]

# ----------------- helpers -----------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    # normalize whitespace
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = t.strip()
    return t

def pdf_extract_pages_text(pdf_path: Path):
    """Return list of page texts. Try pdfplumber; if page has low text, do OCR via pymupdf + pytesseract."""
    pages_text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = clean_text(txt)
            # if page text is short, try OCR
            if len(txt) < 80:
                try:
                    # render page to image via pymupdf for OCR
                    doc = fitz.open(str(pdf_path))
                    pg = doc.load_page(i)
                    mat = fitz.Matrix(2, 2)  # render at higher zoom for OCR quality
                    pix = pg.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("png")
                    ocr_txt = pytesseract.image_to_string(img_bytes)
                    ocr_txt = clean_text(ocr_txt)
                    if len(ocr_txt) > len(txt):
                        txt = ocr_txt
                except Exception:
                    # fallback: keep whatever text we had
                    pass
            pages_text.append(txt)
    return pages_text

def chunk_text(text: str, chunk_size_words=CHUNK_SIZE, overlap_words=CHUNK_OVERLAP):
    """Split by words into overlapping chunks"""
    words = text.split()
    if len(words) <= chunk_size_words:
        return [ " ".join(words) ]
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i+chunk_size_words]
        chunks.append(" ".join(chunk))
        i += (chunk_size_words - overlap_words)
    return chunks

# ----------------- ingest PDFs and build docs list -----------------
def build_docs_from_pdfs(pdf_dir: Path):
    pdf_files = []
    for g in PDF_GLOBS:
        pdf_files.extend(sorted(pdf_dir.glob(g)))
    docs = []
    for pdf in tqdm(pdf_files, desc="PDFs"):
        try:
            pages = pdf_extract_pages_text(pdf)
        except Exception as e:
            print(f"[WARN] Failed to extract {pdf}: {e}")
            pages = []
        # aggregate pages into a single long text and chunk
        full_text = "\n\n".join(pages)
        full_text = clean_text(full_text)
        if not full_text:
            continue
        chunks = chunk_text(full_text)
        for idx, c in enumerate(chunks):
            doc_id = f"{pdf.stem}_p{idx}"
            docs.append({"id": doc_id, "source": str(pdf), "text": c})
    print(f"[BUILD] PDF-derived docs: {len(docs)}")
    return docs

# ----------------- ingest rag db text files -----------------
def build_docs_from_rag(rag_dir: Path):
    docs = []
    if not rag_dir.exists():
        return docs
    for fname in sorted(rag_dir.glob("*.txt")):
        try:
            txt = fname.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            doc_id = fname.stem
            docs.append({"id": doc_id, "source": str(fname), "text": clean_text(txt)})
        except Exception as e:
            print(f"[WARN] could not read {fname}: {e}")
    print(f"[BUILD] RAG docs: {len(docs)}")
    return docs

# ----------------- TF-IDF (sparse) builder -----------------
def build_sparse_index(docs, out_prefix):
    print("[BUILD] Building sparse TF-IDF...")
    corpus = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vectorizer.fit_transform(corpus)
    out_path = OUT_DIR / f"{out_prefix}_sparse.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": X, "docs": docs}, f)
    print("[BUILD] Saved", out_path)
    return out_path

# ----------------- dense embeddings + faiss -----------------
def build_dense_and_faiss(docs, out_prefix, model_name=EMBED_MODEL):
    print(f"[BUILD] Loading embed model: {model_name}")
    model = SentenceTransformer(model_name)
    corpus = [d["text"] for d in docs]
    emb = model.encode(corpus, batch_size=BATCH_SIZE, show_progress_bar=True)
    emb = np.array(emb).astype("float32")
    dense_npy = OUT_DIR / f"{out_prefix}_dense.npy"
    np.save(str(dense_npy), emb)
    # metadata
    metadata = [{"id": d["id"], "source": d["source"], "text": d["text"]} for d in docs]
    meta_path = OUT_DIR / f"{out_prefix}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    # FAISS HNSW
    dim = emb.shape[1]
    print(f"[BUILD] Creating FAISS HNSW index (dim={dim}) ...")
    index = faiss.IndexHNSWFlat(dim, FAISS_M)
    try:
        index.hnsw.efConstruction = 40
    except Exception:
        pass
    index.add(emb)
    faiss_path = OUT_DIR / f"{out_prefix}_faiss.index"
    # faiss.write_index requires string path on many builds
    faiss.write_index(index, str(faiss_path))
    print("[BUILD] Saved", faiss_path)
    return dense_npy, meta_path, faiss_path

# ----------------- merge doc lists utility -----------------
def merge_docs(primary_docs, secondary_docs):
    # primary_docs first (e.g., atlas), then rag. Avoid duplicate IDs.
    ids = set(d["id"] for d in primary_docs)
    merged = list(primary_docs)
    for d in secondary_docs:
        if d["id"] in ids:
            # if same id, skip (or could rename)
            continue
        merged.append(d)
    return merged

# ----------------- main build flow -----------------
def main():
    start = time.time()
    print("[BUILD] Starting multi-source index build...")
    # 1) PDF-derived docs (atlas + other PDFs)
    pdf_docs = build_docs_from_pdfs(PDF_DIR)

    # 2) RAG text docs (existing)
    rag_docs = build_docs_from_rag(RAG_DIR)

    # 3) Build atlas-only indices (from pdfs)
    if len(pdf_docs) > 0:
        atlas_sparse = build_sparse_index(pdf_docs, ATLAS_PREFIX)
        atlas_dense, atlas_meta, atlas_faiss = build_dense_and_faiss(pdf_docs, ATLAS_PREFIX)
    else:
        print("[BUILD] No PDF docs found; skipping atlas build.")
        atlas_sparse, atlas_dense, atlas_meta, atlas_faiss = None, None, None, None

    # 4) Merge with rag and build merged indices
    merged_docs = merge_docs(pdf_docs, rag_docs) if pdf_docs else rag_docs
    if len(merged_docs) == 0:
        print("[BUILD] No merged docs available. Exiting.")
        return

    merged_sparse = build_sparse_index(merged_docs, MERGED_PREFIX)
    merged_dense, merged_meta, merged_faiss = build_dense_and_faiss(merged_docs, MERGED_PREFIX)

    elapsed = time.time() - start
    print(f"[BUILD] Done. Time: {elapsed:.1f}s")
    print("Outputs in", OUT_DIR.resolve())
    print(" - atlas: ", atlas_sparse if atlas_sparse else "none", atlas_dense, atlas_meta, atlas_faiss)
    print(" - merged:", merged_sparse, merged_dense, merged_meta, merged_faiss)

if __name__ == "__main__":
    main()
