# hybrid_retrieval.py
import os, json, pickle
from pathlib import Path
import numpy as np

# Optional imports guarded
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent
RETRIEVAL_DIR = BASE / "retrieval_data"
SPARSE_PATH = RETRIEVAL_DIR / "sparse_tfidf.pkl"
DENSE_EMB = RETRIEVAL_DIR / "dense_embeddings.npy"
FAISS_PATH = RETRIEVAL_DIR / "faiss_hnsw.index"
DENSE_META = RETRIEVAL_DIR / "dense_metadata.json"

# Load sparse (required)
if not SPARSE_PATH.exists():
    raise FileNotFoundError(f"Sparse index missing at {SPARSE_PATH}")
with open(SPARSE_PATH, "rb") as fh:
    sparse_bundle = pickle.load(fh)
    vectorizer = sparse_bundle["vectorizer"]
    sparse_matrix = sparse_bundle["matrix"]
    SPARSE_DOCS = sparse_bundle["docs"]

# Build a lookup by source from sparse
SPARSE_TEXT_LOOKUP = {d["source"]: d["text"] for d in SPARSE_DOCS}

# Dense artifacts (optional)
HAS_DENSE = False
dense_index = None
dense_meta = None
embedder = None

if FAISS_AVAILABLE and FAISS_PATH.exists() and DENSE_META.exists():
    try:
        dense_index = faiss.read_index(str(FAISS_PATH))
        dense_meta = json.load(open(DENSE_META, "r", encoding="utf-8"))
        HAS_DENSE = True
    except Exception:
        HAS_DENSE = False

# function to resolve dense text -> fallback to sparse
def resolve_text_for_source(src: str) -> str:
    if dense_meta:
        # dense_meta entries may or may not have "text"
        m = next((x for x in dense_meta if x.get("source") == src), None)
        if m and m.get("text"):
            return m.get("text")
    # fallback to sparse
    return SPARSE_TEXT_LOOKUP.get(src, "")

def compute_query_embedding(query: str):
    global embedder
    try:
        if embedder is None:
            model_name = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
            embedder = SentenceTransformer(model_name)
        emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        return emb.astype("float32")
    except Exception:
        return None

def query_sparse(query: str, top_k: int = 8):
    qv = vectorizer.transform([query])
    scores = (sparse_matrix @ qv.T).toarray().ravel()
    idxs = np.argsort(-scores)[:top_k]
    results = []
    for i in idxs:
        results.append((SPARSE_DOCS[int(i)], float(scores[int(i)])))
    return results

def query_dense_vector(q_emb: np.ndarray, top_k: int = 8):
    if not HAS_DENSE or dense_index is None:
        return []
    D, I = dense_index.search(q_emb.reshape(1, -1), top_k)
    D = D.ravel().tolist()
    I = I.ravel().tolist()
    res = []
    for idx, dist in zip(I, D):
        if idx < 0 or idx >= len(dense_meta):
            continue
        meta = dense_meta[idx]
        src = meta.get("source")
        score = 1.0 / (1.0 + float(dist))
        text = meta.get("text") or resolve_text_for_source(src)
        res.append(({"id": meta.get("id", f"dense-{idx}"), "source": src, "text": text}, float(score)))
    return res

def hybrid_retrieve(query: str, top_k: int = 6, use_hybrid: bool = True, dense_weight: float = 0.55, sparse_weight: float = 0.45):
    # sparse candidates
    sparse_hits = query_sparse(query, top_k=top_k*2)
    sparse_docs = [h[0] for h in sparse_hits]
    sparse_scores_raw = np.array([h[1] for h in sparse_hits], dtype=float)
    if sparse_scores_raw.max() > 0:
        sparse_scores = sparse_scores_raw / float(sparse_scores_raw.max())
    else:
        sparse_scores = sparse_scores_raw

    if use_hybrid and HAS_DENSE:
        q_emb = compute_query_embedding(query)
        if q_emb is None:
            # fallback to sparse only
            docs = []
            scores = []
            for d, s in zip(sparse_docs[:top_k], sparse_scores[:top_k]):
                docs.append({"id": d.get("id"), "source": d.get("source"), "text": d.get("text")})
                scores.append(float(s))
            return docs, scores
        dense_hits = query_dense_vector(q_emb, top_k=top_k*2)
        # merge by source
        candidate = {}
        for d, s in zip(sparse_docs, sparse_scores):
            src = d.get("source")
            candidate[src] = {"doc": d, "sparse": float(s), "dense": 0.0}
        for dmeta, dscore in dense_hits:
            src = dmeta.get("source")
            if src in candidate:
                candidate[src]["dense"] = float(dscore)
                if dmeta.get("text"):
                    candidate[src]["doc"]["text"] = dmeta.get("text")
            else:
                candidate[src] = {"doc": {"id": dmeta.get("id"), "source": src, "text": dmeta.get("text")}, "sparse": 0.0, "dense": float(dscore)}
        fused = []
        for src, parts in candidate.items():
            s = parts["sparse"]
            d = parts["dense"]
            fused_score = dense_weight * float(d) + sparse_weight * float(s)
            fused.append((parts["doc"], float(fused_score)))
        fused_sorted = sorted(fused, key=lambda x: x[1], reverse=True)[:top_k]
        docs_out = [x[0] for x in fused_sorted]
        scores_out = [x[1] for x in fused_sorted]
        return docs_out, scores_out
    else:
        docs_out = [{"id": d.get("id"), "source": d.get("source"), "text": d.get("text")} for d in sparse_docs[:top_k]]
        scores_out = sparse_scores[:top_k].tolist()
        return docs_out, scores_out
