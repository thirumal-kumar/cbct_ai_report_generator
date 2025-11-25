# hybrid_fusion.py
import os, pickle, json, numpy as np
import faiss
from sklearn.preprocessing import minmax_scale

def load_sparse(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["vectorizer"], d["matrix"], d["docs"]

def load_dense_emb_and_meta(emb_path, meta_path, faiss_path):
    emb = np.load(emb_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    idx = faiss.read_index(faiss_path)
    return emb, meta, idx

def sparse_query(vectorizer, matrix, docs, query, topk=6):
    qv = vectorizer.transform([query])
    scores = (matrix @ qv.T).toarray().ravel()
    idxs = np.argsort(-scores)[:topk]
    return [{"score": float(scores[i]), "source": docs[i]["source"], "text": docs[i]["text"]} for i in idxs if scores[i] > 0]

def dense_query(faiss_idx, emb_matrix, meta, model_encode_fn, query, topk=6):
    q_emb = model_encode_fn([query])  # must return np.ndarray shape (1, dim)
    D, I = faiss_idx.search(q_emb.astype(np.float32), topk)
    out = []
    for dist_row, idx_row in zip(D, I):
        for dist, i in zip(dist_row, idx_row):
            if i < 0: continue
            # convert distance to similarity-ish (faiss returns L2 if index is flat; HNSW may differ)
            score = float(1.0 / (1.0 + dist))
            out.append({"score": score, "source": meta[i]["source"], "text": meta[i].get("text","")})
    return out

def normalize_and_weight(results_by_source, weights):
    # results_by_source = {"clinical": [{"score":..}, ...], "atlas": [...]}
    merged = []
    for name, res in results_by_source.items():
        arr = np.array([r["score"] for r in res], dtype=float) if res else np.array([])
        if arr.size:
            arr_n = minmax_scale(arr)  # scale 0-1
        else:
            arr_n = arr
        for i,r in enumerate(res):
            merged.append({
                "text": r["text"],
                "source": name,
                "raw_score": float(r["score"]),
                "norm_score": float(arr_n[i]) if arr_n.size else 0.0,
                "weighted": float(arr_n[i] * weights.get(name, 1.0))
            })
    # sort by weighted desc
    merged = sorted(merged, key=lambda x: -x["weighted"])
    # dedupe by text similarity (simple exact or prefix check)
    seen = set()
    dedup = []
    for item in merged:
        t = item["text"].strip()[:250]  # first 250 chars
        if t in seen: 
            continue
        seen.add(t)
        dedup.append(item)
    return dedup

# Example wrapper
def hybrid_retrieve_all(query, clinical_paths, atlas_paths, model_encode_fn, topk=6, weights=None):
    weights = weights or {"clinical": 0.6, "atlas": 0.4}
    # load clinical sparse/dense
    vec_c, Xc, docs_c = load_sparse(clinical_paths["sparse"])
    clinical_sparse = sparse_query(vec_c, Xc, docs_c, query, topk=topk)
    emb_c, meta_c, faiss_c = load_dense_emb_and_meta(clinical_paths["dense_np"], clinical_paths["meta_json"], clinical_paths["faiss"])
    clinical_dense = dense_query(faiss_c, emb_c, meta_c, model_encode_fn, query, topk=topk)

    # load atlas
    vec_a, Xa, docs_a = load_sparse(atlas_paths["sparse"])
    atlas_sparse = sparse_query(vec_a, Xa, docs_a, query, topk=topk)
    emb_a, meta_a, faiss_a = load_dense_emb_and_meta(atlas_paths["dense_np"], atlas_paths["meta_json"], atlas_paths["faiss"])
    atlas_dense = dense_query(faiss_a, emb_a, meta_a, model_encode_fn, query, topk=topk)

    # combine per-source lists (you can also merge sparse+dense per source before weighting)
    results_by_source = {
        "clinical": clinical_sparse + clinical_dense,
        "atlas": atlas_sparse + atlas_dense
    }
    merged = normalize_and_weight(results_by_source, weights)
    return merged
