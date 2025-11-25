# -------------------------------------------------------------
# build_sparse_index.py (flat folder support)
# -------------------------------------------------------------
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = "./rag_db"
OUTPUT_DIR = "./retrieval_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

docs = []

print("📌 Scanning rag_db for .txt files...")

counter = 1

for fname in os.listdir(BASE_DIR):
    if not fname.endswith(".txt"):
        continue

    fpath = os.path.join(BASE_DIR, fname)

    try:
        with open(fpath, "r") as f:
            text = f.read().strip()

        docs.append({
            "id": f"doc-{counter}",
            "text": text,
            "source": fpath,
            "category": "unknown"  # we can classify later
        })

        counter += 1
    except:
        print(f"⚠️ Could not read {fname}")

print(f"📌 Loaded {len(docs)} documents.")

corpus = [d["text"] for d in docs]

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)
X = vectorizer.fit_transform(corpus)

with open(os.path.join(OUTPUT_DIR, "sparse_tfidf.pkl"), "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "matrix": X,
        "docs": docs
    }, f)

print("✅ TF-IDF Index saved → retrieval_data/sparse_tfidf.pkl")
