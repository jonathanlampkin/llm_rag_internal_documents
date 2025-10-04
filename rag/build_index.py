import os, glob, pickle
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

def load_docs(path="data/docs/*.txt"):
    docs, ids = [], []
    for i, fp in enumerate(glob.glob(path)):
        with open(fp, "r", encoding="utf-8") as f:
            docs.append(f.read())
            ids.append(os.path.basename(fp))
    return ids, docs

def chunk(text, size=400, overlap=50):
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size]); i+= size-overlap
    return out

if __name__ == "__main__":
    ids, docs = load_docs()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks, meta = [], []
    for doc_id, d in zip(ids, docs):
        for j, ch in enumerate(chunk(d)):
            chunks.append(ch); meta.append({"doc_id":doc_id, "chunk_id":j})
    X = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    os.makedirs("artifacts", exist_ok=True)
    faiss.write_index(index, "artifacts/index.faiss")
    with open("artifacts/meta.pkl","wb") as f: pickle.dump({"meta":meta, "chunks":chunks}, f)
    print(f"Indexed {len(chunks)} chunks from {len(ids)} docs")
