# A toy precision@k evaluator given {question, gt_doc_id}
import json, pickle, faiss
from sentence_transformers import SentenceTransformer
import numpy as np

EVAL = [
  {"q":"What does the README say about setup?","gt":"README.pdf"},
  {"q":"What is our privacy policy stance?","gt":"privacy.pdf"}
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("artifacts/index.faiss")
bundle = pickle.load(open("artifacts/meta.pkl","rb"))
meta, chunks = bundle["meta"], bundle["chunks"]
def topk_docs(q, k=4):
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D,I = index.search(qv, k)
    return [meta[i]["doc_id"] for i in I[0]]

hits=0
for item in EVAL:
    docs = topk_docs(item["q"])
    hits += int(item["gt"] in docs)
print(f"precision@4={hits/len(EVAL):.2f}")
