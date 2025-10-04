import time, pickle, faiss
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram
from typing import List, Tuple

REQS = Counter("rag_requests_total", "RAG requests")
LAT = Histogram("rag_latency_seconds", "RAG latency")
RETR_PRECISION = Counter("retrieved_gt_hits_total", "GT hits in topk")  # used by eval

class RAG:
    def __init__(self, idx="artifacts/index.faiss", meta="artifacts/meta.pkl"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(idx)
        bundle = pickle.load(open(meta, "rb"))
        self.meta, self.chunks = bundle["meta"], bundle["chunks"]

    def retrieve(self, q: str, k=4) -> Tuple[List[int], List[str]]:
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(qv, k)
        return I[0].tolist(), [self.chunks[i] for i in I[0]]

    def context_str(self, idxs: List[int]) -> str:
        parts=[]
        for i in idxs:
            m = self.meta[i]
            parts.append(f"[{m['doc_id']}#{m['chunk_id']}] {self.chunks[i]}")
        return "\n---\n".join(parts)

    def answer(self, question: str, generator) -> str:
        start = time.time(); REQS.inc()
        idxs, chunks = self.retrieve(question)
        ctx = self.context_str(idxs)
        out = generator(question, ctx)
        LAT.observe(time.time()-start)
        return out, idxs
