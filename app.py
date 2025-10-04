from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from rag.service import RAG
from rag.llm import generate_answer

class Q(BaseModel):
    question: str
    k: int = 4

app = FastAPI(title="mini-rag")
rag = RAG()

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/ask")
def ask(q: Q):
    ans, idxs = rag.answer(q.question, generate_answer)
    return {"answer": ans, "topk": idxs}
