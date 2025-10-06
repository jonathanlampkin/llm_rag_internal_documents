import os, glob, pickle, subprocess, json
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from pypdf import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def _extract_text_pypdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()

def _extract_text_ocr(pdf_path: str, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for page in doc:
        # Render page to image
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Convert to grayscale to help OCR
        gray = img.convert("L")
        text = pytesseract.image_to_string(gray)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()

def _needs_ocr(text: str, char_threshold: int = 50) -> bool:
    # Heuristic: if too few extractable characters, assume scanned PDF
    # Also check ratio of non-whitespace characters
    non_ws = sum(1 for c in text if not c.isspace())
    return non_ws < char_threshold

def _ocr_router_extract(pdf_path: str) -> str:
    """Use local one-shot OCR router CLI if available; else fallback to built-ins."""
    try:
        # Probe → cluster → route → run (module CLI)
        prefix = os.path.splitext(pdf_path)[0]
        feats = f"{prefix}.features.parquet"
        clust = f"{prefix}.clusters.parquet"
        routes = f"{prefix}.routes.parquet"
        out_jsonl = f"{prefix}.ocr.jsonl"
        subprocess.run(["python","-m","ocr_router.cli","probe", pdf_path, "--out", feats], check=True)
        subprocess.run(["python","-m","ocr_router.cli","cluster", feats, "--out", clust], check=True)
        subprocess.run(["python","-m","ocr_router.cli","route", clust, "--out", routes], check=True)
        subprocess.run(["python","-m","ocr_router.cli","run", routes, "--out", out_jsonl], check=True)
        # Concatenate page texts
        texts: List[str] = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("raw_text", "")
                    if txt:
                        texts.append(txt)
                except Exception:
                    continue
        return "\n".join(texts).strip()
    except Exception:
        return ""

def load_pdfs(path: str = "data/docs/*.pdf") -> Tuple[List[str], List[str]]:
    docs: List[str] = []
    ids: List[str] = []
    for fp in sorted(glob.glob(path)):
        base = os.path.basename(fp)
        # Prefer router if installed; fall back to local text/ocr
        text = _ocr_router_extract(fp)
        if not text:
            text = _extract_text_pypdf(fp)
            if _needs_ocr(text):
                text = _extract_text_ocr(fp)
        ids.append(base)
        docs.append(text or "")
    return ids, docs

def chunk(text, size=400, overlap=50):
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size]); i+= size-overlap
    return out

if __name__ == "__main__":
    ids, docs = load_pdfs()
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
