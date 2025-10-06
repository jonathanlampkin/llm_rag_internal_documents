import json, sys
import pytesseract, fitz
from PIL import Image

def _prep_image(page, dpi: int, binarize: str, deskew: bool):
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    gray = img.convert("L")
    # simple binarization options
    if binarize == "otsu":
        # Pillow lacks Otsu; rely on tesseract internal binarization; keep gray
        proc = gray
    else:
        proc = gray
    return proc

def run_pdf(pdf_path: str, routes_parquet: str, out_jsonl: str):
    import pandas as pd
    doc = fitz.open(pdf_path)
    routes = pd.read_parquet(routes_parquet)
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for _, r in routes.iterrows():
            i = int(r.page_idx)
            page = doc.load_page(i)
            if r.engine == "skip_text":
                txt = page.get_text("text")
                obj = {"doc_id": r.doc_id, "page_idx": i, "raw_text": txt, "engine": "skip_text"}
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            flags = r.preproc_flags or {}
            dpi = int(flags.get("dpi", 200))
            img = _prep_image(page, dpi=dpi, binarize=flags.get("binarize","otsu"), deskew=flags.get("deskew", False))
            text = pytesseract.image_to_string(img)
            obj = {"doc_id": r.doc_id, "page_idx": i, "raw_text": text, "engine": "tesseract"}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

