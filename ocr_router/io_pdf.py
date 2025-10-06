import hashlib, fitz
from typing import Tuple

def page_hash_from_pix(pix) -> str:
    return hashlib.sha256(pix.samples).hexdigest()

def load_page_rawdict(doc, i):
    page = doc.load_page(i)
    return page, page.get_text("rawdict")

def has_text_layer(rawdict) -> bool:
    if not rawdict: return False
    blocks = rawdict.get("blocks", [])
    for b in blocks:
        if b.get("type") == 0:  # text block
            spans = b.get("spans", [])
            if any((s.get("text") or "").strip() for s in spans):
                return True
    return False

def page_image_and_size(page, dpi: int = 96):
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    w_in = page.rect.width / 72.0
    h_in = page.rect.height / 72.0
    return pix, w_in, h_in

