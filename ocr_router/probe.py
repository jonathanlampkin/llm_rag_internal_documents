import numpy as np, cv2, pandas as pd, json
import fitz
from .io_pdf import load_page_rawdict, has_text_layer, page_image_and_size, page_hash_from_pix
from .schemas import PageFeature

def _blankness_metrics(img_gray: np.ndarray):
    mean = float(img_gray.mean())
    std = float(img_gray.std())
    return mean, std

def _lap_var(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def _bin_diff(img_gray: np.ndarray) -> float:
    # Otsu vs Sauvola disagreement rate as difficulty proxy
    _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    sau = cv2.ximgproc.niBlackThreshold(img_gray, 255, cv2.THRESH_BINARY, 51, k=0.2) if hasattr(cv2, 'ximgproc') else otsu
    return float((otsu != sau).mean())

def _text_density(img_gray: np.ndarray) -> float:
    # crude: count dark pixels after light blur
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return float((th==0).mean())

def probe_pdf(pdf_path: str, out_parquet: str):
    doc = fitz.open(pdf_path)
    rows = []
    for i in range(len(doc)):
        page, raw = load_page_rawdict(doc, i)
        pix, w_in, h_in = page_image_and_size(page, dpi=96)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean, std = _blankness_metrics(gray)
        lap = _lap_var(gray)
        td = _text_density(gray)
        bd = _bin_diff(gray)
        eff_dpi = min(pix.width/max(w_in,1e-6), pix.height/max(h_in,1e-6))
        feats = PageFeature(
            doc_id=pdf_path.split('/')[-1], page_idx=i,
            page_hash=page_hash_from_pix(pix), page_w_in=w_in, page_h_in=h_in,
            eff_dpi=float(eff_dpi), skew_deg=0.0,
            lap_var=lap, noise_mad=float(std), bg_uniform=float(std),
            bin_diff=bd, text_density=td, xh_px=0.0, est_pt=0.0,
            cols=1, line_spacing_var=0.0,
            printed_prob=1.0, hw_prob=0.0, forms_prob=0.0, photo_prob=0.0,
            brisque=0.0, compression="unknown", has_text_layer=has_text_layer(raw)
        )
        rows.append(feats.__dict__)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)

