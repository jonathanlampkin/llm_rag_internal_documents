from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PageFeature:
    doc_id: str
    page_idx: int
    page_hash: str
    page_w_in: float
    page_h_in: float
    eff_dpi: float
    skew_deg: float
    lap_var: float
    noise_mad: float
    bg_uniform: float
    bin_diff: float
    text_density: float
    xh_px: float
    est_pt: float
    cols: int
    line_spacing_var: float
    printed_prob: float
    hw_prob: float
    forms_prob: float
    photo_prob: float
    brisque: float
    compression: str
    has_text_layer: bool

@dataclass
class PageCluster:
    doc_id: str
    page_idx: int
    cluster_id: int
    cluster_conf: float

@dataclass
class PageRoute:
    doc_id: str
    page_idx: int
    engine: str
    preproc_flags: Dict[str, Any]
    pred_cer: float
    cost_est: float
    rationale: str

