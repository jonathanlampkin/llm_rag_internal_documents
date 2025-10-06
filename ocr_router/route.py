import pandas as pd

def route_pages(features_parquet: str, clusters_parquet: str, out_parquet: str):
    feats = pd.read_parquet(features_parquet)
    clus = pd.read_parquet(clusters_parquet)
    df = feats.merge(clus, on=["doc_id","page_idx"], how="left")
    engines = []
    preproc = []
    pred_cer = []
    cost = []
    rationale = []
    for _, r in df.iterrows():
        if r.get("has_text_layer", False):
            engines.append("skip_text")
            preproc.append({})
            pred_cer.append(0.0)
            cost.append(0.0)
            rationale.append("text layer present")
            continue
        # pre-gates for difficulty
        difficult = (r.eff_dpi < 220) or (r.lap_var < 20) or (r.bin_diff > 0.15)
        if difficult:
            engines.append("tesseract_heavy")
            preproc.append({"deskew": True, "binarize": "sauvola", "dpi": 300})
            pred_cer.append(0.12)
            cost.append(0.0006)
            rationale.append("difficult page heuristics")
            continue
        # cluster-based default
        if int(r.cluster_id) % 2 == 0:
            engines.append("tesseract_light")
            preproc.append({"deskew": False, "binarize": "otsu", "dpi": 200})
            pred_cer.append(0.05)
            cost.append(0.0002)
            rationale.append("clean cluster")
        else:
            engines.append("tesseract_heavy")
            preproc.append({"deskew": True, "binarize": "sauvola", "dpi": 300})
            pred_cer.append(0.1)
            cost.append(0.0006)
            rationale.append("noisy cluster")
    out = df[["doc_id","page_idx"]].copy()
    out["engine"] = engines
    out["preproc_flags"] = preproc
    out["pred_cer"] = pred_cer
    out["cost_est"] = cost
    out["rationale"] = rationale
    out.to_parquet(out_parquet, index=False)

