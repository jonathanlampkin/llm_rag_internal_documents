import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FEATURE_COLS = [
    "eff_dpi","lap_var","noise_mad","bg_uniform","bin_diff","text_density"
]

def cluster_features(in_parquet: str, out_parquet: str, k: int = 8):
    df = pd.read_parquet(in_parquet)
    X = df[FEATURE_COLS].fillna(0.0).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(Xs)
    df_out = df[["doc_id","page_idx"]].copy()
    df_out["cluster_id"] = labels
    df_out["cluster_conf"] = 1.0
    df_out.to_parquet(out_parquet, index=False)

