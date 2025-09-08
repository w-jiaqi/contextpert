import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

def title_case_drug(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if s.isupper() or s.islower():
        s = s.lower().title()
    ACR = {"DNA","RNA","EGFR","HER2","MEK","JAK","BRAF","HDAC","VEGF","PI3K","PD1","PD-L1","CDK"}
    toks = re.split(r"([-/\s])", s)
    toks = [t.upper() if t.upper() in ACR else t for t in toks]
    return "".join(toks)

def flat_upper(m3: np.ndarray) -> np.ndarray:
    n, g, _ = m3.shape
    iu = np.triu_indices(g, 1)
    return m3[:, iu[0], iu[1]].astype(np.float32, copy=False)

def cdist_fast(mat: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean distance calculation."""
    mat = np.asarray(mat, dtype=np.float32, order="C")
    xy = mat @ mat.T
    norms = (mat * mat).sum(1)
    d2 = np.add.outer(norms, norms) - 2 * xy
    np.fill_diagonal(d2, 0.0)
    d2[d2 < 0] = 0.0
    return squareform(np.sqrt(d2, dtype=np.float32))

def palette(values, palette_name="tab20"):
    vals = list(values)
    uniq = pd.unique(pd.Series(vals))
    pal = sns.color_palette(palette_name, n_colors=max(20, len(uniq)))
    lut = {u: pal[i % len(pal)] for i, u in enumerate(uniq)}
    return [lut[v] for v in vals]


def load_and_prepare_data(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    required = ["disease_labels", "pert_id", "drug_name", "cell_id"]
    missing = [f for f in required if f not in z.files]
    if missing:
        raise ValueError(f"NPZ missing fields: {missing}")
    
    N = len(z["disease_labels"])
    def get(k, default=None):
        return z[k] if k in z.files else np.array([default]*N, dtype=object)

    tbl = pd.DataFrame({
        "disease":  np.asarray(get("disease_labels"), dtype=object),
        "drug_id":  np.asarray(get("pert_id"), dtype=object),
        "drugName": np.asarray(get("drug_name"), dtype=object),
        "cell":     np.asarray(get("cell_id"), dtype=object),
    })

    # filter valid rows
    ok = (
        tbl["disease"].astype(str).str.strip().ne("") &
        tbl["disease"].astype(str).str.lower().ne("nan") &
        tbl["drug_id"].notna() & tbl["cell"].notna()
    )
    tbl = tbl.loc[ok].reset_index(drop=True)

    # clean drug names
    disp = tbl["drugName"].fillna("").astype(str).map(title_case_drug)
    disp = np.where(pd.Series(disp).astype(str).str.strip().eq(""), tbl["drug_id"].astype(str), disp)
    tbl["drug_disp"] = disp
    
    return z, tbl, ok.values

def stratified_sample(tbl, max_rows, seed=42):
    if len(tbl) <= max_rows:
        return list(range(len(tbl)))
    
    disease_indices = defaultdict(list)
    for i, disease in enumerate(tbl["disease"]):
        disease_indices[disease].append(i)
    
    samples_per_disease = max(1, max_rows // len(disease_indices))
    idx = []
    
    np.random.seed(seed)
    for disease, indices in disease_indices.items():
        n_take = min(samples_per_disease, len(indices))
        if len(indices) > n_take:
            selected = np.random.choice(indices, n_take, replace=False)
        else:
            selected = indices
        idx.extend(selected)
    
    # Fill remaining slots
    if len(idx) < max_rows:
        remaining = set(range(len(tbl))) - set(idx)
        need_more = max_rows - len(idx)
        if remaining:
            additional = np.random.choice(list(remaining), 
                                        min(need_more, len(remaining)), replace=False)
            idx.extend(additional)
    
    return sorted(idx[:max_rows])

def get_features(z, keep_mask, rep_name):
    if rep_name == "pca":
        if "expressions" not in z.files:
            raise ValueError("expressions not found in NPZ")
        return np.asarray(z["expressions"])[keep_mask]
    elif rep_name == "expr":
        if "expr_raw" not in z.files:
            raise ValueError("expr_raw not found in NPZ")
        return np.asarray(z["expr_raw"])[keep_mask]
    else:  # net
        if "correlations" not in z.files:
            raise ValueError("correlations not found in NPZ")
        return flat_upper(np.asarray(z["correlations"])[keep_mask])

def generate_clustermap(X, tbl, rep_name, out_path, linkage="average"):
    if rep_name == "expr":
        X = StandardScaler().fit_transform(X).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    
    # calc distance and linkage
    dist = cdist_fast(X)
    Z = hierarchy.linkage(dist, method=linkage)
    D = squareform(dist)
    
    idx_labels = [f"sample_{i}" for i in range(len(tbl))]
    df = pd.DataFrame(D, index=idx_labels, columns=idx_labels)
    
    row_colors_df = pd.DataFrame({
        'Disease': palette(tbl["disease"].tolist(), "tab20"),
        'Drug': palette(tbl["drug_disp"].tolist(), "tab20"),
        'Cell type': palette(tbl["cell"].tolist(), "tab20"),
    }, index=idx_labels)
    
    g = sns.clustermap(
        df, row_linkage=Z, col_linkage=Z, cmap="vlag",
        xticklabels=False, yticklabels=False, figsize=(15, 15),
        row_colors=row_colors_df, col_colors=row_colors_df,
        dendrogram_ratio=(0.12, 0.12), colors_ratio=(0.06, 0.06),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.patch.set_facecolor("white")
    g.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(g.fig)

def main():
    parser = argparse.ArgumentParser(description="Simple clustermap generator")
    parser.add_argument("--npz", required=True, help="Path to aligned NPZ file")
    parser.add_argument("--out_dir", default="clustermaps", help="Output directory")
    parser.add_argument("--max_rows", type=int, default=1500, help="Max samples to plot")
    parser.add_argument("--linkage", default="average", help="Linkage method")
    args = parser.parse_args()
    
    # load data
    z, tbl, keep_mask = load_and_prepare_data(args.npz)
    
    # sample data
    sample_idx = stratified_sample(tbl, args.max_rows)
    tbl_sampled = tbl.iloc[sample_idx].reset_index(drop=True)
    
    out_dir = Path(args.out_dir)
    representations = ["net", "pca", "expr"]
    
    print(f"Generating clustermaps for {len(tbl_sampled)} samples...")
    
    for rep in representations:
        try:
            X_full = get_features(z, keep_mask, rep)
            X_sampled = X_full[sample_idx]
            out_path = out_dir / f"clustermap_{rep}.png"
            generate_clustermap(X_sampled, tbl_sampled, rep, out_path, args.linkage)
            print(f"Saved {rep}: {out_path}")
            
        except ValueError as e:
            print(f"Skipping {rep}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    sns.set(context="notebook", style="white")
    main()