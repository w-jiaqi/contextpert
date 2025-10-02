"""
Drug to Genetic Top-K Retrieval

Inputs: aligned_predictions_topk.npz
  - expr_raw (optional)     -> "raw"
  - retrieval_pca_metagene  -> "expr"
  - retrieval_net_corr      -> "net"

A query (drug) is a "hit" at rank r if any of the top-K neighbors in the
gallery (genetic perturbations) share the same __target_key.

Outputs:
  metrics_summary.json
  per_cell_metrics_<repr>.csv
  per_target_metrics_<repr>.csv
"""

import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import balanced_accuracy_score


def default_metadata_path(npz_path: Path) -> Path:
    return npz_path.parent / "aligned_metadata.parquet"


def load_views(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)

    # raw
    raw = z['expr_raw'].astype(np.float32) if 'expr_raw' in z.files else None

    # expr (pca metagene)
    if 'retrieval_pca_metagene' in z.files:
        expr = z['retrieval_pca_metagene'].astype(np.float32)
    elif 'pca_metagene' in z.files:
        expr = z['pca_metagene'].astype(np.float32)
    else:
        raise ValueError("PCA metagene view missing (retrieval_pca_metagene / pca_metagene).")

    # net
    if 'retrieval_net_corr' in z.files:
        net = z['retrieval_net_corr'].astype(np.float32)
    elif 'correlations' in z.files:
        corr = z['correlations']
        if corr.ndim == 3:
            g = corr.shape[1]
            iu = np.triu_indices(g, k=1)
            net = corr[:, iu[0], iu[1]].astype(np.float32)
        elif corr.ndim == 2:
            net = corr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected correlations shape: {corr.shape}")
    else:
        net = None

    tri_i = z['tri_upper_i'] if 'tri_upper_i' in z.files else None
    tri_j = z['tri_upper_j'] if 'tri_upper_j' in z.files else None

    return raw, expr, net, (tri_i, tri_j)


def build_target_keys(meta: pd.DataFrame) -> pd.Series:
    pt = meta['pert_type'].astype(str)

    drug_symbol_candidates = [
        'target_gene', 'target_gene_drug', 'target_symbol', 'targetSymbol',
        'primaryTargetSymbol', 'primaryTarget', 'drug_target_gene'
    ]
    gen_symbol_candidates = [
        'target_gene','target_symbol','targetSymbol','gene','Gene','symbol','pert_iname'
    ]

    def first_present(df, cols):
        for c in cols:
            if c in df.columns:
                return df[c].astype(str)
        return pd.Series([np.nan]*len(df))

    drug_gene = first_present(meta, drug_symbol_candidates)
    drug_id   = first_present(meta, ['targetId','target_id','targetID'])
    gen_gene  = first_present(meta, gen_symbol_candidates)

    key = pd.Series([np.nan]*len(meta), dtype=object)
    is_drug = pt.eq('trt_cp'); is_ctl = pt.str.startswith('ctl', na=False); is_gen = (~is_drug) & (~is_ctl)

    kg = drug_gene.where(drug_gene.str.strip().ne(''), np.nan)
    key = key.mask(is_drug, kg)
    fill = drug_id.where(drug_id.str.strip().ne(''), np.nan)
    key = key.where(~(is_drug & key.isna()), fill)

    kg2 = gen_gene.where(gen_gene.str.strip().ne(''), np.nan)
    key = key.mask(is_gen, kg2)

    key = key.astype(str).str.strip()
    key = key.mask((key == '') | (key.str.lower() == 'nan'), np.nan)
    key = key.str.upper()
    return key


def sanitize(X):
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def normalize_train_test(Xg: np.ndarray, Xq: np.ndarray, metric: str, mode: str | None):
    if mode in (None, "auto"):
        mode = "zscore" if metric == "euclidean" else "l2"
    if mode == "l2":
        return normalize(Xg, norm='l2', axis=1), normalize(Xq, norm='l2', axis=1)
    if mode == "zscore":
        sc = StandardScaler(with_mean=True, with_std=True)
        Xg_n = sc.fit_transform(Xg)
        Xq_n = sc.transform(Xq)
        return Xg_n, Xq_n
    if mode == "none":
        return Xg, Xq
    raise ValueError(f"Unknown normalize mode: {mode}")


def kneighbors(Xg: np.ndarray, Xq: np.ndarray, k: int, metric: str):
    if len(Xg) == 0:
        d = np.full((len(Xq), k), np.inf, dtype=float)
        i = np.full((len(Xq), k), -1, dtype=int)
        return d, i
    kk = min(k, len(Xg))
    nn = NearestNeighbors(n_neighbors=kk, metric=metric, algorithm='brute')
    nn.fit(Xg)
    d, i = nn.kneighbors(Xq, return_distance=True)
    if kk < k:
        pad_d = np.full((len(Xq), k-kk), np.inf)
        pad_i = np.full((len(Xq), k-kk), -1, dtype=int)
        d = np.hstack([d, pad_d])
        i = np.hstack([i, pad_i])
    return d, i


def metrics_from_neighbors_crossmodal(y_query: np.ndarray, y_gallery: np.ndarray, nbrs: np.ndarray, k_list):
    """
    Returns metrics averaged over all drugs
    """
    out = {}
    if nbrs.ndim != 2 or len(y_gallery) == 0:
        out['balanced_acc_1NN'] = float('nan')
        for K in k_list:
            out[f'precision@{K}_micro'] = 0.0
            out[f'precision@{K}_macro'] = 0.0
            out[f'hits@{K}'] = 0.0
            out[f'mrr@{K}'] = 0.0
        return out

    Nq, Kmax = nbrs.shape
    valid = (nbrs >= 0)
    safe_idx = np.where(valid, nbrs, 0)
    y_nbr = y_gallery[safe_idx]  # (Nq, Kmax)

    # 1-NN balanced accuracy
    rows_1nn = valid[:, 0]
    if rows_1nn.any() and len(np.unique(y_query[rows_1nn])) > 1:
        y_pred_1 = y_nbr[rows_1nn, 0]
        try:
            out['balanced_acc_1NN'] = float(balanced_accuracy_score(y_query[rows_1nn], y_pred_1))
        except Exception:
            out['balanced_acc_1NN'] = float('nan')
    else:
        out['balanced_acc_1NN'] = float('nan')

    for K in k_list:
        K_eff = min(K, Kmax)
        if K_eff == 0:
            out[f'precision@{K}_micro'] = 0.0
            out[f'precision@{K}_macro'] = 0.0
            out[f'hits@{K}'] = 0.0
            out[f'mrr@{K}'] = 0.0
            continue

        V = valid[:, :K_eff]
        agree = (y_nbr[:, :K_eff] == y_query[:, None]) & V  # (Nq, K_eff)

        # precision@K
        micro = (agree.sum(axis=1) / float(K_eff)).mean()

        macro_vals = []
        for lab in np.unique(y_query):
            m = (y_query == lab)
            if m.any():
                macro_vals.append((agree[m].sum(axis=1) / float(K_eff)).mean())
        macro = float(np.mean(macro_vals)) if macro_vals else 0.0

        # hits@K
        hits = agree.any(axis=1).mean()

        # MRR@K 
        first_pos = np.argmax(agree, axis=1) + 1
        has_pos = agree.any(axis=1)
        recip = np.zeros(Nq, dtype=np.float64)
        recip[has_pos] = 1.0 / first_pos[has_pos]
        mrr = float(recip.mean())

        out[f'precision@{K}_micro'] = float(micro)
        out[f'precision@{K}_macro'] = float(macro)
        out[f'hits@{K}'] = float(hits)
        out[f'mrr@{K}'] = float(mrr)

    return out


def filter_min_and_cap(meta: pd.DataFrame, label_col: str, minimum: int) -> pd.DataFrame:
    if minimum <= 1:
        return meta.copy()
    vc = meta[label_col].value_counts()
    keep = vc[vc >= minimum].index
    return meta[meta[label_col].isin(keep)].copy()


def strict_single_target_filter(q_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only drug rows where 'pert_id' maps to exactly one __target_key 
    across the dataset (not per-cell).
    """
    if 'pert_id' not in q_df.columns:
        return q_df.copy()
    grp = q_df.groupby('pert_id')['__target_key'].nunique(dropna=True)
    single = set(grp[grp == 1].index)
    return q_df[q_df['pert_id'].isin(single)].copy()


def print_breakdown(df: pd.DataFrame, title: str, max_rows: int = 50):
    print(f"\n{title}")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)-max_rows} more rows; full table saved to csv)")
    else:
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Drug to Genetic Top-K retrieval (raw / expr / net) with breakdowns")
    ap.add_argument("--aligned_data", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--reprs", nargs="+", default=["raw","expr","net"],
                    choices=["raw","expr","net"])
    ap.add_argument("--within_cell", default="True", choices=["True","False"])
    ap.add_argument("--include_ligands", default="False", choices=["True","False"])
    ap.add_argument("--strict_single_target", default="True", choices=["True","False"])
    ap.add_argument("--min_support_per_cell", type=int, default=3)
    ap.add_argument("--metric", default="euclidean", choices=["auto","cosine","euclidean"])
    ap.add_argument("--normalize", default="zscore", choices=["auto","l2","zscore","none"])
    ap.add_argument("--k_list", default="1,5,10,50")
    ap.add_argument("--max_neighbors", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_targets", default="", help="Comma-separated list of target symbols (upper/lower ok) to exclude from both query and gallery, e.g. 'SMO,KCNJ11,IMPDH1'")
    args = ap.parse_args()

    npz_path = Path(args.aligned_data)
    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.metadata) if args.metadata else default_metadata_path(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found at {meta_path}")

    within_cell = args.within_cell == "True"
    include_ligands = args.include_ligands == "True"
    strict_single = args.strict_single_target == "True"
    metric = "euclidean" if args.metric in (None, "auto") else args.metric
    norm_mode = args.normalize if args.normalize is not None else ("zscore" if metric == "euclidean" else "l2")
    k_list = [int(k.strip()) for k in args.k_list.split(",") if k.strip()]
    K = max(k_list + [args.max_neighbors])

    print("="*80)
    print("DRUG to GENETIC TOP-K")
    print("="*80)
    print(f"NPZ      : {npz_path}")
    print(f"Metadata : {meta_path}")
    print(f"Output   : {out_dir}")
    print(f"Reprs    : {args.reprs}")
    print(f"Metric   : {metric}  (normalize={norm_mode})")
    print(f"Within-cell: {within_cell} | Include ligands: {include_ligands}")
    print(f"Filters  : strict_single_target={strict_single} | min_support_per_cell={args.min_support_per_cell}")
    print(f"K list   : {k_list}  (max_neighbors={args.max_neighbors})\n")

    # load arrays & metadata
    raw, expr, net, tri_idx = load_views(npz_path)

    # base metadata written by training
    base = pd.read_parquet(meta_path).reset_index(drop=True)

    # ensure __target_key
    if '__target_key' in base.columns:
        base['__target_key'] = (
            base['__target_key'].astype(str).str.strip()
            .replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
            .str.upper()
        )
    else:
        base['__target_key'] = build_target_keys(base)

    # length check vs npz
    N = expr.shape[0]
    if len(base) != N:
        raise ValueError(f"Metadata rows ({len(base)}) != NPZ rows ({N}).")

    meta = base.copy()
    meta['global_row'] = np.arange(len(meta), dtype=np.int64)

    # define
    pt = meta['pert_type'].astype(str)
    is_drug = pt.eq('trt_cp')
    is_ctl  = pt.str.startswith('ctl', na=False)
    is_gen  = (~is_drug) & (~is_ctl)
    if not include_ligands:
        is_gen &= ~pt.str.startswith('trt_lig', na=False)
    has_key = meta['__target_key'].notna()

    q_df = meta[is_drug & has_key].copy()
    g_df = meta[is_gen  & has_key].copy()

    # (restrict to labels present on both sides?)
    common = np.intersect1d(q_df['__target_key'].unique(), g_df['__target_key'].unique())
    q_df = q_df[q_df['__target_key'].isin(common)].copy()
    g_df = g_df[g_df['__target_key'].isin(common)].copy()

    # exclude specific targets (testing only)
    excluded_targets = []
    if args.exclude_targets:
        excl = {t.strip().upper() for t in args.exclude_targets.split(",") if t.strip()}
        if excl:
            before_labels = set(common)
            # apply to both sides
            q_df = q_df[~q_df['__target_key'].isin(excl)].copy()
            g_df = g_df[~g_df['__target_key'].isin(excl)].copy()
            # recompute common after exclusion
            common = np.intersect1d(q_df['__target_key'].unique(), g_df['__target_key'].unique())
            q_df = q_df[q_df['__target_key'].isin(common)].copy()
            g_df = g_df[g_df['__target_key'].isin(common)].copy()
            excluded_targets = sorted(before_labels.intersection(excl))
            print(f"Excluded targets: {excluded_targets if excluded_targets else 'none matched'}")
            print("After exclusion -> queries:", len(q_df), "gallery:", len(g_df), "unique labels:", len(common))

    # strict single-target for drugs
    if strict_single:
        q_df = strict_single_target_filter(q_df)

    if args.min_support_per_cell > 1 and within_cell:
        parts_q = []
        parts_g = []
        for cell, q_c in q_df.groupby('cell_id'):
            g_c = g_df[g_df['cell_id'] == cell]
            if len(q_c) == 0:
                continue
            q_c2 = filter_min_and_cap(q_c, '__target_key', args.min_support_per_cell)
            g_c2 = filter_min_and_cap(g_c, '__target_key', args.min_support_per_cell)

            common_c = np.intersect1d(q_c2['__target_key'].unique(), g_c2['__target_key'].unique())
            parts_q.append(q_c2[q_c2['__target_key'].isin(common_c)])
            parts_g.append(g_c2[g_c2['__target_key'].isin(common_c)])
        q_df = pd.concat(parts_q).reset_index(drop=True) if parts_q else q_df.iloc[0:0]
        g_df = pd.concat(parts_g).reset_index(drop=True) if parts_g else g_df.iloc[0:0]

    if len(q_df) == 0 or len(g_df) == 0:
        raise RuntimeError("After global filters, queries or gallery are empty.")

    if within_cell and 'cell_id' not in meta.columns:
        warnings.warn("within_cell=True but 'cell_id' not found")
        within_cell = False

    def take_rows(X, idx):
        return None if X is None else X[idx]

    # prepare representations
    reps = {}
    if "raw" in args.reprs and raw is not None: reps["raw"] = raw
    if "expr" in args.reprs: reps["expr"] = expr
    if "net" in args.reprs and net is not None: reps["net"] = net
    if not reps:
        raise RuntimeError("No valid representations requested/found (raw missing? net missing?).")

    print(f"After global filters -> queries: {len(q_df)} gallery: {len(g_df)} unique labels: {len(common)}")
    if within_cell:
        sizes = g_df.groupby('cell_id').size()
        if not sizes.empty:
            print(f"Gallery per-cell size: mean={sizes.mean():.1f}, median={sizes.median():.0f}, min={sizes.min()}, max={sizes.max()}")
    print("")

    summary = {
        "config": {
            "metric": metric,
            "normalize": norm_mode,
            "k_list": k_list,
            "within_cell": within_cell,
            "include_ligands": include_ligands,
            "strict_single_target": strict_single,
            "min_support_per_cell": args.min_support_per_cell,
            "excluded_targets": excluded_targets,
        },
        "counts": {
            "queries": int(len(q_df)),
            "gallery": int(len(g_df)),
            "targets": int(len(common)),
        },
        "representations": {}
    }

    rng = np.random.default_rng(args.seed)

    for name, X in reps.items():
        print(f"=== {name} ===")

        q_idx = q_df['global_row'].to_numpy()
        g_idx = g_df['global_row'].to_numpy()
        Xq = sanitize(take_rows(X, q_idx))
        Xg = sanitize(take_rows(X, g_idx))
        yq = q_df['__target_key'].astype(str).values
        yg = g_df['__target_key'].astype(str).values

        # normalize
        Xg_n, Xq_n = normalize_train_test(Xg, Xq, metric=metric, mode=norm_mode)

        # aggregate
        metrics_accum = []      # list of (n_queries_cell, metrics_dict)
        pc_rows = []            # per-cell breakdown rows
        pt_rows = []            # per-target breakdown rows

        if not within_cell:
            # global KNN
            d, i = kneighbors(Xg_n, Xq_n, k=args.max_neighbors, metric=metric)
            m_global = metrics_from_neighbors_crossmodal(yq, yg, i, k_list)
            summary["representations"][name] = m_global

            # Per-target breakdown (global)
            for tgt, q_block in q_df.groupby('__target_key'):
                mask = (q_df['__target_key'] == tgt).values
                if mask.sum() == 0:
                    continue
                d_t = d[mask]; i_t = i[mask]
                yq_t = yq[mask]
                mt = metrics_from_neighbors_crossmodal(yq_t, yg, i_t, k_list)
                row = {"repr": name, "target": tgt, "n_queries": int(mask.sum())}
                for K in k_list:
                    row.update({
                        f"P@{K}": mt[f"precision@{K}_micro"],
                        f"Hits@{K}": mt[f"hits@{K}"],
                        f"MRR@{K}": mt[f"mrr@{K}"],
                    })
                pt_rows.append(row)

            pc_df = pd.DataFrame(columns=["repr","cell_id","n_queries","n_gallery"] + [f"P@{K}" for K in k_list] + [f"Hits@{K}" for K in k_list] + [f"MRR@{K}" for K in k_list])

        else:
            # Per-cell evaluation
            for cell, q_block in q_df.groupby('cell_id'):
                g_block = g_df[g_df['cell_id'] == cell]
                q_rows = q_block['global_row'].to_numpy()
                g_rows = g_block['global_row'].to_numpy()

                if len(q_rows) == 0:
                    continue

                mask_q = np.isin(q_idx, q_rows)
                Xq_c = Xq_n[mask_q]

                if len(g_rows) == 0:
                    i_local = np.full((len(Xq_c), args.max_neighbors), -1, dtype=int)
                    d_local = np.full((len(Xq_c), args.max_neighbors), np.inf, dtype=float)
                    yg_c = np.array([], dtype=str)
                    yq_c = q_block['__target_key'].astype(str).values
                else:
                    mask_g = np.isin(g_idx, g_rows)
                    Xg_c = Xg_n[mask_g]
                    d_local, i_local = kneighbors(Xg_c, Xq_c, k=args.max_neighbors, metric=metric)
                    yg_c = g_block['__target_key'].astype(str).values
                    yq_c = q_block['__target_key'].astype(str).values

                m_c = metrics_from_neighbors_crossmodal(yq_c, yg_c, i_local, k_list)
                metrics_accum.append((len(yq_c), m_c))

                # Per-cell breakdown row
                row_c = {"repr": name, "cell_id": str(cell), "n_queries": int(len(yq_c)), "n_gallery": int(len(yg_c))}
                for K in k_list:
                    row_c.update({
                        f"P@{K}": m_c[f"precision@{K}_micro"],
                        f"Hits@{K}": m_c[f'hits@{K}'],
                        f"MRR@{K}": m_c[f'mrr@{K}'],
                    })
                pc_rows.append(row_c)

                # Per-target-in-cell breakdown
                if len(yq_c) > 0 and len(yg_c) > 0:
                    for tgt in np.unique(yq_c):
                        m_t = (yq_c == tgt)
                        if not m_t.any():
                            continue
                        i_t = i_local[m_t]
                        yq_t = yq_c[m_t]
                        mt = metrics_from_neighbors_crossmodal(yq_t, yg_c, i_t, k_list)
                        row_t = {
                            "repr": name,
                            "cell_id": str(cell),
                            "target": str(tgt),
                            "n_queries": int(m_t.sum()),
                            "n_gallery": int((yg_c == tgt).sum()),
                        }
                        for K in k_list:
                            row_t.update({
                                f"P@{K}": mt[f"precision@{K}_micro"],
                                f"Hits@{K}": mt[f'hits@{K}'],
                                f"MRR@{K}": mt[f'mrr@{K}'],
                            })
                        pt_rows.append(row_t)

            # aggregate metrics across cells
            if metrics_accum:
                agg = {}
                keys = list(metrics_accum[0][1].keys())
                for k in keys:
                    vals, weights = [], []
                    for w, m in metrics_accum:
                        v = m.get(k, float('nan'))
                        if not (isinstance(v, float) and np.isnan(v)):
                            vals.append(v); weights.append(w)
                    agg[k] = float(np.average(np.array(vals, dtype=float), weights=np.array(weights, dtype=float))) if weights else float('nan')
                summary["representations"][name] = agg
            else:
                summary["representations"][name] = {}

            pc_df = pd.DataFrame(pc_rows)

        if not pc_df.empty:
            pc_out = out_dir / f"per_cell_metrics_{name}.csv"
            pc_df.to_csv(pc_out, index=False)
            print_breakdown(pc_df[["repr","cell_id","n_queries","n_gallery"] + [f"P@{K}" for K in k_list] + [f"Hits@{K}" for K in k_list] + [f"MRR@{K}" for K in k_list]],
                            title=f"Per-cell breakdown ({name})")

        pt_df = pd.DataFrame(pt_rows)
        if not pt_df.empty:
            agg_cols = {f"P@{K}": "mean" for K in k_list}
            agg_cols.update({f"Hits@{K}": "mean" for K in k_list})
            agg_cols.update({f"MRR@{K}": "mean" for K in k_list})

            def _wavg(g, col):
                return np.average(g[col].values, weights=g["n_queries"].values) if g["n_queries"].sum() > 0 else np.nan
            rows = []
            for tgt, g in pt_df.groupby("target"):
                row = {"repr": name, "target": tgt, "n_queries": int(g["n_queries"].sum())}
                for K in k_list:
                    row[f"P@{K}"] = float(_wavg(g, f"P@{K}"))
                    row[f"Hits@{K}"] = float(_wavg(g, f"Hits@{K}"))
                    row[f"MRR@{K}"] = float(_wavg(g, f"MRR@{K}"))
                rows.append(row)
            pt_agg = pd.DataFrame(rows)
        else:
            pt_agg = pd.DataFrame(columns=["repr","target","n_queries"] + [f"P@{K}" for K in k_list] + [f"Hits@{K}" for K in k_list] + [f"MRR@{K}" for K in k_list])

        pt_out = out_dir / f"per_target_metrics_{name}.csv"
        pt_agg.sort_values("n_queries", ascending=False).to_csv(pt_out, index=False)
        print_breakdown(pt_agg.sort_values("n_queries", ascending=False)[["repr","target","n_queries"] + [f"P@{K}" for K in k_list] + [f"Hits@{K}" for K in k_list] + [f"MRR@{K}" for K in k_list]],
                        title=f"Per-target breakdown ({name})")

        m = summary["representations"][name]
        if m:
            ba = m.get('balanced_acc_1NN', float('nan'))
            print(f"[{name}]  1-NN BA: {ba if not (isinstance(ba,float) and np.isnan(ba)) else float('nan'):.4f}")
        for kk in k_list:
                print(f"  P@{kk} micro={m.get(f'precision@{kk}_micro', float('nan')):.4f} | "
                      f"macro={m.get(f'precision@{kk}_macro', float('nan')):.4f} | "
                      f"Hits@{kk}={m.get(f'hits@{kk}', float('nan')):.4f} | "
                      f"MRR@{kk}={m.get(f'mrr@{kk}', float('nan')):.4f}")
        print("")

    out_json = out_dir / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_json}")
    print("Saved per-cell/per-target csv at:", out_dir)


if __name__ == "__main__":
    main()