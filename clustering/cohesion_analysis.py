"""
Label Cohesion Evaluation (Expression vs. Network)

  - Silhouette (labels-as-clusters), overall and macro (per-label average)
  - 1-NN Balanced Accuracy (predict label by nearest neighbor)
  - k-NN label agreement (micro & macro) at K in {1,5,10,50}
  - Pairwise AUC (prob. same-disease pairs are closer than different-disease pairs)

Inputs: aligned NPZ produced by training script:
  - expressions: (N, d_expr)
  - correlations: (N, G, G)
  - contexts: (N, d_ctx)
  - disease_labels: (N,) strings

Outputs:
  - Prints metrics for Expression and Network
  - Saves a JSON file with all metrics to <output_dir>/label_cohesion_results.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

EXCLUDE_REGEXES = []
MAX_PER_DISEASE = None
MIN_DISEASE_SAMPLES = 10

K_LIST              = (1, 5, 10, 50)
MAX_PER_CLASS_EVAL  = 1000
SIL_SAMPLE_SIZE     = 10000
PAIRWISE_PAIRS      = 200000
EVAL_SEED           = 42

def load_aligned_npz(path: Path):
    try:
        z = np.load(path, allow_pickle=False)
        labels = z['disease_labels']
    except Exception:
        z = np.load(path, allow_pickle=True)
        labels = z['disease_labels'].astype(str)
    return z['expressions'], z['correlations'], z['contexts'], np.asarray(labels, dtype=str)


def apply_exclusions(expressions, correlations, contexts, labels):
    keep = np.ones(len(labels), dtype=bool)
    if EXCLUDE_REGEXES:
        drop = np.zeros(len(labels), dtype=bool)
        for pat in EXCLUDE_REGEXES:
            rex = re.compile(pat, flags=re.IGNORECASE)
            drop |= np.array([bool(rex.search(lbl)) for lbl in labels], dtype=bool)
        keep &= ~drop

    if not keep.all():
        expressions  = expressions[keep]
        correlations = correlations[keep]
        contexts     = contexts[keep]
        labels       = labels[keep]

    if MAX_PER_DISEASE is not None and MAX_PER_DISEASE > 0:
        rng = np.random.default_rng(EVAL_SEED)
        new_keep = np.zeros(len(labels), dtype=bool)
        for d in np.unique(labels):
            idx = np.where(labels == d)[0]
            if len(idx) <= MAX_PER_DISEASE:
                new_keep[idx] = True
            else:
                pick = rng.choice(idx, size=MAX_PER_DISEASE, replace=False)
                new_keep[pick] = True
        expressions  = expressions[new_keep]
        correlations = correlations[new_keep]
        contexts     = contexts[new_keep]
        labels       = labels[new_keep]

    if MIN_DISEASE_SAMPLES and MIN_DISEASE_SAMPLES > 1:
        counts = Counter(labels)
        valid = np.array([counts[d] >= MIN_DISEASE_SAMPLES for d in labels], dtype=bool)
        expressions  = expressions[valid]
        correlations = correlations[valid]
        contexts     = contexts[valid]
        labels       = labels[valid]

    return expressions, correlations, contexts, labels


def prepare_network_features(correlations):
    n, g, _ = correlations.shape
    iu = np.triu_indices(g, k=1)
    return correlations[:, iu[0], iu[1]]


def stratified_indices(labels, max_per_class=1000, seed=42):
    rng = np.random.default_rng(seed)
    idxs = []
    for lab in np.unique(labels):
        lab_idx = np.where(labels == lab)[0]
        if len(lab_idx) > max_per_class:
            lab_idx = rng.choice(lab_idx, size=max_per_class, replace=False)
        idxs.append(lab_idx)
    return np.concatenate(idxs)


def silhouette_with_sampling(X, y_enc, max_samples, seed):
    if len(X) < 2 or len(np.unique(y_enc)) < 2:
        return float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        sel = rng.choice(len(X), size=max_samples, replace=False)
        Xs, ys = X[sel], y_enc[sel]
    else:
        Xs, ys = X, y_enc

    s_overall = float(silhouette_score(Xs, ys, metric='euclidean'))
    s_samples = silhouette_samples(Xs, ys, metric='euclidean')
    s_macro = float(np.mean([s_samples[ys == lab].mean() for lab in np.unique(ys)]))
    
    return s_overall, s_macro


def eval_representation(X, labels, indices, name):
    Xs = X[indices]
    ys = np.asarray(labels)[indices]

    le = LabelEncoder().fit(ys)
    y_enc = le.transform(ys)

    # Silhouette
    sil_overall, sil_macro = silhouette_with_sampling(Xs, y_enc, max_samples=SIL_SAMPLE_SIZE, seed=EVAL_SEED)

    # kNN neighbor label agreement
    kmax = min(max(K_LIST) + 1, len(Xs))  # +1 to drop self
    if kmax < 2 or len(np.unique(ys)) < 2:
        knn_stats = {f'knn@{k}_micro': float('nan') for k in K_LIST}
        knn_stats.update({f'knn@{k}_macro': float('nan') for k in K_LIST})
        bal_acc_1nn = float('nan')
    else:
        nn = NearestNeighbors(n_neighbors=kmax, metric='euclidean').fit(Xs)
        _, nn_idx = nn.kneighbors(Xs, return_distance=True)
        neigh_labels = ys[nn_idx[:, 1:]]  # drop self

        knn_stats = {}
        for k in K_LIST:
            k_eff = min(k, neigh_labels.shape[1])
            agree = (neigh_labels[:, :k_eff] == ys[:, None]).mean(axis=1)
            knn_stats[f'knn@{k}_micro'] = float(agree.mean())
            knn_stats[f'knn@{k}_macro'] = float(np.mean([agree[ys == lab].mean() for lab in np.unique(ys)]))

        pred_1nn = neigh_labels[:, 0]
        bal_acc_1nn = float(balanced_accuracy_score(ys, pred_1nn))

    # Pairwise AUC
    M_max = len(Xs) * (len(Xs) - 1) // 2
    M = int(min(PAIRWISE_PAIRS, max(1, M_max)))
    auc_val = float('nan')
    if len(np.unique(ys)) >= 2 and M > 1:
        rng = np.random.default_rng(EVAL_SEED)
        i = rng.integers(0, len(Xs), size=M)
        j = rng.integers(0, len(Xs), size=M)
        mask = i != j
        i, j = i[mask], j[mask]
        if len(i) > 1:
            d = np.linalg.norm(Xs[i] - Xs[j], axis=1)
            same = (ys[i] == ys[j]).astype(int)
            if np.unique(same).size == 2:
                auc_val = float(roc_auc_score(same, -d))

    result = {
        'name': name,
        'n_used': int(len(Xs)),
        'n_labels': int(len(np.unique(ys))),
        'silhouette_label': sil_overall,
        'silhouette_label_macro': sil_macro,
        'balanced_acc_1NN': bal_acc_1nn,
        'pairwise_auc_same_vs_diff': auc_val,
    }
    result.update(knn_stats)
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate label cohesion for Expression vs. Network representations")
    ap.add_argument("--aligned_data", required=True, help="Path to simple_aligned_predictions.npz")
    ap.add_argument("--output_dir", default="label_cohesion_results", help="Directory to save JSON results")
    args = ap.parse_args()

    aligned_path = Path(args.aligned_data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LABEL COHESION EVALUATION")
    print("=" * 80)
    print(f"Aligned data: {aligned_path}")
    print(f"Output dir : {out_dir}")

    expressions, correlations, contexts, labels = load_aligned_npz(aligned_path)
    print(f"\nLoaded arrays:")
    print(f"  • expressions : {expressions.shape}")
    print(f"  • correlations: {correlations.shape}")
    print(f"  • contexts    : {contexts.shape}")
    print(f"  • labels      : {len(labels)} (unique={len(set(labels))})")
    print(f"  • top-10 labels: {dict(Counter(labels).most_common(10))}")

    expressions, correlations, contexts, labels = apply_exclusions(expressions, correlations, contexts, labels)
    if len(np.unique(labels)) < 2:
        print("\nNot enough distinct diseases after filtering. Exiting.")
        return

    print(f"\nAfter filtering:")
    print(f"  • samples: {len(labels)}")
    print(f"  • labels : {len(set(labels))}")
    print(f"  • top-10 : {dict(Counter(labels).most_common(10))}")

    # prepare features
    print("\nPreparing features...")
    net_feats = prepare_network_features(correlations)
    scaler_expr = StandardScaler()
    scaler_net = StandardScaler()
    expr_scaled = scaler_expr.fit_transform(expressions)
    net_scaled = scaler_net.fit_transform(net_feats)

    # use the same stratified subset for a fair comparison
    sel_idx = stratified_indices(labels, max_per_class=MAX_PER_CLASS_EVAL, seed=EVAL_SEED)
    print(f"\nUsing stratified subset: {len(sel_idx)} samples "
          f"(<= {MAX_PER_CLASS_EVAL} per disease)")

    # evaluate
    print("\n=== Evaluating label cohesion (no clustering) ===")
    expr_res = eval_representation(expr_scaled, labels, sel_idx, "Expression")
    net_res  = eval_representation(net_scaled,  labels, sel_idx, "Network")

    def pretty(res):
        print(f"\n[{res['name']}]  n_used={res['n_used']}  n_labels={res['n_labels']}")
        print(f"  Silhouette (labels-as-clusters): {res['silhouette_label']:.4f} "
              f"| macro: {res['silhouette_label_macro']:.4f}")
        print(f"  1-NN balanced accuracy       : {res['balanced_acc_1NN']:.4f}")
        print(f"  Pairwise AUC (same closer)   : {res['pairwise_auc_same_vs_diff']:.4f}")
        for k in K_LIST:
            mi = res.get(f'knn@{k}_micro', float('nan'))
            ma = res.get(f'knn@{k}_macro', float('nan'))
            print(f"  kNN@{k} label agreement      : micro={mi:.4f} | macro={ma:.4f}")

    pretty(expr_res)
    pretty(net_res)

    # Save results
    out = {
        'config': {
            'EXCLUDE_REGEXES': EXCLUDE_REGEXES,
            'MAX_PER_DISEASE': MAX_PER_DISEASE,
            'MIN_DISEASE_SAMPLES': MIN_DISEASE_SAMPLES,
            'K_LIST': K_LIST,
            'MAX_PER_CLASS_EVAL': MAX_PER_CLASS_EVAL,
            'SIL_SAMPLE_SIZE': SIL_SAMPLE_SIZE,
            'PAIRWISE_PAIRS': PAIRWISE_PAIRS,
            'EVAL_SEED': EVAL_SEED,
        },
        'dataset_after_filtering': {
            'n_samples': int(len(labels)),
            'n_labels': int(len(set(labels))),
            'top10_distribution': dict(Counter(labels).most_common(10)),
        },
        'expression': expr_res,
        'network': net_res,
    }
    out_file = out_dir / "label_cohesion_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
