"""
Label Cohesion Evaluation

  1) Expression             -> key: 'expr_raw'
  2) PCA Metagene           -> key: 'pca_metagene'
  3) Morgan fingerprints    -> key: 'morgan_fp'
  4) Network predictions    -> key: 'correlations'

Metrics
  - Silhouette
  - 1-NN Balanced Accuracy
  - k-NN label agreement
  - Pairwise AUC

Outputs:
  - JSON with metrics per modality at <output_dir>/label_cohesion_results.json
"""

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize

EXCLUDE_REGEXES = []          # e.g., [r'(?i)^\s*neoplasm\s*$']
MAX_PER_DISEASE = None        # e.g., 1000
MIN_DISEASE_SAMPLES = 10

K_LIST              = (1, 5, 10, 50)
MAX_PER_CLASS_EVAL  = 1000
SIL_SAMPLE_SIZE     = 10000
PAIRWISE_PAIRS      = 200000
EVAL_SEED           = 42

def load_aligned_npz(path: Path):
    """load aligned npz; returns dict."""
    try:
        z = np.load(path, allow_pickle=False)
        labels = z['disease_labels']
    except Exception:
        z = np.load(path, allow_pickle=True)
        labels = z['disease_labels'].astype(str)

    data = {
        'labels': np.asarray(labels, dtype=str),
        'pca_metagene': z['pca_metagene'],
        'correlations': z['correlations'],
        'contexts': z['contexts'],
        'expr_raw': z['expr_raw'] if 'expr_raw' in z.files else None,
        'morgan_fp': z['morgan_fp'] if 'morgan_fp' in z.files else None,
    }
    return data


def apply_exclusions_block(arr_dict, labels):
    keep = np.ones(len(labels), dtype=bool)

    # regex excludes
    if EXCLUDE_REGEXES:
        drop = np.zeros(len(labels), dtype=bool)
        for pat in EXCLUDE_REGEXES:
            rex = re.compile(pat, flags=re.IGNORECASE)
            drop |= np.array([bool(rex.search(lbl)) for lbl in labels], dtype=bool)
        keep &= ~drop

    def _mask(x, m):
        if x is None:
            return None
        return x[m]
    
    if not keep.all():
        labels = labels[keep]
        for k in list(arr_dict.keys()):
            arr_dict[k] = _mask(arr_dict[k], keep)

    # per-disease cap
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
        labels = labels[new_keep]
        for k in list(arr_dict.keys()):
            arr_dict[k] = _mask(arr_dict[k], new_keep)

    # min per-disease count
    if MIN_DISEASE_SAMPLES and MIN_DISEASE_SAMPLES > 1:
        counts = Counter(labels)
        ok = np.array([counts[d] >= MIN_DISEASE_SAMPLES for d in labels], dtype=bool)
        labels = labels[ok]
        for k in list(arr_dict.keys()):
            arr_dict[k] = _mask(arr_dict[k], ok)

    return arr_dict, labels


def prepare_network_features(corr):
    n, g, _ = corr.shape
    iu = np.triu_indices(g, k=1)
    return corr[:, iu[0], iu[1]]


def pairwise_random_distances(X, idx_i, idx_j, metric='euclidean', eps=1e-8):
    Xi, Xj = X[idx_i], X[idx_j]
    if metric == 'euclidean':
        return np.linalg.norm(Xi - Xj, axis=1)
    elif metric == 'cosine':
        # cosine similarity
        num = np.sum(Xi * Xj, axis=1)
        den = np.linalg.norm(Xi, axis=1) * np.linalg.norm(Xj, axis=1) + eps
        return 1.0 - (num / den)
    else:
        raise ValueError(f"Unsupported metric for pairwise distances: {metric}")


def silhouette_with_sampling(X, y_enc, max_samples, seed, metric='euclidean'):
    if len(X) < 2 or len(np.unique(y_enc)) < 2:
        return float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        sel = rng.choice(len(X), size=max_samples, replace=False)
        Xs, ys = X[sel], y_enc[sel]
    else:
        Xs, ys = X, y_enc

    s_overall = float(silhouette_score(Xs, ys, metric=metric))
    s_samples = silhouette_samples(Xs, ys, metric=metric)
    s_macro = float(np.mean([s_samples[ys == lab].mean() for lab in np.unique(ys)]))
    return s_overall, s_macro


def eval_representation(X, labels, indices, name, metric='euclidean'):
    """compute label-cohesion metrics"""
    Xs = X[indices]
    ys = np.asarray(labels)[indices]

    # encode labels
    le = LabelEncoder().fit(ys)
    y_enc = le.transform(ys)

    # silhouette
    sil_overall, sil_macro = silhouette_with_sampling(
        Xs, y_enc, max_samples=SIL_SAMPLE_SIZE, seed=EVAL_SEED, metric=metric
    )

    # kNN
    knn_stats = {}
    kmax = min(max(K_LIST) + 1, len(Xs))
    if kmax >= 2 and len(np.unique(ys)) >= 2:
        algo = 'brute' if metric == 'cosine' else 'auto'
        nn = NearestNeighbors(n_neighbors=kmax, metric=metric, algorithm=algo).fit(Xs)
        _, nn_idx = nn.kneighbors(Xs, return_distance=True)
        neigh_labels = ys[nn_idx[:, 1:]]  # drop self

        for k in K_LIST:
            k_eff = min(k, neigh_labels.shape[1])
            agree = (neigh_labels[:, :k_eff] == ys[:, None]).mean(axis=1)
            knn_stats[f'knn@{k}_micro'] = float(agree.mean())
            knn_stats[f'knn@{k}_macro'] = float(np.mean([agree[ys == lab].mean() for lab in np.unique(ys)]))

        pred_1nn = neigh_labels[:, 0]
        bal_acc_1nn = float(balanced_accuracy_score(ys, pred_1nn))
    else:
        bal_acc_1nn = float('nan')
        for k in K_LIST:
            knn_stats[f'knn@{k}_micro'] = float('nan')
            knn_stats[f'knn@{k}_macro'] = float('nan')

    # pairwise AUC
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
            d = pairwise_random_distances(Xs, i, j, metric=metric)
            same = (ys[i] == ys[j]).astype(int)
            if np.unique(same).size == 2:
                auc_val = float(roc_auc_score(same, -d))  # higher is better

    result = OrderedDict(
        name=name,
        metric=metric,
        n_used=int(len(Xs)),
        n_labels=int(len(np.unique(ys))),
        silhouette_label=sil_overall,
        silhouette_label_macro=sil_macro,
        balanced_acc_1NN=bal_acc_1nn,
        pairwise_auc_same_vs_diff=auc_val,
    )
    result.update(knn_stats)
    return result

def print_knn(res):
    for k in K_LIST:
        mi = res.get(f'knn@{k}_micro', float('nan'))
        ma = res.get(f'knn@{k}_macro', float('nan'))
        print(f"  kNN@{k} label agreement: micro={mi:.4f} | macro={ma:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate label cohesion across raw, pca, morgan, network)")
    ap.add_argument("--aligned_data", required=True, help="Path to simple_aligned_predictions.npz")
    ap.add_argument("--output_dir", default="label_cohesion_results", help="Directory to save results")
    args = ap.parse_args()

    aligned_path = Path(args.aligned_data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LABEL COHESION")
    print("=" * 80)
    print(f"Aligned data: {aligned_path}")
    print(f"Output dir : {out_dir}")

    # load
    D = load_aligned_npz(aligned_path)
    labels = D['labels']

    # exclusions, caps, min count
    arrs = {
        'expr_raw': D['expr_raw'],
        'pca_metagene': D['pca_metagene'],
        'correlations': D['correlations'],
        'morgan_fp': D['morgan_fp'],
    }
    arrs, labels = apply_exclusions_block(arrs, labels)

    # if len(np.unique(labels)) < 2:
    #     print("\n Not enough distinct diseases after filtering. Exiting.")
    #     return

    print("\nPreparing features...")
    net_feats = prepare_network_features(arrs['correlations']).astype(np.float32)
    net_scaled = StandardScaler().fit_transform(net_feats)
    pca_metagene_scaled = StandardScaler().fit_transform(arrs['pca_metagene'].astype(np.float32))
    raw_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(arrs['expr_raw'].astype(np.float32))
    fp_norm = normalize(arrs['morgan_fp'].astype(np.float32), norm='l2', axis=1, copy=False)

    sel_idx = []
    rng = np.random.default_rng(EVAL_SEED)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) > MAX_PER_CLASS_EVAL:
            idx = rng.choice(idx, size=MAX_PER_CLASS_EVAL, replace=False)
        sel_idx.append(idx)
    sel_idx = np.concatenate(sel_idx)

    # eval
    results = OrderedDict()

    res_raw = eval_representation(raw_scaled, labels, sel_idx, "Expression (RAW)", metric='euclidean')
    results['expression_raw'] = res_raw
    print(f"\n[RAW]  n_used={res_raw['n_used']}  silhouette={res_raw['silhouette_label']:.4f}  1NN-BA={res_raw['balanced_acc_1NN']:.4f}  AUC={res_raw['pairwise_auc_same_vs_diff']:.4f}")
    print_knn(res_raw)

    res_pca = eval_representation(pca_metagene_scaled, labels, sel_idx, "PCA Metagene", metric='euclidean')
    results['pca_metagene'] = res_pca
    print(f"\n[PCA]  n_used={res_pca['n_used']}  silhouette={res_pca['silhouette_label']:.4f}  1NN-BA={res_pca['balanced_acc_1NN']:.4f}  AUC={res_pca['pairwise_auc_same_vs_diff']:.4f}")
    print_knn(res_pca)

    res_net = eval_representation(net_scaled, labels, sel_idx, "Network", metric='euclidean')
    results['network'] = res_net
    print(f"\n[NET]  n_used={res_net['n_used']}  silhouette={res_net['silhouette_label']:.4f}  1NN-BA={res_net['balanced_acc_1NN']:.4f}  AUC={res_net['pairwise_auc_same_vs_diff']:.4f}")
    print_knn(res_net)

    res_fp = eval_representation(fp_norm, labels, sel_idx, "Morgan Fingerprint", metric='cosine')
    results['fingerprint_morgan'] = res_fp
    print(f"\n[FP ]  n_used={res_fp['n_used']}  silhouette={res_fp['silhouette_label']:.4f}  1NN-BA={res_fp['balanced_acc_1NN']:.4f}  AUC={res_fp['pairwise_auc_same_vs_diff']:.4f}")
    print_knn(res_fp)

    # save
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
        'results': results,
        'modalities_present': {
            'expr_raw': bool(raw_scaled is not None),
            'pca_metagene': True,
            'network': True,
            'morgan_fp': bool(fp_norm is not None),
        }
    }
    out_file = out_dir / "label_cohesion_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n saved results to: {out_file}")


if __name__ == "__main__":
    main()