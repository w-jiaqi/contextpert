"""
Top-K Training & Aligned Prediction

Output:
  - training_data.npz
  - training_results.npz
  - transforms.npz
  - aligned_predictions_topk.npz
  - aligned_metadata.parquet
  - counts_disease.csv, counts_drug.csv
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from contextualized.baselines.networks import CorrelationNetwork, GroupedNetworks
from contextualized.data import CorrelationDataModule
from contextualized.regression.lightning_modules import ContextualizedCorrelation

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

# Context mode: 'expression' , 'embeddings'
CONTEXT_MODE = 'embeddings'

# File Paths
PATH_L1000 = 'path/to/full_lincs_with_drugs.csv'
PATH_CTLS  = 'data/ctrls.csv'
EMB_FILE   = 'data/aido_cell_100m_lincs_embeddings.npy'

# Perturbation types
PERT_TYPES_TRAIN = ['trt_cp', 'trt_sh', 'trt_sh.cgs', 'trt_oe', 'trt_oe.cgs', 'trt_lig']
PERT_TYPES_PRED  = ['trt_cp', 'trt_sh', 'trt_sh.cgs', 'trt_oe', 'trt_oe.cgs', 'trt_lig']

# Model / data params
N_DATA_PCS       = 50
TEST_SIZE        = 0.33
RANDOM_STATE     = 42
N_CTRL_PCS       = 20
N_EMBEDDING_PCS  = 20

PERT_CONTEXT_MODE = 'morgan'

# Output dir
OUTPUT_DIR = 'table6_results'

# HGNC/Ensembl mapping
HGNC_TSV     = 'path/to/hgnc_complete_set.txt'
ENSEMBL_TSV  = None


def ensure_outdir(d: str | Path) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def col_ok(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def summarize_and_save_counts(df: pd.DataFrame, outdir: Path,
                              disease_col='diseaseName', drug_col='pert_id'):
    if disease_col in df.columns:
        c = (df[disease_col].astype(str).replace({'nan': np.nan, '': np.nan}).dropna())
        c.value_counts().to_csv(outdir / 'counts_disease.csv', header=['count'])
        print(f"Saved disease counts → {outdir/'counts_disease.csv'}")
    if drug_col in df.columns:
        c = (df[drug_col].astype(str).replace({'nan': np.nan, '': np.nan}).dropna())
        c.value_counts().to_csv(outdir / 'counts_drug.csv', header=['count'])
        print(f"Saved drug counts → {outdir/'counts_drug.csv'}")

def build_upper_tri_indices(p: int):
    iu, ju = np.triu_indices(p, k=1)
    return iu.astype(np.int32), ju.astype(np.int32)

BAD_SMILES_TOKENS = {'', '-666', 'nan', 'NaN', 'None', 'none', 'NULL'}

def smiles_to_morgan_fp(smiles: Optional[str], generator) -> np.ndarray:
    try:
        s = str(smiles).strip()
        if (s in BAD_SMILES_TOKENS) or pd.isna(smiles):
            return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
        for idx in fp.GetOnBits():
            arr[idx] = 1
        return arr
    except Exception:
        return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)

def compute_morgan_for_df(df: pd.DataFrame, smiles_col='canonical_smiles',
                          radius=3, nbits=4096) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    smiles = df.get(smiles_col, pd.Series([None]*len(df)))
    return np.vstack([smiles_to_morgan_fp(sm, gen) for sm in smiles])  # (N, n)


# HGNC/Ensembl mapping

def norm_ensg(x):
    if pd.isna(x): return None
    s = str(x).strip().upper()
    if not s: return None
    return s.split(".")[0]

def norm_symbol(s):
    if pd.isna(s): return None
    t = str(s).strip().upper()
    return t if t and t != "NAN" else None

def load_hgnc(hgnc_path: str | None):
    if not hgnc_path:
        return set(), {}, {}
    df = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    df = df.rename(columns=str.lower)

    approved = df["symbol"].dropna().str.upper().str.strip()
    approved_set = set(approved)

    alias2approved = {}
    for col in ["alias_symbol", "prev_symbol"]:
        if col in df.columns:
            tmp = df[[col, "symbol"]].dropna()
            for _, row in tmp.iterrows():
                aliases = [a.strip().upper() for a in str(row[col]).split("|") if a.strip()]
                appr = str(row["symbol"]).strip().upper()
                for a in aliases:
                    if a and a != "NAN":
                        alias2approved[a] = appr

    ensembl2symbol = {}
    if "ensembl_gene_id" in df.columns:
        for _, r in df[["ensembl_gene_id","symbol"]].dropna().iterrows():
            eg = norm_ensg(r["ensembl_gene_id"])
            sym = norm_symbol(r["symbol"])
            if eg and sym: ensembl2symbol[eg] = sym

    return approved_set, alias2approved, ensembl2symbol

def load_ensembl_tsv(tsv_path: str | None):
    if not tsv_path: return {}
    m = {}
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    for _, r in df.iterrows():
        eg = norm_ensg(r.get("ensembl_id"))
        sym = norm_symbol(r.get("hgnc_symbol"))
        if eg and sym: m[eg] = sym
    return m


def main():
    pl.seed_everything(RANDOM_STATE, workers=True)

    outdir = ensure_outdir(OUTPUT_DIR)

    if not os.path.exists(PATH_L1000): raise FileNotFoundError(PATH_L1000)
    if not os.path.exists(PATH_CTLS):  raise FileNotFoundError(PATH_CTLS)
    if CONTEXT_MODE == 'embeddings' and not os.path.exists(EMB_FILE):
        raise FileNotFoundError(EMB_FILE)

    print(f"Using cell line context mode: {CONTEXT_MODE}\n")

    # load
    print("Loading and filtering L1000 data...")
    df = pd.read_csv(PATH_L1000, engine='pyarrow')
    df = df[df['pert_type'].isin(PERT_TYPES_TRAIN)].reset_index(drop=True)

    bad = ((df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
           (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna()))
    df = df[~bad].reset_index(drop=True)
    print(f"After filtering: {len(df)} samples remaining")

    df['ignore_flag_pert_time'] = (df['pert_time'] == -666).astype(int)
    df['ignore_flag_pert_dose'] = (df['pert_dose'] == -666).astype(int)
    pert_time_mean = df.loc[df['pert_time'] != -666, 'pert_time'].mean()
    pert_dose_mean = df.loc[df['pert_dose'] != -666, 'pert_dose'].mean()
    df['pert_time'] = df['pert_time'].replace(-666, pert_time_mean)
    df['pert_dose'] = df['pert_dose'].replace(-666, pert_dose_mean)

    # gene expression features
    print("Preparing gene expression data...")
    feature_cols = [c for c in df.columns if '|' in c]
    if len(feature_cols) == 0:
        raise ValueError("No gene columns detected (expected 'SYMBOL|ENTREZ' headers).")
    X_raw = df[feature_cols].values.astype(np.float32)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    print(f"Gene expression data shape: {X_scaled.shape}")

    # perturbation context
    print("Preparing context features...")
    pert_dummy_cols = None
    dummy_index = None
    if PERT_CONTEXT_MODE in {'onehot', 'both'}:
        pert_dummies = pd.get_dummies(df['pert_id'].astype(str), drop_first=False)
        pert_dummy_cols = pert_dummies.columns.tolist()
        dummy_index = {pid: j for j, pid in enumerate(pert_dummy_cols)}
        print(f"Pert one-hot dims: {len(pert_dummy_cols)}")

    morgan_all = None
    if PERT_CONTEXT_MODE in {'morgan', 'both'}:
        if not col_ok(df, 'canonical_smiles'):
            raise ValueError("canonical_smiles column is required for Morgan fingerprints.")
        print(f"Computing Morgan fingerprints (radius=3, nBits=4096)...")
        morgan_all = compute_morgan_for_df(df, smiles_col='canonical_smiles',
                                           radius=3, nbits=4096)
        print(f"  Morgan matrix: {morgan_all.shape} (uint8)")

    # cell line context
    print("Preparing cell line context...")
    cell2vec = {}
    unique_cells_in_df = np.sort(df['cell_id'].unique())

    if CONTEXT_MODE == 'expression':
        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_in_df)]
        if ctrls_df.empty:
            raise ValueError("No overlapping cell_ids for control expression context.")
        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
        n_cells = ctrls_scaled.shape[0]
        n_ctx = min(N_CTRL_PCS, n_cells)
        pca_ctrls = PCA(n_components=n_ctx, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
        print(f"Control expression context: {n_ctx}D for {len(cell2vec)} cells")

    elif CONTEXT_MODE == 'embeddings':
        emb = np.load(EMB_FILE)
        ctrls_meta = pd.read_csv(PATH_CTLS, index_col=0)
        emb_ids = ctrls_meta.index.to_numpy()
        if len(emb_ids) != emb.shape[0]:
            raise ValueError("Embeddings count mismatch vs ctrls.csv index.")
        scaler_emb = StandardScaler()
        emb_scaled = scaler_emb.fit_transform(emb)
        n_ctx = min(N_EMBEDDING_PCS, emb_scaled.shape[1])
        pca_emb = PCA(n_components=n_ctx, random_state=RANDOM_STATE)
        emb_pcs = pca_emb.fit_transform(emb_scaled)
        cell2vec = dict(zip(emb_ids, emb_pcs))
        print(f"AIDO context: original {emb_scaled.shape[1]}D → {n_ctx}D for {len(cell2vec)} cells")
    else:
        raise ValueError(f"Invalid CONTEXT_MODE: {CONTEXT_MODE}")

    print("Building context matrix...")
    unique_cells = np.sort(list(cell2vec.keys()))
    if unique_cells.size == 0:
        raise RuntimeError("No cell IDs after context loading.")

    cell_ids = df['cell_id'].to_numpy()
    pt = df['pert_time'].to_numpy().reshape(-1, 1)
    pdose = df['pert_dose'].to_numpy().reshape(-1, 1)
    ign_t = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1).astype(np.float32)
    ign_d = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1).astype(np.float32)

    cont_blocks, other_blocks = [], []
    for cid in unique_cells:
        mask = (cell_ids == cid)
        if mask.sum() == 0: continue

        cont = np.hstack([np.tile(cell2vec[cid], (mask.sum(), 1)), pt[mask], pdose[mask]])
        
        blocks = []
        if PERT_CONTEXT_MODE in {'onehot', 'both'}:
            oh = np.zeros((mask.sum(), len(pert_dummy_cols)), dtype=np.float32)
            perts_here = df.loc[mask, 'pert_id'].astype(str).tolist()
            for r, pid in enumerate(perts_here):
                j = dummy_index.get(pid, None)
                if j is not None: oh[r, j] = 1.0
            blocks.append(oh)

        if PERT_CONTEXT_MODE in {'morgan', 'both'}:
            blocks.append(morgan_all[mask].astype(np.float32))

        blocks.append(ign_t[mask])
        blocks.append(ign_d[mask])

        other = np.hstack(blocks)

        cont_blocks.append(cont)
        other_blocks.append(other)

    C_cont = np.vstack(cont_blocks)
    C_other = np.vstack(other_blocks)

    print("Scaling continuous context features...")
    scaler_ctx = StandardScaler()
    C_cont_sc = scaler_ctx.fit_transform(C_cont)
    
    if hasattr(scaler_ctx, "scale_"):
        bad = ~np.isfinite(scaler_ctx.scale_) | (scaler_ctx.scale_ == 0)
        if bad.any():
            print(f"ctx scaler had {bad.sum()} zero/non-finite scales: setting to 1.0")
            scaler_ctx.scale_[bad] = 1.0
    
    C_cont_sc = np.nan_to_num(C_cont_sc, nan=0.0, posinf=0.0, neginf=0.0)
    C_all = np.hstack([C_cont_sc, C_other]).astype(np.float32)
    print(f"Context dims: continuous: {C_cont.shape[1]}, other: {C_other.shape[1]}, total: {C_all.shape[1]}")

    print("Splitting data into train/test sets (per cell)...")
    X_tr, X_te, C_tr, C_te = [], [], [], []
    id_tr, id_te = [], []
    start = 0
    for cid in unique_cells:
        mask = (cell_ids == cid)
        if mask.sum() < 2: continue
        end = start + mask.sum()

        Xc = X_scaled[mask]
        Cc = C_all[start:end]
        ids = cell_ids[mask]

        xa, xb, ca, cb, ia, ib = train_test_split(Xc, Cc, ids, test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE, shuffle=True)
        X_tr.append(xa); X_te.append(xb)
        C_tr.append(ca); C_te.append(cb)
        id_tr.append(ia); id_te.append(ib)
        start = end

    X_train = np.vstack(X_tr); X_test = np.vstack(X_te)
    C_train = np.vstack(C_tr); C_test = np.vstack(C_te)
    ids_train = np.concatenate(id_tr); ids_test = np.concatenate(id_te)
    print(f'Context matrix: train {C_train.shape}, test {C_test.shape}')

    print("Applying PCA to gene expression data...")
    pca = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
    X_tr_p = pca.fit_transform(X_train)
    X_te_p = pca.transform(X_test)
    mu, sigma = X_tr_p.mean(0), X_tr_p.std(0); sigma[sigma == 0] = 1.0
    X_train_z = (X_tr_p - mu) / sigma
    X_test_z  = (X_te_p - mu) / sigma
    print(f"Gene expression: {N_DATA_PCS} PCs + latent z-score")

    # save transforms
    tf_file = outdir / 'transforms.npz'
    np.savez(
        tf_file,
        feature_cols=np.array(feature_cols, dtype=object),
        pert_dummy_cols=np.array(pert_dummy_cols if pert_dummy_cols is not None else [], dtype=object),
        context_mode=CONTEXT_MODE,
        pert_context_mode=PERT_CONTEXT_MODE,
        n_ctrl_pcs=N_CTRL_PCS,
        n_embedding_pcs=N_EMBEDDING_PCS,
        scaler_X_mean=getattr(scaler_X, 'mean_', None),
        scaler_X_scale=getattr(scaler_X, 'scale_', None),
        pca_components_=getattr(pca, 'components_', None),
        pca_explained_variance_=getattr(pca, 'explained_variance_', None),
        pca_mean_=getattr(pca, 'mean_', None),
        latent_mu=mu, latent_sigma=sigma,
        ctx_scaler_mean=getattr(scaler_ctx, 'mean_', None),
        ctx_scaler_scale=getattr(scaler_ctx, 'scale_', None),
    )
    print(f"Saved transforms & column orders to {tf_file}")

    # baseline`
    print("Training baseline models...")
    pop = CorrelationNetwork().fit(X_train_z)
    grp = GroupedNetworks(CorrelationNetwork).fit(X_train_z, ids_train)
    print(f"Population model - Test MSE: {pop.measure_mses(X_test_z).mean():.6f}")
    print(f"Grouped model - Test MSE: {grp.measure_mses(X_test_z, ids_test).mean():.6f}")

    # contextualized model
    print("Training contextualized model...")
    model_kwargs = dict(context_dim=C_train.shape[1], x_dim=X_train_z.shape[1],
                        encoder_type='mlp', num_archetypes=50)
    model = ContextualizedCorrelation(**model_kwargs)

    X_trn, X_val, C_trn, C_val = train_test_split(X_train_z, C_train, test_size=0.2,
                                                  random_state=RANDOM_STATE, shuffle=True)

    ckpt = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model')

    accelerator, devices = ('gpu', 1) if torch.cuda.is_available() else ('cpu', 'auto')
    if torch.cuda.is_available():
        print("CUDA available: Using GPU")
    else:
        print("CUDA not available: Using CPU")

    dm = CorrelationDataModule(C_train=C_trn, X_train=X_trn,
                               C_val=C_val, X_val=X_val,
                               C_test=C_test, X_test=X_test_z,
                               C_predict=C_trn[:1], X_predict=X_trn[:1],
                               batch_size=64)

    trainer = Trainer(max_epochs=10, accelerator=accelerator, devices=devices, callbacks=[ckpt], logger=False)
    trainer.fit(model, datamodule=dm)
    print("Testing model performance...")
    trainer.test(model, datamodule=dm)

    best_ckpt = ckpt.best_model_path
    print(f"Best model saved at: {best_ckpt}")
    model = ContextualizedCorrelation.load_from_checkpoint(best_ckpt, **model_kwargs).eval()

    print("Preparing prediction rows...")

    dfp = pd.read_csv(PATH_L1000, engine='pyarrow')
    dfp = dfp[dfp['pert_type'].isin(PERT_TYPES_PRED)].reset_index(drop=True)
    badp = ((dfp['distil_cc_q75'] < 0.2) | (dfp['distil_cc_q75'] == -666) | (dfp['distil_cc_q75'].isna()) |
            (dfp['pct_self_rank_q25'] > 5) | (dfp['pct_self_rank_q25'] == -666) | (dfp['pct_self_rank_q25'].isna()))
    dfp = dfp[~badp].reset_index(drop=True)

    dfp['ignore_flag_pert_time'] = (dfp['pert_time'] == -666).astype(int)
    dfp['ignore_flag_pert_dose'] = (dfp['pert_dose'] == -666).astype(int)
    dfp['pert_time'] = dfp['pert_time'].replace(-666, pert_time_mean)
    dfp['pert_dose'] = dfp['pert_dose'].replace(-666, pert_dose_mean)

    print(f"Predict rows: {len(dfp)}")

    morgan_pred = None
    if PERT_CONTEXT_MODE in {'morgan', 'both'}:
        if not col_ok(dfp, 'canonical_smiles'):
            raise ValueError("canonical_smiles required for Morgan block in prediction set.")
        print(f"Computing Morgan fingerprints for prediction set (radius=3, nBits=4096)...")
        morgan_pred = compute_morgan_for_df(dfp, smiles_col='canonical_smiles',
                                            radius=3, nbits=4096).astype(np.float32)

    pca_metagene_rows, expr_raw_rows, contexts = [], [], []
    meta_df_rows = []

    dummy_index_pred = {pid: j for j, pid in enumerate(pert_dummy_cols)} if pert_dummy_cols is not None else {}
    onehot_dim = len(pert_dummy_cols) if pert_dummy_cols else 0

    non_gene_cols = [c for c in dfp.columns if c not in feature_cols]

    for ridx, row in dfp.iterrows():
        cid = row['cell_id']
        if cid not in cell2vec:
            continue

        # raw gene vector
        try:
            x_raw = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        except Exception:
            continue
        x_scaled = scaler_X.transform(x_raw)
        x_p = pca.transform(x_scaled)
        x_z = (x_p - mu) / sigma

        expr_raw_rows.append(x_raw.astype(np.float32).squeeze(0))
        pca_metagene_rows.append(x_z.astype(np.float32).squeeze(0))

        # continuous ctx
        cont = np.hstack([cell2vec[cid], [row['pert_time']], [row['pert_dose']]]).reshape(1, -1)
        cont_sc = scaler_ctx.transform(cont)
        cont_sc = np.nan_to_num(cont_sc, nan=0.0, posinf=0.0, neginf=0.0)

        blocks = [cont_sc]

        if PERT_CONTEXT_MODE in {'onehot', 'both'} and pert_dummy_cols is not None:
            oh = np.zeros((1, onehot_dim), dtype=np.float32)
            j = dummy_index_pred.get(str(row['pert_id']), None)
            if j is not None: oh[0, j] = 1.0
            blocks.append(oh)

        if PERT_CONTEXT_MODE in {'morgan', 'both'}:
            blocks.append(morgan_pred[ridx:ridx+1, :])

        flags = np.array([[row['ignore_flag_pert_time'], row['ignore_flag_pert_dose']]], dtype=np.float32)
        blocks.append(flags)

        C_row = np.hstack(blocks)
        contexts.append(C_row.astype(np.float32).squeeze(0))

        meta_df_rows.append(row[non_gene_cols])

    if len(pca_metagene_rows) == 0:
        raise RuntimeError("No rows prepared for prediction.")

    pca_metagene = np.asarray(pca_metagene_rows, dtype=np.float32)
    expr_raw     = np.asarray(expr_raw_rows, dtype=np.float32)
    contexts     = np.asarray(contexts, dtype=np.float32)
    
    # load HGNC/Ensembl mappings
    approved_set, alias2approved, ensembl2symbol_hgnc = load_hgnc(HGNC_TSV)
    ensembl2symbol_gtf = load_ensembl_tsv(ENSEMBL_TSV)
    ensembl2symbol = dict(ensembl2symbol_gtf); ensembl2symbol.update(ensembl2symbol_hgnc)
    
    # build metadata frame
    meta_aligned  = pd.DataFrame(meta_df_rows).reset_index(drop=True)

    if 'sig_id' not in meta_aligned.columns:
        meta_aligned['sig_id'] = [f"sig_{i}" for i in range(len(meta_aligned))]

    def derive_symbol(row):
        pt = str(row.get('pert_type',''))
        # genetics
        if pt.startswith('trt_sh') or pt.startswith('trt_oe'):
            s = norm_symbol(row.get('pert_iname'))
            if s and s not in approved_set and s in alias2approved:
                s = alias2approved[s]
            return s

        # drugs
        if pt == 'trt_cp':
            for col in ('target_gene','target_gene_drug','target_symbol','targetSymbol',
                        'primaryTargetSymbol','primaryTarget'):
                s = norm_symbol(row.get(col))
                if s:
                    return alias2approved.get(s, s)
            eg = norm_ensg(row.get('targetId'))
            if eg:
                s = ensembl2symbol.get(eg)
                if s: return s
            return None

        # others
        return None

    meta_aligned['__target_key'] = meta_aligned.apply(derive_symbol, axis=1)
    meta_aligned['__target_key'] = meta_aligned['__target_key'].map(norm_symbol)

    pt_series = meta_aligned['pert_type'].astype(str)
    is_drug = pt_series.eq('trt_cp')
    is_gen  = pt_series.str.startswith(('trt_sh','trt_oe'))
    n_drug_sym = int((is_drug & meta_aligned['__target_key'].notna()).sum())
    n_gen_sym  = int((is_gen  & meta_aligned['__target_key'].notna()).sum())
    genes_drug = set(meta_aligned.loc[is_drug & meta_aligned['__target_key'].notna(), '__target_key'])
    genes_gen  = set(meta_aligned.loc[is_gen  & meta_aligned['__target_key'].notna(), '__target_key'])
    print(f"[mapping] drugs with symbol: {n_drug_sym} | genetics with symbol: {n_gen_sym} | intersection: {len(genes_drug & genes_gen)}")

    print(f"\nPrepared {len(pca_metagene)} aligned rows for prediction")

    # Predict
    dm_pred = CorrelationDataModule(C_train=C_trn, X_train=X_trn,
                                    C_val=C_val, X_val=X_val,
                                    C_test=C_test, X_test=X_test_z,
                                    C_predict=contexts, X_predict=pca_metagene,
                                    batch_size=64)

    print("Running batch prediction...")
    tr_pred = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1 if torch.cuda.is_available() else 'auto',
                      enable_progress_bar=True, logger=False)
    preds = tr_pred.predict(model, datamodule=dm_pred)

    # collect prediction outputs
    cors = np.concatenate([b['correlations'].detach().cpu().numpy() for b in preds], axis=0)

    if cors.ndim == 3:
        p = cors.shape[1]
        tri_i, tri_j = build_upper_tri_indices(p)
        cors_vec = cors[:, tri_i, tri_j]
    elif cors.ndim == 2:
        cors_vec = cors.astype(np.float32)
        p = N_DATA_PCS
        tri_i, tri_j = build_upper_tri_indices(p)
    else:
        raise ValueError(f"Unexpected correlations shape: {cors.shape}")

    cors_vec = cors_vec.astype(np.float32)
    
    if not np.isfinite(cors_vec).all():
        nbad = np.size(cors_vec) - np.isfinite(cors_vec).sum()
        print(f"{nbad} non-finite correlation entries → zeroing")
        cors_vec = np.nan_to_num(cors_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    assert cors_vec.shape[0] == len(pca_metagene) == len(contexts)

    # retrieval views
    retrieval_pca_metagene = pca_metagene
    retrieval_net_corr     = cors_vec
    retrieval_concat       = np.hstack([retrieval_pca_metagene, retrieval_net_corr]).astype(np.float32)

    meta_path = outdir / 'aligned_metadata.parquet'
    meta_aligned.to_parquet(meta_path, index=False)
    print(f"Saved aligned metadata to {meta_path}")

    # save aligned arrays
    final_npz = outdir / 'aligned_predictions_topk.npz'
    np.savez_compressed(
        final_npz,
        pca_metagene=pca_metagene,
        correlations=cors_vec,
        contexts=contexts,

        expr_raw=expr_raw, # (N, n_genes)
        feature_cols=np.array(feature_cols, dtype=object),

        retrieval_pca_metagene=retrieval_pca_metagene,
        retrieval_net_corr=retrieval_net_corr,
        retrieval_concat=retrieval_concat,

        tri_upper_i=tri_i, tri_upper_j=tri_j,
    )
    saved = np.load(final_npz, allow_pickle=True)
    print(f"\nPrediction saved to: {final_npz}")
    print(f"  • pca_metagene : {saved['pca_metagene'].shape}")
    print(f"  • correlations : {saved['correlations'].shape}")
    print(f"  • contexts     : {saved['contexts'].shape}")
    print(f"  • expr_raw     : {saved['expr_raw'].shape}")
    print(f"  • retrieval_concat: {saved['retrieval_concat'].shape}")

    print("\nSaving training data and results...")
    training_data_file = outdir / 'training_data.npz'
    np.savez(training_data_file,
             C_train=C_train, C_test=C_test, X_train=X_train_z, X_test=X_test_z,
             unique_cells=np.sort(list(cell2vec.keys())),
             cell_ids_train=ids_train, cell_ids_test=ids_test,
             config={
                 'path_l1000': PATH_L1000,
                 'path_ctls': PATH_CTLS,
                 'emb_file': EMB_FILE,
                 'context_mode': CONTEXT_MODE,
                 'pert_context_mode': PERT_CONTEXT_MODE,
                 'pert_types_train': PERT_TYPES_TRAIN,
                 'pert_types_pred': PERT_TYPES_PRED,
                 'n_data_pcs': N_DATA_PCS,
                 'test_size': TEST_SIZE,
                 'random_state': RANDOM_STATE,
                 'n_ctrl_pcs': N_CTRL_PCS,
                 'n_embedding_pcs': N_EMBEDDING_PCS
             })
    print(f"Saved training data to {training_data_file}")

    results_file = outdir / 'training_results.npz'
    np.savez(results_file,
             checkpoint_path=str(best_ckpt),
             aligned_data_path=str(final_npz),
             metadata_path=str(meta_path),
             note='Aligned predictions + full metadata; retrieval views for top-k.')
    print(f"Saved training results to {results_file}")

    # counts
    summarize_and_save_counts(df, outdir, disease_col='diseaseName', drug_col='pert_id')

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {outdir}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Aligned metadata: {meta_path}")
    print(f"Aligned predictions: {final_npz}")
    print(f"Retrieval keys in NPZ: 'retrieval_pca_metagene', 'retrieval_net_corr', 'retrieval_concat'")

    return {
        'checkpoint_path': best_ckpt,
        'training_data_file': str(training_data_file),
        'results_file': str(results_file),
        'aligned_data_file': str(final_npz),
        'metadata_file': str(meta_path),
        'output_dir': str(outdir)
    }

if __name__ == "__main__":
    _ = main()
