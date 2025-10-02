"""
Contextualized Network Training & Aligned Prediction Script

Outputs:
  - table5_results/training_data.npz
  - table5_results/simple_aligned_predictions.npz
  - table5_results/training_results.npz
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


# Context mode: 'expression' , 'embeddings'
CONTEXT_MODE = 'embeddings'

# File Paths
PATH_L1000 = 'path/to/trt_cp_with_disease.csv'
PATH_CTLS  = 'data/ctrls.csv'
EMB_FILE   = 'data/aido_cell_100m_lincs_embeddings.npy'

# Model / data params
N_DATA_PCS       = 50
TEST_SIZE        = 0.33
RANDOM_STATE     = 42
N_CTRL_PCS       = 20
N_EMBEDDING_PCS  = 20
pert_to_fit_on   = ['trt_cp']

# Output dir
OUTPUT_DIR = 'table5_results'


def ensure_outdir(d: str | Path) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def col_ok(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def get_or_none(row: pd.Series, name: str):
    return row[name] if name in row.index else None

def filter_generic_diseases(disease_labels):
    """Filter out overly-generic disease terms."""
    generic_terms = {'neoplasm', 'cancer', 'carcinoma', 'lymphoma', 'leukemia', 'adenocarcinoma'}
    return np.array([str(d).strip().lower() not in generic_terms for d in disease_labels])

def save_training_data(C_train, C_test, X_train, X_test, unique_cells, cell_ids_train, cell_ids_test, output_dir):
    output_path = ensure_outdir(output_dir)
    training_data = {
        'C_train': C_train,
        'C_test': C_test,
        'X_train': X_train,
        'X_test': X_test,
        'unique_cells': unique_cells,
        'cell_ids_train': cell_ids_train,
        'cell_ids_test': cell_ids_test,
        'config': {
            'path_l1000': PATH_L1000,
            'path_ctls': PATH_CTLS,
            'emb_file': EMB_FILE,
            'context_mode': CONTEXT_MODE,
            'pert_to_fit_on': pert_to_fit_on,
            'n_data_pcs': N_DATA_PCS,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'n_ctrl_pcs': N_CTRL_PCS,
            'n_embedding_pcs': N_EMBEDDING_PCS
        }
    }
    save_file = output_path / 'training_data.npz'
    np.savez(save_file, **training_data)
    # print(f"Saved training data to {save_file}")
    return save_file

def smiles_to_morgan_fp(smiles: Optional[str], generator) -> np.ndarray:
    if smiles is None:
        return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
    s = str(smiles).strip()
    if not s:
        return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros(generator.GetOptions().fpSize, dtype=np.uint8)
    for idx in fp.GetOnBits():
        arr[idx] = 1
    return arr

def compute_morgan_for_df(df: pd.DataFrame, smiles_col='canonical_smiles',
                          radius=0, nbits=4) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    smiles = df.get(smiles_col, pd.Series([None]*len(df)))
    return np.vstack([smiles_to_morgan_fp(sm, gen) for sm in smiles])  # (N, n)


def main():
    pl.seed_everything(RANDOM_STATE, workers=True)

    outdir = ensure_outdir(OUTPUT_DIR)

    if not os.path.exists(PATH_L1000): raise FileNotFoundError(PATH_L1000)
    if not os.path.exists(PATH_CTLS):  raise FileNotFoundError(PATH_CTLS)
    if CONTEXT_MODE == 'embeddings' and not os.path.exists(EMB_FILE):
        raise FileNotFoundError(EMB_FILE)

    # print(f"Using cell line context mode: {CONTEXT_MODE}\n")

    # load
    print("Loading and filtering L1000 data...")
    df = pd.read_csv(PATH_L1000, engine='pyarrow')
    df = df[df['pert_type'].isin(pert_to_fit_on)].reset_index(drop=True)
    bad = ((df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
           (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna()))
    df = df[~bad].reset_index(drop=True)
    # print(f"After filtering: {len(df)} samples remaining")

    df['ignore_flag_pert_time'] = (df['pert_time'] == -666).astype(int)
    df['ignore_flag_pert_dose'] = (df['pert_dose'] == -666).astype(int)
    pert_time_mean = df.loc[df['pert_time'] != -666, 'pert_time'].mean()
    pert_dose_mean = df.loc[df['pert_dose'] != -666, 'pert_dose'].mean()
    df['pert_time'] = df['pert_time'].replace(-666, pert_time_mean)
    df['pert_dose'] = df['pert_dose'].replace(-666, pert_dose_mean)

    # gene expression features
    # print("Preparing gene expression data...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    X_raw = df[feature_cols].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    # print(f"Gene expression data shape: {X_scaled.shape}")

    # perturbation context
    # print("Preparing context features...")
    pert_dummies = pd.get_dummies(df['pert_id'].astype(str), drop_first=True)
    pert_dummy_cols = pert_dummies.columns.tolist()
    dummy_index = {pid: j for j, pid in enumerate(pert_dummy_cols)}
    # print(f"Pert one-hot dims: {len(pert_dummy_cols)}")

    # cell line context
    # print("Preparing cell line context...")
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
        # print(f"Control expression context: {n_ctx}D for {len(cell2vec)} cells")

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
        # print(f"AIDO context: original {emb_scaled.shape[1]}D → {n_ctx}D for {len(cell2vec)} cells")
    else:
        raise ValueError(f"Invalid CONTEXT_MODE: {CONTEXT_MODE}")

    # print("Building context matrix...")
    unique_cells = np.sort(list(cell2vec.keys()))
    if unique_cells.size == 0:
        raise RuntimeError("No cell IDs after context loading.")

    cell_ids = df['cell_id'].to_numpy()
    pt = df['pert_time'].to_numpy().reshape(-1, 1)
    pdose = df['pert_dose'].to_numpy().reshape(-1, 1)
    ign_t = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1).astype(np.float32)
    ign_d = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1).astype(np.float32)

    cont_blocks, other_blocks = [], []
    start = 0
    for cid in unique_cells:
        mask = (cell_ids == cid)
        if mask.sum() == 0: continue

        cont = np.hstack([np.tile(cell2vec[cid], (mask.sum(), 1)), pt[mask], pdose[mask]])
        
        # one-hot pert
        oh = np.zeros((mask.sum(), len(pert_dummy_cols)), dtype=np.float32)
        perts_here = df.loc[mask, 'pert_id'].astype(str).tolist()
        for r, pid in enumerate(perts_here):
            j = dummy_index.get(pid, None)
            if j is not None: oh[r, j] = 1.0

        other = np.hstack([oh, ign_t[mask], ign_d[mask]])

        cont_blocks.append(cont)
        other_blocks.append(other)
        start += mask.sum()

    C_cont = np.vstack(cont_blocks)
    C_other = np.vstack(other_blocks)

    # print("Scaling continuous context features...")
    scaler_ctx = StandardScaler()
    C_cont_sc = scaler_ctx.fit_transform(C_cont)
    C_all = np.hstack([C_cont_sc, C_other])
    # print(f"Context dims - continuous: {C_cont.shape[1]}, other: {C_other.shape[1]}, total: {C_all.shape[1]}")

    # train/test split
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
    # print(f'Context matrix: train {C_train.shape}, test {C_test.shape}')

    # pca + z-score
    # print("Applying PCA to gene expression data...")
    pca = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
    X_tr_p = pca.fit_transform(X_train)
    X_te_p = pca.transform(X_test)
    mu, sigma = X_tr_p.mean(0), X_tr_p.std(0); sigma[sigma == 0] = 1.0
    X_train_z = (X_tr_p - mu) / sigma
    X_test_z  = (X_te_p - mu) / sigma
    # print(f"Gene expression: {N_DATA_PCS} PCs + latent z-score")

    # baseline
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
                               batch_size=32)

    trainer = Trainer(max_epochs=10, accelerator=accelerator, devices=devices, callbacks=[ckpt], logger=False)
    trainer.fit(model, datamodule=dm)
    # print("Testing model performance...")
    trainer.test(model, datamodule=dm)

    best_ckpt = ckpt.best_model_path
    # print(f"Best model saved at: {best_ckpt}")
    model = ContextualizedCorrelation.load_from_checkpoint(best_ckpt, **model_kwargs).eval()

    # predict disease samples
    dfp = pd.read_csv(PATH_L1000, engine='pyarrow')
    dfp = dfp[dfp['pert_type'].isin(pert_to_fit_on)].reset_index(drop=True)
    badp = ((dfp['distil_cc_q75'] < 0.2) | (dfp['distil_cc_q75'] == -666) | (dfp['distil_cc_q75'].isna()) |
            (dfp['pct_self_rank_q25'] > 5) | (dfp['pct_self_rank_q25'] == -666) | (dfp['pct_self_rank_q25'].isna()))
    dfp = dfp[~badp].reset_index(drop=True)

    dfp['ignore_flag_pert_time'] = (dfp['pert_time'] == -666).astype(int)
    dfp['ignore_flag_pert_dose'] = (dfp['pert_dose'] == -666).astype(int)
    dfp['pert_time'] = dfp['pert_time'].replace(-666, pert_time_mean)
    dfp['pert_dose'] = dfp['pert_dose'].replace(-666, pert_dose_mean)

    has_disease = dfp.get('diseaseName', pd.Series(['']*len(dfp))).astype(str).str.strip().ne('')
    pred_df = dfp[has_disease].copy()
    if len(pred_df) == 0:
        pred_df = dfp.copy()
    
    # filter out generic disease terms
    if 'diseaseName' in pred_df.columns:
        keep_mask = filter_generic_diseases(pred_df['diseaseName'].values)
        pred_df = pred_df[keep_mask].reset_index(drop=True)
    # print(f"Predict rows: {len(pred_df)}")

    # morgan fingerprint
    morgan_pred_save = None
    if col_ok(pred_df, 'canonical_smiles'):
        morgan_pred_save = compute_morgan_for_df(pred_df, 'canonical_smiles', radius=0, nbits=4)

    dummy_index_pred = {pid: j for j, pid in enumerate(pert_dummy_cols)} if pert_dummy_cols else {}
    onehot_dim = len(pert_dummy_cols) if pert_dummy_cols else 0

    # build aligned transforms
    pca_metagene_rows, expr_raw_rows, contexts, disease_names = [], [], [], []

    for _, row in pred_df.iterrows():
        cid = row['cell_id']
        if cid not in cell2vec:  # skip unseen cell
            continue

        # raw gene vector in training order
        try:
            x_raw = row[list(feature_cols)].to_numpy().reshape(1, -1)
        except Exception:
            continue
        x_scaled = scaler_X.transform(x_raw)
        x_p = pca.transform(x_scaled)
        x_z = (x_p - mu) / sigma

        # continuous ctx
        cont = np.hstack([cell2vec[cid], [row['pert_time']], [row['pert_dose']]]).reshape(1, -1)
        cont_sc = scaler_ctx.transform(cont)

        # one-hot perturbation
        oh = np.zeros((1, onehot_dim), dtype=np.float32)
        j = dummy_index_pred.get(str(row['pert_id']), None)
        if j is not None: oh[0, j] = 1.0

        flags = np.array([[row['ignore_flag_pert_time'], row['ignore_flag_pert_dose']]], dtype=np.float32)
        C_row = np.hstack([cont_sc, oh, flags])

        pca_metagene_rows.append(x_z.squeeze(0))
        expr_raw_rows.append(x_raw.astype(np.float32).squeeze(0))
        contexts.append(C_row.squeeze(0))
        disease_names.append(get_or_none(row, 'diseaseName'))

    pca_metagene = np.asarray(pca_metagene_rows, dtype=np.float32)
    expr_raw     = np.asarray(expr_raw_rows, dtype=np.float32)
    contexts     = np.asarray(contexts, dtype=np.float32)
    disease_names = np.asarray(disease_names, dtype=object)

    # print(f"\nPrepared {len(pca_metagene)} aligned disease rows for prediction")

    dm_pred = CorrelationDataModule(C_train=C_train, X_train=X_train_z,
                                    C_val=C_val, X_val=X_val,
                                    C_test=C_test, X_test=X_test_z,
                                    C_predict=contexts, X_predict=pca_metagene,
                                    batch_size=64)
    tr_pred = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1 if torch.cuda.is_available() else 'auto',
                      enable_progress_bar=True, logger=False)
    preds = tr_pred.predict(model, datamodule=dm_pred)

    cors = np.concatenate([b['correlations'].detach().cpu().numpy() for b in preds], axis=0).astype(np.float32)
    assert cors.shape[0] == pca_metagene.shape[0]

    # print(f"Processed {len(pca_metagene)} disease samples")

    # Save aligned npz
    aligned_path = outdir / 'simple_aligned_predictions.npz'
    payload = {
        'pca_metagene': pca_metagene,
        'expr_raw': expr_raw,
        'correlations': cors,
        'contexts': contexts,
        'disease_labels': disease_names,
    }
    if morgan_pred_save is not None:
        payload['morgan_fp'] = morgan_pred_save  # (N, n)
    
    np.savez_compressed(aligned_path, **payload)
    print(f"Saved aligned NPZ to {aligned_path}")

    training_npz = save_training_data(C_train, C_test, X_train_z, X_test_z,
                                      np.sort(list(cell2vec.keys())), ids_train, ids_test, outdir)
    results_npz = outdir / 'training_results.npz'
    results_data = {
        'checkpoint_path': str(best_ckpt),
        'aligned_data_path': str(aligned_path),
        'note': 'Per-row transforms; predictions aligned 1:1 with expressions/labels'
    }
    np.savez(results_npz, **results_data)
    print(f"Saved results to {results_npz}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {outdir}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Training data: {training_npz}")
    print(f"Disease samples: {aligned_path}")

    print(f"\nModel Performance Summary:")
    print(f"  • Population model test MSE: {pop.measure_mses(X_test_z).mean():.6f}")
    print(f"  • Grouped model test MSE:    {grp.measure_mses(X_test_z, ids_test).mean():.6f}")
    print(f"  • Contextualized model: training complete; aligned correlations saved")

    return {
        'checkpoint_path': best_ckpt,
        'training_data_file': training_npz,
        'results_file': results_npz,
        'aligned_data_file': aligned_path,
        'output_dir': outdir
    }

if __name__ == "__main__":
    _ = main()