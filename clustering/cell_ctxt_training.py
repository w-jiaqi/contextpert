"""
Contextualized Network Training & Aligned Prediction Script

Outputs:
  - training_results_new/training_data.npz
  - training_results_new/simple_aligned_predictions.npz
  - training_results_new/training_results.npz
"""

import os
from pathlib import Path
from collections import Counter

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

# Context mode: 'expression' for PCA of control expression, 'embeddings' for AIDO embeddings
CONTEXT_MODE = 'embeddings'

# File Paths
PATH_L1000 = 'path/to/full_lincs_dataset_with_disease.csv'
PATH_CTLS  = 'data/ctrls.csv'
EMB_FILE  = 'data/aido_cell_100m_lincs_embeddings.npy'

# Model / data params
N_DATA_PCS       = 50
TEST_SIZE        = 0.33
RANDOM_STATE     = 42
N_CTRL_PCS       = 20
N_EMBEDDING_PCS  = 20
pert_to_fit_on   = ['trt_cp']

OUTPUT_DIR = 'training_results_new'


def save_training_data(C_train, C_test, X_train, X_test, unique_cells, cell_ids_train, cell_ids_test, output_dir):
    """
    Save all necessary data for downstream analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

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
    print(f"Saved training data to {save_file}")
    return save_file


def filter_generic_diseases(disease_labels):
    """Filter out overly-generic disease terms."""
    generic_terms = {'neoplasm', 'cancer', 'carcinoma', 'lymphoma', 'leukemia', 'adenocarcinoma'}
    mask = np.array([str(d).strip().lower() not in generic_terms for d in disease_labels])
    return mask


def main():
    print("=" * 80)
    print("CONTEXTUALIZED NETWORK TRAINING")
    print("=" * 80)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    if not os.path.exists(PATH_L1000):
        raise FileNotFoundError(f"L1000 data not found at {PATH_L1000}")
    if not os.path.exists(PATH_CTLS):
        raise FileNotFoundError(f"Controls data not found at {PATH_CTLS}")
    if CONTEXT_MODE == 'embeddings' and not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f"Embeddings file not found at {EMB_FILE}")

    print(f"Using cell line context mode: {CONTEXT_MODE}\n")
    print("Loading and filtering L1000 data...")
    df = pd.read_csv(PATH_L1000, engine='pyarrow')
    df = df[df['pert_type'].isin(pert_to_fit_on)]

    bad = (
        (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
        (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
    )
    df = df[~bad]
    print(f"After filtering: {len(df)} samples remaining")

    df['ignore_flag_pert_time'] = (df['pert_time'] == -666).astype(int)
    df['ignore_flag_pert_dose'] = (df['pert_dose'] == -666).astype(int)
    pert_time_mean = df.loc[df['pert_time'] != -666, 'pert_time'].mean()
    pert_dose_mean = df.loc[df['pert_dose'] != -666, 'pert_dose'].mean()
    df['pert_time'] = df['pert_time'].replace(-666, pert_time_mean)
    df['pert_dose'] = df['pert_dose'].replace(-666, pert_dose_mean)

    print("Preparing gene expression data...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    X_raw = df[feature_cols].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    print(f"Gene expression data shape: {X_scaled.shape}")

    print("Preparing context features...")
    pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True)
    pert_dummy_cols = pert_dummies.columns.tolist()  # freeze columns order

    pert_time = df['pert_time'].to_numpy().reshape(-1, 1)
    pert_dose = df['pert_dose'].to_numpy().reshape(-1, 1)
    ignore_time = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
    ignore_dose = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)

    # Cell line context
    print("Preparing cell line context...")
    cell2vec = {}
    unique_cells_in_l1000 = np.sort(df['cell_id'].unique())

    if CONTEXT_MODE == 'expression':
        ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
        ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_in_l1000)]
        if ctrls_df.empty:
            raise ValueError("No common cell IDs between L1000 and ctrls.csv for PCA control expression.")

        scaler_ctrls = StandardScaler()
        ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
        n_cells = ctrls_scaled.shape[0]
        n_components_for_context = min(N_CTRL_PCS, n_cells)

        pca_ctrls = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
        ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
        cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
        print(f"Control expression context: {n_components_for_context}D for {len(cell2vec)} cells")

    elif CONTEXT_MODE == 'embeddings':
        all_embeddings_raw = np.load(EMB_FILE)
        ctrls_meta_df = pd.read_csv(PATH_CTLS, index_col=0)
        embedding_cell_ids_full = ctrls_meta_df.index.to_numpy()

        if len(embedding_cell_ids_full) != all_embeddings_raw.shape[0]:
            raise ValueError(
                f"Embeddings count mismatch: {all_embeddings_raw.shape[0]} vs ctrls.csv IDs {len(embedding_cell_ids_full)}"
            )

        scaler_embeddings = StandardScaler()
        embeddings_scaled = scaler_embeddings.fit_transform(all_embeddings_raw)
        n_embeddings_dim = embeddings_scaled.shape[1]
        n_components_for_context = min(N_EMBEDDING_PCS, n_embeddings_dim)

        pca_embeddings = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
        embeddings_pcs = pca_embeddings.fit_transform(embeddings_scaled)
        full_cell_embedding_map = dict(zip(embedding_cell_ids_full, embeddings_pcs))

        for cell_id in unique_cells_in_l1000:
            if cell_id in full_cell_embedding_map:
                cell2vec[cell_id] = full_cell_embedding_map[cell_id]

        if not cell2vec:
            raise ValueError("No common cell IDs between L1000 and embeddings/ctrls.csv.")

        print(f"AIDO context: original {n_embeddings_dim}D → {n_components_for_context}D for {len(cell2vec)} cells")
    else:
        raise ValueError(f"Invalid CONTEXT_MODE: {CONTEXT_MODE}")

    # Context matrix
    print("Building full context matrix...")
    unique_cells = np.sort(list(cell2vec.keys()))
    if unique_cells.size == 0:
        raise RuntimeError("No cell IDs after context loading/filtering.")

    continuous_context_list, other_context_list = [], []
    cell_ids = df['cell_id'].to_numpy()
    start_idx = 0

    for cell_id in unique_cells:
        mask = (cell_ids == cell_id)
        if mask.sum() == 0:
            continue

        continuous_context = np.hstack([
            np.tile(cell2vec[cell_id], (mask.sum(), 1)),
            pert_time[mask],
            pert_dose[mask],
        ])
        other_context = np.hstack([
            pert_dummies.loc[mask].values,  # same index alignment as df
            ignore_time[mask],
            ignore_dose[mask],
        ])

        continuous_context_list.append(continuous_context)
        other_context_list.append(other_context)

        start_idx += mask.sum()

    all_continuous_context = np.vstack(continuous_context_list)
    all_other_context = np.vstack(other_context_list)

    print("Scaling continuous context features...")
    scaler_continuous_context = StandardScaler()
    all_continuous_context_scaled = scaler_continuous_context.fit_transform(all_continuous_context)
    all_context_scaled = np.hstack([all_continuous_context_scaled, all_other_context])

    print(f"Context dims → continuous: {all_continuous_context.shape[1]}, "
          f"other: {all_other_context.shape[1]}, total: {all_context_scaled.shape[1]}")

    # train/test split
    print("Splitting data into train/test sets (per cell)...")
    X_tr_lst, X_te_lst, C_tr_lst, C_te_lst = [], [], [], []
    cell_tr_lst, cell_te_lst = [], []

    start_idx = 0
    for cell_id in unique_cells:
        mask = (cell_ids == cell_id)
        if mask.sum() < 2:
            continue
        end_idx = start_idx + mask.sum()

        X_cell = X_scaled[mask]
        C_cell = all_context_scaled[start_idx:end_idx]
        ids_cell = cell_ids[mask]

        X_tr, X_te, C_tr, C_te, ids_tr, ids_te = train_test_split(
            X_cell, C_cell, ids_cell, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

        X_tr_lst.append(X_tr); X_te_lst.append(X_te)
        C_tr_lst.append(C_tr); C_te_lst.append(C_te)
        cell_tr_lst.append(ids_tr); cell_te_lst.append(ids_te)

        start_idx = end_idx

    if not X_tr_lst or not X_te_lst:
        raise RuntimeError("No data collected for training/testing after splits.")

    X_train = np.vstack(X_tr_lst)
    X_test  = np.vstack(X_te_lst)
    C_train = np.vstack(C_tr_lst)
    C_test  = np.vstack(C_te_lst)
    cell_ids_train = np.concatenate(cell_tr_lst)
    cell_ids_test  = np.concatenate(cell_te_lst)

    print(f'Context matrix: train {C_train.shape}, test {C_test.shape}')

    print("Applying PCA to gene expression data...")
    pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
    X_train_pca = pca_data.fit_transform(X_train)
    X_test_pca  = pca_data.transform(X_test)

    mu, sigma   = X_train_pca.mean(0), X_train_pca.std(0)  # latent z-score params
    X_train     = (X_train_pca - mu) / sigma
    X_test      = (X_test_pca  - mu) / sigma
    print(f"Gene expression: {N_DATA_PCS} PCs + latent z-score")

    train_group_ids = cell_ids_train
    test_group_ids  = cell_ids_test

    print("Training baseline models...")
    pop_model = CorrelationNetwork()
    pop_model.fit(X_train)
    print(f"Population model - Train MSE: {pop_model.measure_mses(X_train).mean():.6f}")
    print(f"Population model - Test  MSE: {pop_model.measure_mses(X_test).mean():.6f}")

    grouped_model = GroupedNetworks(CorrelationNetwork)
    grouped_model.fit(X_train, train_group_ids)
    print(f"Grouped model - Train MSE: {grouped_model.measure_mses(X_train, train_group_ids).mean():.6f}")
    print(f"Grouped model - Test  MSE: {grouped_model.measure_mses(X_test,  test_group_ids).mean():.6f}")

    print("Training contextualized model...")

    model_kwargs = dict(
        context_dim=C_train.shape[1],
        x_dim=X_train.shape[1],
        encoder_type='mlp',
        num_archetypes=50,
    )
    contextualized_model = ContextualizedCorrelation(**model_kwargs)

    X_train, X_val, C_train, C_val = train_test_split(
        X_train, C_train, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    ckpt_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, filename='best_model')

    if torch.cuda.is_available():
        accelerator, devices = 'gpu', 1
        print("CUDA available: Using GPU")
    else:
        accelerator, devices = 'cpu', 'auto'
        print("CUDA not available: Using CPU")

    C_pred_placeholder = C_train[:1].copy()
    X_pred_placeholder = X_train[:1].copy()

    datamodule = CorrelationDataModule(
        C_train=C_train, X_train=X_train,
        C_val=C_val,     X_val=X_val,
        C_test=C_test,   X_test=X_test,
        C_predict=C_pred_placeholder,
        X_predict=X_pred_placeholder,
        batch_size=32,
    )

    logger = None
    try:
        import wandb  # noqa
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project='contextpert', name='cell_line_context', log_model=True, save_dir=str(output_path / 'logs'))
    except Exception as e:
        print(f"wandb logger not enabled: {e}")

    trainer = Trainer(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        callbacks=[ckpt_cb],
        logger=logger,
    )

    trainer.fit(contextualized_model, datamodule=datamodule)
    print("Testing model performance...")
    trainer.test(contextualized_model, datamodule=datamodule)

    best_ckpt = ckpt_cb.best_model_path
    print(f"Best model saved at: {best_ckpt}")

    contextualized_model = ContextualizedCorrelation.load_from_checkpoint(best_ckpt, **model_kwargs)
    contextualized_model.eval()

    print("Processing disease samples with aligned transforms...")

    df_full = pd.read_csv(PATH_L1000, engine='pyarrow')
    df_full = df_full[df_full['pert_type'].isin(pert_to_fit_on)]
    bad_full = (
        (df_full['distil_cc_q75'] < 0.2) | (df_full['distil_cc_q75'] == -666) | (df_full['distil_cc_q75'].isna()) |
        (df_full['pct_self_rank_q25'] > 5) | (df_full['pct_self_rank_q25'] == -666) | (df_full['pct_self_rank_q25'].isna())
    )
    df_full = df_full[~bad_full]

    df_full['ignore_flag_pert_time'] = (df_full['pert_time'] == -666).astype(int)
    df_full['ignore_flag_pert_dose'] = (df_full['pert_dose'] == -666).astype(int)
    df_full['pert_time'] = df_full['pert_time'].replace(-666, pert_time_mean)
    df_full['pert_dose'] = df_full['pert_dose'].replace(-666, pert_dose_mean)

    has_disease = df_full['diseaseName'].notna() \
        & (df_full['diseaseName'].astype(str).str.strip() != '') \
        & (df_full['diseaseName'].astype(str).str.lower() != 'nan')
    disease_samples = df_full[has_disease]

    print(f"Dataset info:")
    print(f"  • Total samples: {len(df_full)}")
    print(f"  • Samples with disease names: {len(disease_samples)}")

    if len(disease_samples) > 0:
        print("\nFiltering out generic disease terms...")
        disease_labels_raw = disease_samples['diseaseName'].values
        non_generic_mask = filter_generic_diseases(disease_labels_raw)
        disease_samples = disease_samples[non_generic_mask]

        print(f"  • After filtering: {len(disease_samples)} samples")
        disease_counts = disease_samples['diseaseName'].value_counts()
        print(f"  • Unique diseases: {len(disease_counts)}")
        print(f"  • Top diseases: {dict(disease_counts.head(5))}")

        disease_expressions, disease_contexts, disease_names = [], [], []
        dummy_col_set = set(pert_dummy_cols)

        for _, row in disease_samples.iterrows():
            cell_id = row['cell_id']
            if cell_id not in cell2vec:
                # unseen cell embedding, skip safely
                continue

            # expression: same pipeline as training
            try:
                x_raw_row = row[feature_cols].to_numpy().reshape(1, -1)
            except KeyError:
                continue

            x_scaled_row = scaler_X.transform(x_raw_row)
            x_pca_row    = pca_data.transform(x_scaled_row)
            x_norm_row   = (x_pca_row - mu) / sigma  # latent z-score

            # context
            continuous_context_row = np.hstack([
                cell2vec[cell_id],
                [row['pert_time']],
                [row['pert_dose']],
            ]).reshape(1, -1)
            continuous_scaled_row = scaler_continuous_context.transform(continuous_context_row)

            pert_vec = np.zeros((1, len(pert_dummy_cols)), dtype=np.float32)
            pid = row['pert_id']
            if pid in dummy_col_set:
                j = pert_dummy_cols.index(pid)
                pert_vec[0, j] = 1.0
            # else baseline (all zeros), consistent with drop_first=True

            other_context_row = np.hstack([
                pert_vec,
                [[row['ignore_flag_pert_time']]],
                [[row['ignore_flag_pert_dose']]],
            ])

            c_row = np.hstack([continuous_scaled_row, other_context_row])

            disease_expressions.append(x_norm_row.squeeze(0))
            disease_contexts.append(c_row.squeeze(0))
            disease_names.append(row['diseaseName'])

        disease_expressions = np.asarray(disease_expressions, dtype=np.float32)
        disease_contexts   = np.asarray(disease_contexts,   dtype=np.float32)
        disease_names      = np.asarray(disease_names,      dtype=object)

        print(f"\nPrepared {len(disease_expressions)} aligned disease rows for prediction")

        if len(disease_expressions) > 0:
            # predict with deterministic order
            batch_datamodule = CorrelationDataModule(
                C_train=C_train, X_train=X_train,
                C_val=C_val,     X_val=X_val,
                C_test=C_test,   X_test=X_test,
                C_predict=disease_contexts,
                X_predict=disease_expressions,
                batch_size=64,
            )

            print("Running prediction for aligned disease samples...")
            batch_trainer = Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1 if torch.cuda.is_available() else 'auto',
                enable_progress_bar=True,
                logger=False,
            )
            prediction_results = batch_trainer.predict(contextualized_model, datamodule=batch_datamodule)

            # extract predictions preserving order
            print("Extracting predictions...")
            all_correlations = []
            for batch_result in prediction_results:
                # expect dict with 'correlations'
                corr = batch_result['correlations'].detach().cpu().numpy()
                all_correlations.append(corr)
            all_correlations = np.concatenate(all_correlations, axis=0).astype(np.float32)

            # sanity check alignment
            n = len(disease_expressions)
            assert all_correlations.shape[0] == n == len(disease_contexts) == len(disease_names), \
                "Mismatch in lengths between predictions and inputs!"

            print(f"Processed {n} disease samples")

            # save aligned arrays
            final_file = output_path / 'simple_aligned_predictions.npz'
            np.savez(
                final_file,
                expressions=disease_expressions,
                correlations=all_correlations,
                contexts=disease_contexts,
                disease_labels=disease_names,
            )

            saved_data = np.load(final_file, allow_pickle=True)
            print(f"\nDisease processing complete and saved to: {final_file}")
            for key in saved_data.files:
                val = saved_data[key]
                if key == 'disease_labels':
                    print(f"  • {key}: {len(val)} items")
                else:
                    print(f"  • {key}: {val.shape}")
        else:
            print("No valid aligned disease samples found after filtering.")
    else:
        print("No samples with disease names found!")

    print("\nSaving training data and results...")
    training_data_file = save_training_data(
        C_train, C_test, X_train, X_test, unique_cells, cell_ids_train, cell_ids_test, output_path
    )

    results_data = {
        'checkpoint_path': str(best_ckpt),
        'aligned_data_path': str(output_path / 'simple_aligned_predictions.npz'),
        'note': 'Per-row transforms; predictions aligned 1:1 with expressions/labels'
    }
    results_file = output_path / 'training_results.npz'
    np.savez(results_file, **results_data)
    print(f"Saved training results to {results_file}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Training data: {training_data_file}")
    print(f"Disease samples: {output_path / 'simple_aligned_predictions.npz'}")

    print(f"\nModel Performance Summary:")
    print(f"  • Population model test MSE: {pop_model.measure_mses(X_test).mean():.6f}")
    print(f"  • Grouped model test MSE:    {grouped_model.measure_mses(X_test, test_group_ids).mean():.6f}")
    print(f"  • Contextualized model: training complete; aligned correlations saved")

    return {
        'checkpoint_path': best_ckpt,
        'training_data_file': training_data_file,
        'results_file': results_file,
        'aligned_data_file': output_path / 'simple_aligned_predictions.npz',
        'output_dir': output_path
    }


if __name__ == "__main__":
    _ = main()
