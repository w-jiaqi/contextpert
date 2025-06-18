# code for table 2 (post perturbation with full context w PCA compressed average control expression or AIDO control embeddings for cell line context)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from contextualized.easy import ContextualizedCorrelationNetworks
import os

# expression for pca compressed avg control expression, embeddings for aido embeddings for cell line context

# CONTEXT_MODE = 'expression' 
CONTEXT_MODE = 'embeddings' 

# File Paths
PATH_L1000 = '/home/user/contextulized/lincs1000.csv'
PATH_CTLS = '/home/user/contextulized/ctrls.csv'
EMB_FILE = '/home/user/contextulized/embeddings/aido_cell_100m_lincs_embeddings.npy'

N_DATA_PCS = 50    
TEST_SIZE = 0.33
RANDOM_STATE = 42

N_CTRL_PCS = 20     
N_EMBEDDING_PCS = 20 

#specify type of perturbation to fit on
pert_to_fit_on = ['trt_cp']


# Validate file paths
if not os.path.exists(PATH_L1000):
    raise FileNotFoundError(f"L1000 data not found at {PATH_L1000}")
if not os.path.exists(PATH_CTLS):
    raise FileNotFoundError(f"Controls data not found at {PATH_CTLS}")
if CONTEXT_MODE == 'embeddings' and not os.path.exists(EMB_FILE):
    raise FileNotFoundError(f"Embeddings file not found at {EMB_FILE}")

print(f"Using cell line context mode: {CONTEXT_MODE}\n")

# Load L1000 data
df = pd.read_csv(PATH_L1000, engine='pyarrow')

df = df[df['pert_type'].isin(pert_to_fit_on)]

# Quality filters
bad = (
    (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
)
df = df[~bad]

# Ignore-flag columns for missing meta-data
df['ignore_flag_pert_time'] = (df['pert_time'] == -666).astype(int)
df['ignore_flag_pert_dose'] = (df['pert_dose'] == -666).astype(int)

# Replace –666 with column mean
for col in ['pert_time', 'pert_dose']:
    mean_val = df.loc[df[col] != -666, col].mean()
    df[col] = df[col].replace(-666, mean_val)

# Get X (gene expression data)
numeric_cols = df.select_dtypes(include=[np.number]).columns
drop_cols = ['pert_dose', 'pert_dose_unit', 'pert_time',
             'distil_cc_q75', 'pct_self_rank_q25']
feature_cols = [c for c in numeric_cols if c not in drop_cols]
X_raw = df[feature_cols].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)  # shape (N, p)

# Get context components
pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True)

pert_time = df['pert_time'].to_numpy().reshape(-1, 1)
pert_dose = df['pert_dose'].to_numpy().reshape(-1, 1)
ignore_time = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
ignore_dose = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)


cell2vec = {}
unique_cells_in_l1000 = np.sort(df['cell_id'].unique())

if CONTEXT_MODE == 'expression':
    print("Preparing cell line context using PCA of control expression...")
    ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)  # index = cell_id
    
    # Filter controls to only include cells present in the L1000 dataset
    ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_in_l1000)]
    
    if ctrls_df.empty:
        raise ValueError("No common cell IDs found between lincs1000.csv and ctrls.csv for PCA control expression.")

    scaler_ctrls = StandardScaler()
    ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

    n_cells = ctrls_scaled.shape[0]
    n_components_for_context = min(N_CTRL_PCS, n_cells)

    pca_ctrls = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
    ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)  # shape (#cells, N_CTRL_PCS)

    cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

elif CONTEXT_MODE == 'embeddings':
    all_embeddings_raw = np.load(EMB_FILE)

    # Use ctrls.csv to map embedding rows to cell IDs
    ctrls_meta_df = pd.read_csv(PATH_CTLS, index_col=0)
    embedding_cell_ids_full = ctrls_meta_df.index.to_numpy()

    if len(embedding_cell_ids_full) != all_embeddings_raw.shape[0]:
        raise ValueError(
            f"Mismatch: embeddings file '{EMB_FILE}' has {all_embeddings_raw.shape[0]} entries, "
            f"but ctrls.csv has {len(embedding_cell_ids_full)} cell IDs. "
            "Please ensure they correspond row-wise."
        )

    # Z-score normalize embeddings
    scaler_embeddings = StandardScaler()
    embeddings_scaled = scaler_embeddings.fit_transform(all_embeddings_raw)

    # Apply PCA to embeddings
    n_embeddings_dim = embeddings_scaled.shape[1]
    n_components_for_context = min(N_EMBEDDING_PCS, n_embeddings_dim)

    pca_embeddings = PCA(n_components=n_components_for_context, random_state=RANDOM_STATE)
    embeddings_pcs = pca_embeddings.fit_transform(embeddings_scaled)

    # Create a mapping from cell_id to its processed embedding vector for all loaded embeddings
    full_cell_embedding_map = dict(zip(embedding_cell_ids_full, embeddings_pcs))

    # Filter to only include cells present in the L1000 dataset
    for cell_id in unique_cells_in_l1000:
        if cell_id in full_cell_embedding_map:
            cell2vec[cell_id] = full_cell_embedding_map[cell_id]

    if not cell2vec:
        raise ValueError(
            "No common cell IDs found between lincs1000.csv and embeddings/ctrls.csv. "
            "Cannot proceed. Please check your data files."
        )

    print(f"AIDO embeddings context: Original dim {n_embeddings_dim}, "
          f"now {n_components_for_context}D after scaling and PCA for {len(cell2vec)} unique cells.")

else:
    raise ValueError(f"Invalid CONTEXT_MODE: {CONTEXT_MODE}. Choose 'expression' or 'embeddings'.")

# Update the list of unique cells to process, based on what's available in cell2vec
unique_cells = np.sort(list(cell2vec.keys()))

if not unique_cells.shape[0] > 0:
    raise RuntimeError("No cell IDs found to process after context loading and filtering. Check data consistency.")

continuous_context_list = []
other_context_list = []
cell_ids = df['cell_id'].to_numpy()

for cell_id in unique_cells:
    mask = cell_ids == cell_id
    if mask.sum() == 0:
        continue
        
    if mask.sum() < 2:  # At least 2 samples needed for a train/test split
        print(f"Skipping cell {cell_id} due to insufficient samples ({mask.sum()}). Needs at least 2 for split.")
        continue

    # Continuous context features: cell context + pert_time + pert_dose
    continuous_context = np.hstack([
        np.tile(cell2vec[cell_id], (mask.sum(), 1)),  # Cell-specific context (PCA of expr or embeddings)
        pert_time[mask],                              
        pert_dose[mask],                              
    ])
    
    # Other (categorical/binary) context features
    other_context = np.hstack([
        pert_dummies.loc[mask].values,                # Perturbation ID (one-hot encoded)
        ignore_time[mask],                            
        ignore_dose[mask],                            
    ])
    
    continuous_context_list.append(continuous_context)
    other_context_list.append(other_context)

# Concatenate all continuous context features across cells
all_continuous_context = np.vstack(continuous_context_list)
all_other_context = np.vstack(other_context_list)

# Scale the continuous context features together
print("Scaling continuous context features...")
scaler_continuous_context = StandardScaler()
all_continuous_context_scaled = scaler_continuous_context.fit_transform(all_continuous_context)

# Combine scaled continuous context with other context features
all_context_scaled = np.hstack([all_continuous_context_scaled, all_other_context])

print(f"Context scaling complete. Continuous features scaled: {all_continuous_context.shape[1]}, "
      f"Other features: {all_other_context.shape[1]}, Total: {all_context_scaled.shape[1]}")

# Now split the data
X_tr_lst, X_te_lst = [], []
C_tr_lst, C_te_lst = [], []
cell_tr_lst, cell_te_lst = [], []

start_idx = 0
for i, cell_id in enumerate(unique_cells):
    mask = cell_ids == cell_id
    if mask.sum() == 0:
        continue
        
    if mask.sum() < 2:
        continue

    end_idx = start_idx + mask.sum()
    
    X_cell = X_scaled[mask]
    C_cell = all_context_scaled[start_idx:end_idx]
    ids_cell = cell_ids[mask]
    
    X_tr, X_te, C_tr, C_te, ids_tr, ids_te = train_test_split(
        X_cell, C_cell, ids_cell,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_tr_lst.append(X_tr)
    X_te_lst.append(X_te)
    C_tr_lst.append(C_tr)
    C_te_lst.append(C_te)
    cell_tr_lst.append(ids_tr)
    cell_te_lst.append(ids_te)
    
    start_idx = end_idx

if not X_tr_lst or not X_te_lst:
    raise RuntimeError(
        "No data collected for training/testing after splits. "
        "This might be due to all cells being filtered or having insufficient samples."
    )

# Concatenate splits across cells
X_train = np.vstack(X_tr_lst)
X_test = np.vstack(X_te_lst)
C_train = np.vstack(C_tr_lst)
C_test = np.vstack(C_te_lst)
cell_ids_train = np.concatenate(cell_tr_lst)
cell_ids_test = np.concatenate(cell_te_lst)

print(f'\nContext matrix:   train {C_train.shape}   test {C_test.shape}')

# --- PCA on Gene Expression Data (X) ---
print("Applying PCA to gene expression data...")
pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca = pca_data.transform(X_test)

# Z-score in latent space
mu, sigma = X_train_pca.mean(0), X_train_pca.std(0)
X_train_norm = (X_train_pca - mu) / sigma
X_test_norm = (X_test_pca - mu) / sigma
print(f"Gene expression data: Reduced to {N_DATA_PCS} PCs and Z-score normalized.")

# fit ccn
print("\nFitting Contextualized Correlation Networks model...")
ccn = ContextualizedCorrelationNetworks(
    encoder_type='mlp',  
    num_archetypes=50,
    n_bootstraps=1
)
ccn.fit(C_train, X_train_norm)
print("Model fitting complete.")

# --- Evaluation ---
print("\nEvaluating model performance...")
mse_train = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
mse_test = ccn.measure_mses(C_test, X_test_norm, individual_preds=False)

print('\n────────────────────────────────────────────────────────')
print('Performance (mean squared error)')
print(f'  overall train MSE : {mse_train.mean():.4f}')
print(f'  overall  test MSE : {mse_test.mean():.4f}')
print('────────────────────────────────────────────────────────\n')

# Iterate over the unique cells that were included in the splits for per-cell MSE
print("Per-cell MSE:")
for cell_id in unique_cells:
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test == cell_id

    if tr_mask.sum() == 0 and te_mask.sum() == 0:
        continue

    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
    print(f'Cell {cell_id:<15}:  train MSE = {tr_mse:7.4f}   '
          f'test MSE = {te_mse:7.4f}   (n={tr_mask.sum():3d}/{te_mask.sum():3d})')
