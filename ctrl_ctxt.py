#code for table 2 (post perturbation networks with full context, PCA compressed average control expression for cell line context)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from contextualized.easy import ContextualizedCorrelationNetworks


PATH_L1000   = '/home/user/contextulized/lincs1000.csv'
PATH_CTLS    = '/home/user/contextulized/ctrls.csv'      # rows = cell_id, cols = genes
N_CTRL_PCS   = 20    # number of PCs for control profiles
N_DATA_PCS   = 50    # number of PCs for signature features
TEST_SIZE    = 0.33  
RANDOM_STATE = 42

# load
df = pd.read_csv(PATH_L1000, engine='pyarrow')

# keep only compound perturbations
pert_to_fit_on = ['trt_cp']
df = df[df['pert_type'].isin(pert_to_fit_on)]

# quality filters
bad = (
    (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
)
df = df[~bad]

# ignore-flag columns for missing meta-data
df['ignore_flag_pert_time'] = (df['pert_time'] == -666).astype(int)
df['ignore_flag_pert_dose'] = (df['pert_dose'] == -666).astype(int)

# replace  –666 with column mean
for col in ['pert_time', 'pert_dose']:
    mean_val = df.loc[df[col] != -666, col].mean()
    df[col] = df[col].replace(-666, mean_val)

# X
numeric_cols   = df.select_dtypes(include=[np.number]).columns
drop_cols      = ['pert_dose', 'pert_dose_unit', 'pert_time',
                  'distil_cc_q75', 'pct_self_rank_q25']
feature_cols   = [c for c in numeric_cols if c not in drop_cols]
X_raw          = df[feature_cols].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)                 # shape (N, p)

#contexts
pert_dummies       = pd.get_dummies(df['pert_id'],       drop_first=True)
pert_unit_dummies  = pd.get_dummies(df['pert_dose_unit'], drop_first=True)

pert_time   = df['pert_time'  ].to_numpy().reshape(-1, 1)
pert_dose   = df['pert_dose'  ].to_numpy().reshape(-1, 1)
ignore_time = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
ignore_dose = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)

# cell expr udd stuff
ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)           # index = cell_id
unique_cells = np.sort(df['cell_id'].unique())
ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells)]

scaler_ctrls = StandardScaler()
ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

n_cells         = ctrls_scaled.shape[0]
N_CTRL_PCS  = min(N_CTRL_PCS, n_cells)               

pca_ctrls = PCA(n_components=N_CTRL_PCS)
ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)        # shape (#cells, N_CTRL_PCS)

cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

# tra9n test
X_tr_lst, X_te_lst = [], []
C_tr_lst, C_te_lst = [], []
cell_tr_lst, cell_te_lst = [], []

cell_ids = df['cell_id'].to_numpy()

for cell_id in unique_cells:
    mask = cell_ids == cell_id
    if mask.sum() == 0:
        continue

    X_cell   = X_scaled[mask]
    ids_cell = cell_ids[mask]

    # same order as above for every sample
    C_cell = np.hstack([
        np.tile(cell2vec[cell_id], (mask.sum(), 1)),  # PCA control profile
        pert_dummies.loc[mask].values,
        pert_time[mask],
        pert_dose[mask],
        ignore_time[mask],
        ignore_dose[mask],
    ])

    X_tr, X_te, C_tr, C_te, ids_tr, ids_te = train_test_split(
        X_cell, C_cell, ids_cell,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_tr_lst.append(X_tr); X_te_lst.append(X_te)
    C_tr_lst.append(C_tr); C_te_lst.append(C_te)
    cell_tr_lst.append(ids_tr); cell_te_lst.append(ids_te)

# concatenate splits across cells
X_train = np.vstack(X_tr_lst);  X_test = np.vstack(X_te_lst)
C_train = np.vstack(C_tr_lst);  C_test = np.vstack(C_te_lst)
cell_ids_train = np.concatenate(cell_tr_lst)
cell_ids_test  = np.concatenate(cell_te_lst)

print(f'Context matrix:   train {C_train.shape}   test {C_test.shape}')

# pca
pca_data = PCA(n_components=N_DATA_PCS)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca  = pca_data.transform(X_test)

# z-score in latent space
mu, sigma = X_train_pca.mean(0), X_train_pca.std(0)
X_train_norm = (X_train_pca - mu) / sigma
X_test_norm  = (X_test_pca  - mu) / sigma

# fit ccn
ccn = ContextualizedCorrelationNetworks(
    encoder_type='mlp',
    num_archetypes=50,
    n_bootstraps=1
)
ccn.fit(C_train, X_train_norm)

#evals
mse_train = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
mse_test  = ccn.measure_mses(C_test,  X_test_norm,  individual_preds=False)

print('\n────────────────────────────────────────────────────────')
print('Performance (mean squared error)')
print(f'  overall train MSE : {mse_train.mean():.4f}')
print(f'  overall  test MSE : {mse_test.mean():.4f}')
print('────────────────────────────────────────────────────────\n')

for cell_id in unique_cells:
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test  == cell_id
    if tr_mask.sum() == 0 and te_mask.sum() == 0:
        continue

    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test [te_mask].mean() if te_mask.any() else np.nan
    print(f'Cell {cell_id:<15}:  train MSE = {tr_mse:7.4f}   '
          f'test MSE = {te_mse:7.4f}   (n={tr_mask.sum():3d}/{te_mask.sum():3d})')
