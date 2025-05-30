
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from contextualized.easy import ContextualizedCorrelationNetworks


PATH_L1000   = '/home/user/contextulized/lincs1000.csv'
PATH_CTLS    = '/home/user/contextulized/ctrls.csv'      
N_CTRL_PCS   = 20    
N_DATA_PCS   = 50    
CELL_TEST_SIZE = 0.20 # Proportion of unique cell types to use for testing
RANDOM_STATE = 41


df = pd.read_csv(PATH_L1000)

# keep only compound perturbations
pert_to_fit_on = ['trt_sh']
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

# replace sentinel –666 with column mean
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

# contexts
pert_dummies       = pd.get_dummies(df['pert_id'],       drop_first=True)
pert_unit_dummies  = pd.get_dummies(df['pert_dose_unit'], drop_first=True)

pert_time   = df['pert_time'  ].to_numpy().reshape(-1, 1)
pert_dose   = df['pert_dose'  ].to_numpy().reshape(-1, 1)
ignore_time = df['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
ignore_dose = df['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)

# using ctrl expression as cell type context
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

# train test

# Get unique cell IDs for splitting
all_unique_cell_ids = df['cell_id'].unique()

# Split unique cell IDs into training and testing sets
train_cell_ids, test_cell_ids = train_test_split(
    all_unique_cell_ids, test_size=CELL_TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Total unique cells: {len(all_unique_cell_ids)}")
print(f"Cells for training: {len(train_cell_ids)}")
print(f"Cells for testing: {len(test_cell_ids)}")
print(f"Test cell IDs: {test_cell_ids}")

# Create masks to select rows based on cell IDs
train_mask = df['cell_id'].isin(train_cell_ids)
test_mask  = df['cell_id'].isin(test_cell_ids)


# Get cell vector for each row
cell_vecs_for_all_rows = np.array([cell2vec[cid] for cid in df['cell_id'].values])

C_full = np.hstack([
    cell_vecs_for_all_rows,
    pert_dummies.values,
    pert_unit_dummies.values,
    pert_time,
    pert_dose,
    ignore_time,
    ignore_dose,
])

# Apply masks to get training and testing sets
X_train = X_scaled[train_mask]
X_test  = X_scaled[test_mask]

C_train = C_full[train_mask]
C_test  = C_full[test_mask]

cell_ids_train = df['cell_id'].values[train_mask]
cell_ids_test  = df['cell_id'].values[test_mask]

print(f'Context matrix:   train {C_train.shape}   test {C_test.shape}')
print(f'Signature matrix: train {X_train.shape}   test {X_test.shape}')


pca_data = PCA(n_components=N_DATA_PCS)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca  = pca_data.transform(X_test)


mu, sigma = X_train_pca.mean(0), X_train_pca.std(0)
X_train_norm = (X_train_pca - mu) / sigma
X_test_norm  = (X_test_pca  - mu) / sigma

# train ccn
ccn = ContextualizedCorrelationNetworks(
    encoder_type='mlp',
    num_archetypes=50,
    n_bootstraps=1
)
ccn.fit(C_train, X_train_norm)

# evals
mse_train = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
mse_test  = ccn.measure_mses(C_test,  X_test_norm,  individual_preds=False)

print('\n────────────────────────────────────────────────────────')
print('Performance (mean squared error)')
print(f'  overall train MSE : {mse_train.mean():.4f}')
print(f'  overall  test MSE : {mse_test.mean():.4f}')
print('────────────────────────────────────────────────────────\n')


for cell_id in all_unique_cell_ids: 
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test  == cell_id
    
    # Check if this cell_id is actually in the training or testing set
    if tr_mask.sum() == 0 and te_mask.sum() == 0:
        continue 

    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test [te_mask].mean() if te_mask.any() else np.nan
    print(f'Cell {cell_id:<15}:  train MSE = {tr_mse:7.4f}   '
          f'test MSE = {te_mse:7.4f}   (n={tr_mask.sum():3d}/{te_mask.sum():3d})')
