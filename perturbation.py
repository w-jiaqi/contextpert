import pandas as pd
import numpy as np
from contextualized.easy import ContextualizedCorrelationNetworks
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
df = pd.read_csv('/home/saddagud/pert/merged_output4.csv')

# Filter controls
controls = ['trt_sh']
mask = df['pert_type'].isin(controls)
df = df[mask]

# Condition to drop rows
condition = (
    (df['distil_cc_q75'] < 0.2) |
    (df['distil_cc_q75'] == -666) |
    (df['distil_cc_q75'].isna()) |  # Check for NaN
    (df['pct_self_rank_q25'] > 5) |
    (df['pct_self_rank_q25'] == -666) |
    (df['pct_self_rank_q25'].isna())  # Check for NaN
)
df = df[~condition]

pert_dummies = pd.get_dummies(df['pert_id'], drop_first=True)
pert_unit_dummies = pd.get_dummies(df['pert_dose_unit'], drop_first=True)

# Extract numeric columns as features
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
columns_to_drop = ['pert_dose', 'pert_dose_unit', 'pert_time', 'distil_cc_q75', 'pct_self_rank_q25']
feature_cols = [col for col in feature_cols if col not in columns_to_drop]
feature_df = df[feature_cols]

# Scale features
scaler = StandardScaler()
scaler.fit(feature_df)
feature_df = scaler.transform(feature_df)
X = feature_df

# Create context matrix and get unique cell IDs
cell_ids = df['cell_id'].values
unique_cells = np.unique(cell_ids)
print(len(cell_ids))
print(cell_ids)

# Initialize lists to store split data
X_train_list = []
X_test_list = []
C_train_list = []
C_test_list = []
cell_ids_train_list = []
cell_ids_test_list = []

# Add separate ignore flag columns for 'pert_time' and 'pert_dose'
df['ignore_flag_pert_time'] = np.where(df['pert_time'] == -666, 1, 0)
df['ignore_flag_pert_dose'] = np.where(df['pert_dose'] == -666, 1, 0)

# Replace -666 in 'pert_time' and 'pert_dose' with column mean for those rows
for col in ['pert_time', 'pert_dose']:
    mean_value = df[df[col] != -666][col].mean()  # Calculate mean excluding -666
    df[col] = df[col].replace(-666, mean_value)

print(df[['pert_type','pert_time', 'pert_dose']])
print(df[['ignore_flag_pert_time', 'ignore_flag_pert_dose']])

# Generate perturbation dummies and time/dose values
pert_time = df['pert_time'].values
pert_dose = df['pert_dose'].values
ignore_time = df['ignore_flag_pert_time'].values
ignore_dose = df['ignore_flag_pert_dose'].values
print(pert_dummies.shape)
print(pert_unit_dummies)

# Initialize lists to store split data
X_train_list = []
X_test_list = []
C_train_list = []
C_test_list = []
cell_ids_train_list = []
cell_ids_test_list = []

only1 = 0
# Split data within each context group
for cell_id in unique_cells:
    # Get indices for current cell type
    cell_mask = cell_ids == cell_id
    X_cell = X[cell_mask]
    cell_ids_cell = cell_ids[cell_mask]
    
    # Get corresponding perturbation information
    pert_dummies_cell = pert_dummies.loc[cell_mask].values
    pert_unit_dummies_cell = pert_unit_dummies.loc[cell_mask].values
    pert_time_cell = pert_time[cell_mask].reshape(-1, 1)
    pert_dose_cell = pert_dose[cell_mask].reshape(-1, 1)
    ignore_time_cell = ignore_time[cell_mask].reshape(-1, 1)
    ignore_dose_cell = ignore_dose[cell_mask].reshape(-1, 1)
    
    # Create one-hot encoding for current cell type
    C_cell = np.zeros((X_cell.shape[0], len(unique_cells)))
    C_cell[:, np.where(unique_cells == cell_id)[0]] = 1
    
    # Concatenate all context information
    # C_cell = np.hstack([C_cell, pert_dummies_cell, pert_unit_dummies_cell, pert_time_cell, pert_dose_cell, ignore_time_cell, ignore_dose_cell])
    C_cell = np.hstack([C_cell, pert_dummies_cell])

    # Split data for current cell type
    if X_cell.shape[0] > 0:  # Only split if we have more than three samples
        X_train_cell, X_test_cell, C_train_cell, C_test_cell, ids_train_cell, ids_test_cell = train_test_split(
            X_cell, C_cell, cell_ids_cell, test_size=0.33, random_state=42
        )
        
        X_train_list.append(X_train_cell)
        X_test_list.append(X_test_cell)
        C_train_list.append(C_train_cell)
        C_test_list.append(C_test_cell)
        cell_ids_train_list.append(ids_train_cell)
        cell_ids_test_list.append(ids_test_cell)
    else:
        only1 = only1 + 1
print('how many cells killed: ', only1)

# Combine split data
# np.set_printoptions(threshold=np.inf)
X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
C_train = np.vstack(C_train_list)
C_test = np.vstack(C_test_list)
cell_ids_train = np.concatenate(cell_ids_train_list)
cell_ids_test = np.concatenate(cell_ids_test_list)
print(C_train.shape)
print(C_test.shape)
print(C_train)

# Apply PCA
pca = PCA(n_components=50)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Normalize train and test PCA data
X_mean = X_train_pca.mean(axis=0)
X_std = X_train_pca.std(axis=0)
X_train_norm = (X_train_pca - X_mean) / X_std
X_test_norm = (X_test_pca - X_mean) / X_std

# Fit CCN model
ccn = ContextualizedCorrelationNetworks(encoder_type='mlp', num_archetypes=50, n_bootstraps=1)
ccn.fit(C_train, X_train_norm)

# Calculate MSE for train and test sets
mse_train = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
mse_test = ccn.measure_mses(C_test, X_test_norm, individual_preds=False)

# Calculate and print metrics
print("\nPerformance Metrics:")
print("-" * 50)

# Overall MSE
print(f"\nOverall MSE on train set: {np.mean(mse_train):.4f}")
print(f"Overall MSE on test set: {np.mean(mse_test):.4f}")

# Per-context metrics
print("\nPer-Context Performance:")
print("-" * 50)

for cell_id in unique_cells:
    # Training set metrics
    train_mask = cell_ids_train == cell_id
    if np.sum(train_mask) > 0:
        train_mse = np.mean(mse_train[train_mask])
        n_train = np.sum(train_mask)
        
        # Test set metrics
        test_mask = cell_ids_test == cell_id
        if np.sum(test_mask) > 0:
            test_mse = np.mean(mse_test[test_mask])
            n_test = np.sum(test_mask)
            
            print(f"Cell ID {cell_id}:")
            print(f"  Train MSE = {train_mse:.4f}")
            print(f"  Test MSE = {test_mse:.4f}")
            print(f"  Train samples = {n_train}")
            print(f"  Test samples = {n_test}")
            print()
