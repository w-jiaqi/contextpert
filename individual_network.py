
import pandas as pd
import numpy as np
from contextualized.easy import ContextualizedCorrelationNetworks
from contextualized.baselines.networks import GroupedNetworks, CorrelationNetwork
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Load and preprocess the dataset with explicit dtype handling
df = pd.read_csv('/home/saddagud/pert/merged_output4.csv')

# Filter controls
controls = ['ctl_vehicle', 'ctl_vector', 'ctl_untrt']
mask = df['pert_type'].isin(controls)
df = df[mask]

# Condition to drop rows
condition = (
    (df['distil_cc_q75'] < 0.2) |
    (df['distil_cc_q75'] == -666) |
    (df['pct_self_rank_q25'] > 5) |
    (df['pct_self_rank_q25'] == -666)
)
df = df[~condition]
print(df)
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
# Initialize lists to store split data
X_train_list = []
X_test_list = []
C_train_list = []
C_test_list = []

only1 = 0
# Split data within each context group
for cell_id in unique_cells:
    # Get indices for current cell type
    cell_mask = cell_ids == cell_id
    X_cell = X[cell_mask]
    
    # Create one-hot encoding for current cell type
    C_cell = np.zeros((X_cell.shape[0], len(unique_cells)))
    C_cell[:, np.where(unique_cells == cell_id)[0]] = 1
    
    # Split data for current cell type
    if X_cell.shape[0] > 3:  # Only split if we have more than one sample
        X_train_cell, X_test_cell, C_train_cell, C_test_cell = train_test_split(
            X_cell, C_cell, test_size=0.33, random_state=42
        )
        
        X_train_list.append(X_train_cell)
        X_test_list.append(X_test_cell)
        C_train_list.append(C_train_cell)
        C_test_list.append(C_test_cell)
    else:
        only1 = only1 + 1
print('how many cells killed: ', only1)

# Combine split data
X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
C_train = np.vstack(C_train_list)
C_test = np.vstack(C_test_list)

# Apply PCA
pca = PCA(n_components=50)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Normalize train and test PCA data
X_mean = X_train_pca.mean(axis=0)
X_std = X_train_pca.std(axis=0)
X_train = (X_train_pca - X_mean) / X_std
X_test = (X_test_pca - X_mean) / X_std

# Convert one-hot encoded context matrix to group labels
train_labels = np.argmax(C_train, axis=1)
test_labels = np.argmax(C_test, axis=1)

# Initialize and fit the grouped correlation networks
grouped_cn = GroupedNetworks(CorrelationNetwork)
grouped_cn.fit(X_train, train_labels)

# Calculate MSE for test set
mse_train = grouped_cn.measure_mses(X_train, train_labels)
mse_test = grouped_cn.measure_mses(X_test, test_labels)

# Calculate and print metrics
print("\nPerformance Metrics:")
print("-" * 50)

# Overall MSE
avg1_mse = np.mean(mse_train)
print(f"\nOverall MSE on train set: {avg1_mse:.4f}")

avg_mse = np.mean(mse_test)
print(f"\nOverall MSE on test set: {avg_mse:.4f}")


# Per-context metrics
print("\nPer-Context Performance:")
print("-" * 50)
unique_contexts = np.unique(test_labels)
for context in unique_contexts:
    context_mask = test_labels == context
    context_mse = np.mean(mse_test[context_mask])
    n_samples = np.sum(context_mask)
    cell_id = unique_cells[context]
    print(f"Cell ID {cell_id}:")
    print(f"  MSE = {context_mse:.4f}")
    print(f"  Test samples = {n_samples}")
    print(f"  Train samples = {np.sum(train_labels == context)}")
