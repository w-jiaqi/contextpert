# code for table 3 (generalizing to unseen small molecule perturbations - cell expression as cell line context, fingerprint for pert context)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, ParameterGrid
from contextualized.easy import ContextualizedCorrelationNetworks
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import warnings

PATH_L1000   = '/home/user/contextulized/trt_cp_smiles.csv' #file filtered with only the trt_cp perts with smiles
PATH_CTLS    = '/home/user/contextulized/ctrls.csv'     
N_DATA_PCS   = 50   
PERTURBATION_HOLDOUT_SIZE = 0.2  
RANDOM_STATE = 42
SUBSAMPLE_FRACTION = None  # None for using full data, or decimal for percent subsample

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)  

# Function to generate Morgan fingerprints from SMILES
def smiles_to_morgan_fp(smiles, generator=morgan_gen):
    """
    Convert SMILES string to Morgan fingerprint using MorganGenerator.
    
    Args:
        smiles (str): SMILES string
        generator: RDKit MorganGenerator instance
    
    Returns:
        np.array: Binary fingerprint array, or array of zeros if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Invalid SMILES: {smiles}")
            return np.zeros(generator.GetOptions().fpSize)
        
        fp = generator.GetFingerprint(mol)
        return np.array(fp)
        # return np.zeros(generator.GetOptions().fpSize)
    except Exception as e:
        warnings.warn(f"Error processing SMILES {smiles}: {str(e)}")
        return np.zeros(generator.GetOptions().fpSize)

#load data
df = pd.read_csv(PATH_L1000, engine='pyarrow')

# pick the perturbation to fit model on here
pert_to_fit_on = ['trt_cp']
df = df[df['pert_type'].isin(pert_to_fit_on)]

# quality filters
bad = (
    (df['distil_cc_q75'] < 0.2) | (df['distil_cc_q75'] == -666) | (df['distil_cc_q75'].isna()) |
    (df['pct_self_rank_q25'] > 5) | (df['pct_self_rank_q25'] == -666) | (df['pct_self_rank_q25'].isna())
)
df = df[~bad]

# Filter out samples with missing SMILES
df = df.dropna(subset=['canonical_smiles'])
df = df[df['canonical_smiles'] != '']

print(f"Processing {len(df)} samples with valid SMILES...")

if SUBSAMPLE_FRACTION is not None:
    df = df.sample(frac=SUBSAMPLE_FRACTION, random_state=RANDOM_STATE)
    print(f"Subsampled to {len(df)} samples ({SUBSAMPLE_FRACTION*100}% of data)")

# PERTURBATION HOLDOUT: Split unique perturbations first
unique_smiles = df['canonical_smiles'].unique()
print(f"Found {len(unique_smiles)} unique perturbations (SMILES)")

# Split unique SMILES into train and test sets
smiles_train, smiles_test = train_test_split(
    unique_smiles, 
    test_size=PERTURBATION_HOLDOUT_SIZE, 
    random_state=RANDOM_STATE
)

print(f"Perturbation split: {len(smiles_train)} train, {len(smiles_test)} test perturbations")

# Create train and test dataframes based on perturbation split
df_train = df[df['canonical_smiles'].isin(smiles_train)].copy()
df_test = df[df['canonical_smiles'].isin(smiles_test)].copy()

print(f"Sample split: {len(df_train)} train, {len(df_test)} test samples")

# Process train/test sets - fit preprocessing on training data only
pert_time_mean = None
pert_dose_mean = None

for df_split, split_name in [(df_train, 'train'), (df_test, 'test')]:
    # ignore-flag columns for missing meta-data
    df_split['ignore_flag_pert_time'] = (df_split['pert_time'] == -666).astype(int)
    df_split['ignore_flag_pert_dose'] = (df_split['pert_dose'] == -666).astype(int)

    # replace –666 with column mean (computed from training set only)
    for col in ['pert_time', 'pert_dose']:
        if split_name == 'train':
            mean_val = df_split.loc[df_split[col] != -666, col].mean()
            # Store the mean for use with val/test sets
            if col == 'pert_time':
                pert_time_mean = mean_val
            else:
                pert_dose_mean = mean_val
        else:
            # Use training set means for test set
            mean_val = pert_time_mean if col == 'pert_time' else pert_dose_mean
        
        df_split[col] = df_split[col].replace(-666, mean_val)

# Function to process data split
def process_data_split(df_split, split_name):
    # Getting X (gene expression features)
    numeric_cols   = df_split.select_dtypes(include=[np.number]).columns
    drop_cols      = ['pert_dose', 'pert_dose_unit', 'pert_time',
                      'distil_cc_q75', 'pct_self_rank_q25']
    feature_cols   = [c for c in numeric_cols if c not in drop_cols]
    X_raw          = df_split[feature_cols].values

    # Generate Morgan fingerprints
    print(f"Generating Morgan fingerprints for {split_name} set...")
    morgan_fps = []
    for smiles in df_split['canonical_smiles']:
        fp = smiles_to_morgan_fp(smiles)
        morgan_fps.append(fp)

    morgan_fps = np.array(morgan_fps)
    print(f"Generated Morgan fingerprints for {split_name}: shape {morgan_fps.shape}")

    # Keep other context features
    pert_unit_dummies  = pd.get_dummies(df_split['pert_dose_unit'], drop_first=True)

    pert_time   = df_split['pert_time'  ].to_numpy().reshape(-1, 1)
    pert_dose   = df_split['pert_dose'  ].to_numpy().reshape(-1, 1)
    ignore_time = df_split['ignore_flag_pert_time'].to_numpy().reshape(-1, 1)
    ignore_dose = df_split['ignore_flag_pert_dose'].to_numpy().reshape(-1, 1)

    return X_raw, morgan_fps, pert_unit_dummies, pert_time, pert_dose, ignore_time, ignore_dose

# Process both splits
X_raw_train, morgan_fps_train, pert_unit_dummies_train, pert_time_train, pert_dose_train, ignore_time_train, ignore_dose_train = process_data_split(df_train, 'train')
X_raw_test, morgan_fps_test, pert_unit_dummies_test, pert_time_test, pert_dose_test, ignore_time_test, ignore_dose_test = process_data_split(df_test, 'test')

print("Applying improved scaling strategy...")

# scaling
scaler_genes = StandardScaler()
X_train_scaled = scaler_genes.fit_transform(X_raw_train)
X_test_scaled = scaler_genes.transform(X_raw_test)
print(f"Gene expression scaled: train {X_train_scaled.shape}, test {X_test_scaled.shape}")

scaler_morgan = StandardScaler()
morgan_train_scaled = morgan_fps_train.astype(float)
morgan_test_scaled = morgan_fps_test.astype(float)
print(f"Morgan fingerprints scaled: train {morgan_train_scaled.shape}, test {morgan_test_scaled.shape}")

# Load and process control data
ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)          # index = cell_id
unique_cells_train = np.sort(df_train['cell_id'].unique())
unique_cells_test = np.sort(df_test['cell_id'].unique())
unique_cells_all = np.sort(np.union1d(unique_cells_train, unique_cells_test))

ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]

# Standardize controls and do PCA
scaler_ctrls = StandardScaler()
ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

n_cells = ctrls_scaled.shape[0]
n_ctrl_pcs = min(50, n_cells)

pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)        # shape (n_cells, n_ctrl_pcs)

# Build mapping from cell_id → compressed control vector
cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))

if not cell2vec:
    raise ValueError(
        "No common cell IDs found between lincs1000.csv and embeddings/ctrls.csv. "
        "Cannot proceed. Please check your data files."
    )

print(f"Loaded and processed control embeddings for {len(cell2vec)} unique cells.")

def build_context_matrix_improved(df_split, morgan_fps_scaled, pert_time, pert_dose, 
                                 ignore_time, ignore_dose, split_name, scaler_context=None, is_train=False):
    """
    Build context matrix with globally consistent scaling
    """
    cell_ids = df_split['cell_id'].to_numpy()
    unique_cells_split = np.sort(df_split['cell_id'].unique())
    
    all_continuous_context = []
    valid_cells = []
    
    for cell_id in unique_cells_split:
        if cell_id not in cell2vec:
            print(f"Warning: Cell {cell_id} not found in control embeddings, skipping...")
            continue
            
        mask = cell_ids == cell_id
        if mask.sum() == 0:
            continue
            
        valid_cells.append(cell_id)
        
        # Build continuous context matrix (cell embeddings + time + dose)
        C_continuous = np.hstack([
            np.tile(cell2vec[cell_id], (mask.sum(), 1)),  # Cell embeddings
            pert_time[mask],                              # Perturbation time
            pert_dose[mask],                              # Perturbation dose
        ])
        all_continuous_context.append(C_continuous)
    
    # Fit scaler on all continuous context (training data only)
    if is_train:
        all_continuous_combined = np.vstack(all_continuous_context)
        scaler_context = StandardScaler()
        scaler_context.fit(all_continuous_combined)
        print(f"Fitted context scaler on {all_continuous_combined.shape} continuous context features")
    
    if scaler_context is None:
        raise ValueError("scaler_context must be provided for non-training data")
    
    X_lst, C_lst, cell_lst = [], [], []
    
    for i, cell_id in enumerate(valid_cells):
        mask = cell_ids == cell_id
        X_cell = X_train_scaled[mask] if split_name == 'train' else X_test_scaled[mask]
        
        # Scale continuous context consistently
        C_continuous_scaled = scaler_context.transform(all_continuous_context[i])
        
        n_samples = mask.sum()
        
        # Combine all context features
        C_cell = np.hstack([
            C_continuous_scaled,                    # Scaled continuous features
            morgan_fps_scaled[mask],               # Pre-scaled molecular features  
            ignore_time[mask],                     # Binary flags (unscaled)
            ignore_dose[mask],
        ])

        X_lst.append(X_cell)
        C_lst.append(C_cell)
        cell_lst.append(cell_ids[mask])

    if not X_lst:
        raise RuntimeError(f"No data collected for {split_name} set.")
    
    X_final = np.vstack(X_lst)
    C_final = np.vstack(C_lst)
    cell_ids_final = np.concatenate(cell_lst)
    
    return X_final, C_final, cell_ids_final, scaler_context

# Build context matrices for both splits with improved scaling
print("Building context matrices with improved scaling...")

X_train, C_train, cell_ids_train, scaler_context = build_context_matrix_improved(
    df_train, morgan_train_scaled, pert_time_train, pert_dose_train,
    ignore_time_train, ignore_dose_train, 'train', is_train=True
)

X_test, C_test, cell_ids_test, _ = build_context_matrix_improved(
    df_test, morgan_test_scaled, pert_time_test, pert_dose_test,
    ignore_time_test, ignore_dose_test, 'test', scaler_context=scaler_context
)

print(f'Context matrix:   train {C_train.shape}   test {C_test.shape}')
print(f'Feature matrix:   train {X_train.shape}   test {X_test.shape}')

# IMPROVED PCA WITH BETTER SCALING
print("Applying PCA with improved scaling...")

# PCA on features (fit on training data only)
pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
X_train_pca = pca_data.fit_transform(X_train)
X_test_pca = pca_data.transform(X_test)

# Improved scaling in PCA space
pca_scaler = StandardScaler()
X_train_norm = pca_scaler.fit_transform(X_train_pca)
X_test_norm = pca_scaler.transform(X_test_pca)

print(f'Normalized PCA features: train {X_train_norm.shape}   test {X_test_norm.shape}')

ccn = ContextualizedCorrelationNetworks(
    encoder_type='mlp',
    num_archetypes=30,  
    n_bootstraps=1,     
)

ccn.fit(C_train, X_train_norm)

# Evaluate on both training and test sets
mse_train = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
mse_test = ccn.measure_mses(C_test, X_test_norm, individual_preds=False)

print('\n' + '='*80)
print('FINAL PERFORMANCE RESULTS (with improved scaling)')
print('='*80)
print('Performance (mean squared error)')
print(f'  Training set MSE                 : {mse_train.mean():.4f} ± {mse_train.std():.4f}')
print(f'  Test set MSE (unseen perturbations): {mse_test.mean():.4f} ± {mse_test.std():.4f}')
print(f'  Generalization gap               : {mse_test.mean() - mse_train.mean():.4f}')
print('='*80)

# Per-cell performance breakdown
print(f"\nPer-cell performance breakdown:")
print("Cell ID          Train MSE    Test MSE     Train N  Test N")
print("─" * 60)

all_unique_cells = np.union1d(cell_ids_train, cell_ids_test)

for cell_id in sorted(all_unique_cells):
    tr_mask = cell_ids_train == cell_id
    te_mask = cell_ids_test == cell_id
    
    tr_mse = mse_train[tr_mask].mean() if tr_mask.any() else np.nan
    te_mse = mse_test[te_mask].mean() if te_mask.any() else np.nan
    tr_n = tr_mask.sum()
    te_n = te_mask.sum()
    
    if tr_n > 0 or te_n > 0:
        print(f'{cell_id:<15}  {tr_mse:8.4f}   {te_mse:8.4f}   {tr_n:6d}   {te_n:6d}')

# Summary statistics about the perturbation split
print(f"\n" + "="*80)
print("PERTURBATION HOLDOUT SUMMARY:")
print(f"  Total unique SMILES: {len(unique_smiles)}")
print(f"  Training SMILES: {len(smiles_train)} ({len(smiles_train)/len(unique_smiles)*100:.1f}%)")
print(f"  Test SMILES: {len(smiles_test)} ({len(smiles_test)/len(unique_smiles)*100:.1f}%)")
print(f"  Training samples: {len(df_train)}")
print(f"  Test samples: {len(df_test)}")
print("="*80)

