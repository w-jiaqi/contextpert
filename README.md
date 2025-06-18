## Installation

Packages: pandas, numpy, rdkit, contextualized

## Instructions 

Before running any script, make sure to **set data paths and parameters at the top of each file**.

The full lincs dataset, `merged_output4.csv`, is in **BOX**.

---

### Table 2 Experiments: Post-Perturbation Networks

Use the following scripts depending on the experimental setup:

#### `pert_context.py`

- Uses:
  - One-hot encoded **cell type context**
  - One-hot encoded **perturbation context**
  - Optional inclusion of **dose** and/or **time**
  - can fit on any pert type

#### `cell_ctxt.py`

- Uses:
  - **Embedding-based** or **PCA-compressed** cell type context
- Requires:
  - `ctrls.csv`
  - Embedding `.npy` files 
  - can fit on any pert type

---

### Table 3 Experiments: Generalization to Unseen Perturbations

#### `unseen_pert.py`

- Requires:
  - `trt_cp_smiles.csv` file with only trt_cp perturbations with smiles
  - `ctrls.csv`
  - (Both in BOX)

---
