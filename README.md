## Instructions

Before running any script, make sure to **set data paths and parameters at the top of each file**.

The core dataset, `merged4.csv`, is located in **BOX** folder.

---

### Table 2 Experiments: Post-Perturbation Networks

Use the following scripts depending on the experimental setup:

#### `pert_context.py`

- Uses:
  - One-hot encoded **cell type context**
  - One-hot encoded **perturbation context**
  - Optional inclusion of **dose** and/or **time**

#### `cell_ctxt.py`

- Uses:
  - **Embedding-based** or **PCA-compressed** cell type context
- Requires:
  - `ctrls.csv`
  - Embedding `.npy` files (both in BOX)

---

### Table 3 Experiments: Generalization to Unseen Perturbations

#### `unseen_pert.py`

- Requires:
  - `trt_cp_smiles.csv`
  - `ctrls.csv`
  - (Both in BOX)

---
