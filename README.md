## Installation

```bash
git clone --recurse-submodules https://github.com/cnellington/contextpert.git
cd Contextualized
pip install -e .
cd ..
pip install -r requirements.txt
```

Verify installation by running:
```bash
python test_installation.py
```

which should produce a test loss on a dummy dataset.

## Data
Download all files from https://cmu.app.box.com/folder/290094222331 and place them in the `data/` directory.

For debugging, consider making a smaller version of the dataset with the first 1000 rows
```bash
head -n 1000 data/full_lincs.csv > data/full_lincs_head.csv
```

## Run Experiments

### Table 2 Experiments: Post-Perturbation Networks

Use the following scripts depending on the experimental setup:

#### `pert_context.py`

- Uses:
  - One-hot encoded **cell type context**
  - One-hot encoded **perturbation context**
  - Optional inclusion of **dose** and/or **time**
  - can fit on any pert type

#### `cell_ctxt.ipynb`

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
