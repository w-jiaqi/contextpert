# Contextualized Network Training & Label Cohesion Analysis

The pipeline consists of two main components: model training and cohesion analysis.

## Files

1. **`ctxt_training.py`** - Main training script for contextualized neural networks
2. **`cohesion_analysis.py`** - Label cohesion evaluation script

## Usage

### 1. Model Training

The script trains the model and saves all necessary results to perform label cohesion analysis.

```bash
python ctxt_training.py
```

### 2. Label Cohesion Analysis

This script evaluates how well samples with the same disease label cluster together in different representations.

```bash
python cohesion_analysis.py \
    --aligned_data path/to/saved/results/simple_aligned_predictions.npz \
    --output_dir cohesion_results
```

**Evaluation Metrics:**
- **Silhouette Score**: Labels-as-clusters evaluation (overall + macro-averaged)
- **k-NN Label Agreement**: Fraction of k nearest neighbors sharing the same label
- **1-NN Balanced Accuracy**: Predict disease label from nearest neighbor
- **Pairwise AUC**: Probability that same-disease pairs are closer than different-disease pairs