# Microbial Growth TPC Predictor (MGTP)

A modular toolkit that generates a full **Temperature-Performance Curve (TPC)**
for a microbial organism from three inputs:

| Input | Description |
|---|---|
| Genomic features | 526 pre-computed features (rRNA, amino-acid usage, codon usage) |
| Proteome ESM embedding | Mean-pooled ESM-2 protein language model vector |
| Medium composition | Exchange-reaction bounds for FBA (optional) |

---

## How it works

```
Genomic features (526-dim)
         |
         v
  +--------------+
  |   OGT MLP    |  sklearn MLP (256-128-64) trained on 3 131 genomes
  +--------------+
         | OGT (degrees C)
         v
  +-----------------------------------------------------+
  |  TPC Shape  -  PINN / UDE (Hybrid-TPC-Model)        |
  |    ESM Transformer -> UTPC params (Pmax, E)          |
  |    + constrained residual ODE correction             |
  +-----------------------------------------------------+
         | Normalised shape (peak = 1)
         v
  +--------------+
  |  FBA Anchor  |  COBRApy FBA at user-specified medium  (optional)
  +--------------+
         | peak growth rate (h-1)
         v
  Absolute TPC:  growth_rate(T) = shape(T) x FBA_peak
```

### Stage 1 - OGT prediction
An MLP replaces the original +/-5 C noise simulator.
CV performance (10-fold, n = 3 131): **RMSE 5.12 C | MAE 3.91 C | R2 0.87**

### Stage 2 - TPC shape
The pre-trained UDE model from **Hybrid-TPC-Model** predicts a normalised
TPC using the ESM-2 proteome embedding anchored at the predicted OGT.

### Stage 3 - FBA peak anchor
FBA is run once at the predicted OGT under the given medium.  The resulting
maximum growth rate scales the normalised shape into absolute units (h-1).
If no metabolic model is supplied, the output remains normalised.

---

## Repository layout

```
MGTP/
├── mgtp/                    # Python package
│   ├── __init__.py
│   ├── ogt_predictor.py     # OGT MLP wrapper
│   ├── tpc_shape.py         # PINN shape predictor (architecture + loader)
│   ├── fba_anchor.py        # COBRApy FBA wrapper
│   └── pipeline.py          # End-to-end orchestration
├── train/
│   └── train_ogt.py         # Train + evaluate OGT MLP; saves artifacts
├── examples/
│   ├── example_pipeline.py  # Runnable demo (scenarios A and B)
│   └── example_medium_ecoli.json
├── models/                  # (not committed) -- place trained artifacts here
│   ├── ogt_mlp/             # mlp.pkl  scaler.pkl  feature_cols.pkl
│   └── tpc_pinn/            # checkpoint.pt  esm_scaler.pkl
├── requirements.txt
├── README.md
└── READMECN.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the OGT model

```bash
python train/train_ogt.py \
    --bacteria_csv /path/to/calculated_features_bacteria.csv \
    --archaea_csv  /path/to/calculated_features_archaea.csv \
    --output_dir   models/ogt_mlp
```

This runs 10-fold CV, prints metrics, then trains a final model on all data
and saves artifacts to `models/ogt_mlp/`.

### 3. Place TPC PINN artifacts

Copy the trained PINN checkpoint and scaler from Hybrid-TPC-Model:

```bash
cp Hybrid-TPC-Model/results/group4_pinn_checkpoint.pt  models/tpc_pinn/checkpoint.pt
cp Hybrid-TPC-Model/results/group4_pinn_scaler.pkl     models/tpc_pinn/esm_scaler.pkl
```

### 4. Run the pipeline (Python API)

```python
import numpy as np
import pandas as pd
from mgtp import MGTPipeline

# Load pipeline (without FBA)
pipe = MGTPipeline(
    ogt_model_dir = "models/ogt_mlp",
    tpc_model_dir = "models/tpc_pinn",
)

# Predict TPC
T, rate, ogt = pipe.predict(
    esm_embedding     = esm_vec,          # ndarray, shape (emb_dim,)
    genomic_features  = feature_df,       # DataFrame with 526 features
    temperature_range = np.arange(5, 70, 1),
)
print(f"Predicted OGT: {ogt:.1f} C")
print(f"Peak normalised rate: {rate.max():.4f}")
```

### 5. With FBA anchor

```python
pipe = MGTPipeline(
    ogt_model_dir  = "models/ogt_mlp",
    tpc_model_dir  = "models/tpc_pinn",
    fba_model_path = "models/iJO1366.xml",
)

medium = {
    "EX_glc__D_e": 10.0,   # glucose
    "EX_o2_e":     20.0,   # oxygen
    "EX_nh4_e":    10.0,   # nitrogen
}

T, rate, ogt = pipe.predict(
    esm_embedding    = esm_vec,
    genomic_features = feature_df,
    medium           = medium,
    temperature_range = np.arange(5, 70, 1),
)
print(f"Peak growth rate (FBA): {rate.max():.4f} h-1")
```

---

## Input specifications

### Genomic features (526 columns)

Computed per genome using the same pipeline as the training data.
The exact column names are stored in `models/ogt_mlp/feature_cols.pkl`.

Feature groups:

| Group | Count |
|---|---|
| Genome size | 1 |
| rRNA nucleotide fractions + MFE/len (5S / 16S / 23S) | 15 |
| tRNA nucleotide fractions + MFE/len | 5 |
| GC content (normalised) | 1 |
| Proteome amino-acid fractions (raw + GC-normalised) | 40 |
| Proteome properties (mean length, charge ratios) | 4 |
| Codon usage (synonymous codon fractions) | 64 |
| Dipeptide frequencies | 400 |

### ESM embedding

Mean-pooled output of ESM-2 (e.g. `facebook/esm2_t33_650M_UR50D`)
over the organism's proteome. Dimension must match the value stored in
`models/tpc_pinn/checkpoint.pt` (key: `emb_len`).

### Medium (FBA)

A dict mapping exchange-reaction IDs to maximum uptake rates
(mmol gDW-1 h-1).  See `examples/example_medium_ecoli.json` for the
M9 minimal-medium example for *E. coli* iJO1366.

---

## OGT model performance

| Split | RMSE (C) | MAE (C) | R2 |
|---|---|---|---|
| 10-fold CV overall | 5.12 | 3.91 | 0.87 |
| Best fold | 4.58 | 3.50 | 0.92 |
| Worst fold | 5.75 | 4.31 | 0.80 |

Training set: 2 869 Bacteria + 262 Archaea (GTDB taxonomy).

---

## Citation

If you use MGTP in your research, please cite:

- The **Hybrid-TPC-Model** for the PINN/UDE architecture.
- The source of the genomic feature data and OGT labels
  (e.g. TEMPURA database, Engqvist 2018).
- COBRApy for FBA: Ebrahim et al. (2013) *BMC Systems Biology*.

---

## License

MIT
