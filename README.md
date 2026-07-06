# Microbial Growth TPC Predictor

A toolkit for predicting the full **Temperature-Performance Curve (TPC)** of microbial growth
from a genome FASTA file, using a physics-informed deep learning model anchored by OGT
prediction and optional FBA-based absolute scaling.

---

## How it works

```
Genome FASTA
    |
    |---[Prodigal: gene calling]---> protein sequences
    |       |
    |       |--> ESM-2 mean-pooled embedding (1280-dim)
    |       |          |
    |       |    +-----+-------------------------------+
    |       |    |   core_model.py (UDE)               |
    |       |    |   ESMTempEncoder + UTPC ODE physics  |
    |       |    |   + constrained residual correction  |
    |       |    +-----+-------------------------------+
    |       |          | normalised TPC shape (peak = 1)
    |       |          v
    |       +--> codon/AA/dipeptide features (526-dim)
    |                  |
    |            +-----+----------+
    |            |  OGT_predictor |  sklearn MLP (256-128-64)
    |            +-----+----------+
    |                  | OGT (C)  <-- hard-anchors Topt in UTPC
    |
    |---[CarveMe GEM reconstruction]
    |           |
    |    +-------+----------+
    |    | FBA_anchor_point |  COBRApy FBA with user medium
    |    +-------+----------+
    |            | peak growth rate (h-1)
    |
    absolute TPC(T) = normalised_shape(T) x peak_rate
```

### Stage 1 -- OGT prediction
`OGT_predictor.py` uses 526 genomic features (rRNA composition, amino-acid usage,
codon usage, dipeptide frequencies) to predict the optimal growth temperature (OGT)
with an MLP trained on 3131 genomes (2869 Bacteria + 262 Archaea).

CV performance (10-fold, n = 3131): **RMSE 5.12 C | MAE 3.91 C | R2 0.87**

### Stage 2 -- TPC shape (core model)
`core_model.py` trains a UDE (Universal Differential Equation) model: an
ESM-2-based transformer encoder generates UTPC physics parameters (Pmax, E),
which seed an Eppley-style ODE; a constrained residual MLP corrects the trajectory.
OGT is used directly as a hard anchor for Topt.

### Stage 3 -- Absolute scaling (optional)
`FBA_anchor_point.py` reconstructs a genome-scale metabolic model with CarveMe
and runs FBA under the user-specified medium to obtain the absolute peak growth rate.

---

## Repository layout

```
Microbial-Growth-TPC-Predictor/
|-- code/
|   |-- core_model.py          Training script for the UDE TPC shape model
|   |-- TPC_predictor.py       Main entry point: FASTA + medium -> absolute TPC
|   |-- OGT_predictor.py       OGT MLP: training and inference from FASTA/CSV
|   |-- FBA_anchor_point.py    CarveMe GEM reconstruction + COBRApy FBA
|-- data/                      (empty -- place training data here locally)
|-- results/
|   |-- core_model_checkpoint.pt   Trained UDE weights
|   |-- core_model_scaler.pkl      ESM embedding StandardScaler
|   |-- ogt_mlp/
|       |-- mlp.pkl                Trained OGT MLP
|       |-- scaler.pkl             Feature StandardScaler
|       |-- feature_cols.pkl       Ordered list of 526 feature names
|       |-- cv_results.json        10-fold CV metrics
|-- Train/
|   |-- core_model_training_log.csv    Per-epoch losses
|   |-- core_model_training_loss.png   Loss curve plot
|   |-- ogt_mlp_cv_log.csv             Per-fold CV results
|-- examples/
|   |-- example_ecoli.py               E. coli K-12 MG1655 (mesophile)
|   |-- example_thermus.py             T. thermophilus HB8 (thermophile)
|   |-- example_medium_ecoli.json      M9 minimal medium for E. coli
|   |-- example_medium_thermus.json    Defined medium for T. thermophilus
|-- requirements.txt
|-- README.md
|-- READMECN.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For FBA support (optional):

```bash
pip install cobra carveme
# CarveMe also requires DIAMOND; see https://carveme.readthedocs.io
```

### 2. Prepare data and train models

#### 2a. Train the OGT MLP

```bash
python code/OGT_predictor.py train \
    --bacteria_csv data/calculated_features_bacteria.csv \
    --archaea_csv  data/calculated_features_archaea.csv
```

This runs 10-fold CV, prints metrics, trains a final model on all data, and saves
artifacts to `results/ogt_mlp/` with CV logs in `Train/`.

#### 2b. Train the core TPC shape model

Place the TPC CSV (with pre-embedded ESM-2 columns) in `data/` then run:

```bash
python code/core_model.py --data data/11800TPC_1_1_with_medium_group_3_with_OGT.csv
```

Training saves `results/core_model_checkpoint.pt`, `results/core_model_scaler.pkl`,
and logs/plots to `Train/`.

### 3. Predict TPC from a genome FASTA

**Normalised output (no FBA):**

```bash
python code/TPC_predictor.py \
    --fasta  genome.fna \
    --temp_min 5 --temp_max 80
```

**With FBA absolute scaling:**

```bash
python code/TPC_predictor.py \
    --fasta  genome.fna \
    --medium examples/example_medium_ecoli.json \
    --temp_min 5 --temp_max 80
```

**Override OGT (skip OGT prediction):**

```bash
python code/TPC_predictor.py --fasta genome.fna --ogt 37.0
```

External tools required for full pipeline:
- **Prodigal** (gene calling) -- https://github.com/hyattpd/Prodigal
- **Barrnap** (rRNA detection) -- https://github.com/tseemann/barrnap
- **ESM-2** -- `pip install fair-esm` or `pip install transformers`
- **CarveMe** (GEM reconstruction, FBA only) -- `pip install carveme`

### 4. Python API

```python
import numpy as np
import sys
sys.path.insert(0, "code")

from TPC_predictor import load_model, predict_shape

model, scaler, meta, device = load_model()

# esm_embedding: mean-pooled ESM-2 vector for the target proteome, shape (1280,)
result = predict_shape(
    model, scaler, meta, device,
    esm_embedding = esm_embedding,
    ogt_c         = 37.0,
    temperatures  = np.arange(5, 75, 1),
)
print(f"ToptC = {result['ToptC']:.1f} C")
print(f"Peak at T = {result['temperatures'][result['pred_shape'].argmax()]:.0f} C")
```

---

## Run the examples

```bash
# E. coli K-12 (mesophile, ~37 C)
python examples/example_ecoli.py

# Thermus thermophilus HB8 (thermophile, ~65 C)
python examples/example_thermus.py
```

Outputs are saved to `examples/output/`.

---

## Input specifications

### Genome FASTA
Nucleotide genome FASTA (`.fna`) or amino-acid proteome FASTA (`.faa`).
The pipeline detects the type automatically (>80% ACGTN = nucleotide).

### Medium (FBA)
JSON dict mapping COBRA exchange-reaction IDs to maximum uptake rates
(mmol gDW-1 h-1, positive values).  See `examples/example_medium_ecoli.json`
for the E. coli M9 minimal medium.

### Feature CSV (OGT training)
One row per genome, one column per feature.  Must contain an `OGT` column.
Column names must match those in `results/ogt_mlp/feature_cols.pkl`.

---

## OGT model performance

| Split              | RMSE (C) | MAE (C) | R2   |
|--------------------|----------|---------|------|
| 10-fold CV overall | 5.12     | 3.91    | 0.87 |
| Best fold          | 4.58     | 3.50    | 0.92 |
| Worst fold         | 5.75     | 4.31    | 0.80 |

Training set: 2869 Bacteria + 262 Archaea (GTDB taxonomy).

---

## Citation

If you use this toolkit in your research, please cite:

- The **Hybrid-TPC-Model** for the UDE / UTPC architecture.
- The source of the genomic feature data and OGT labels
  (e.g., TEMPURA database, Engqvist 2018).
- COBRApy for FBA: Ebrahim et al. (2013) *BMC Systems Biology*.
- ESM-2: Lin et al. (2023) *Science*.

---

## License

MIT
