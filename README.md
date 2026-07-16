# Microbial Growth TPC Predictor

A toolkit for predicting the full **Temperature-Performance Curve (TPC)** of microbial
growth, using a physics-informed deep learning model (UDE/UTPC) anchored by OGT prediction
and optionally scaled to absolute growth rates via FBA.

---

## How it works

**Three inputs → one output:**

| Input | Description |
|---|---|
| Proteome FASTA | Amino acid sequences (genome FASTA also accepted — auto-converted via Prodigal) |
| Temperature range | e.g. 5–80 °C, step 1 °C |
| Medium JSON | Exchange reaction IDs + uptake rates for FBA absolute scaling |

```
Proteome FASTA  (genome → Prodigal → protein, if nucleotide input)
    |
    +---> 425 protein features --> OGT MLP --> OGT (C)  [hard Topt anchor]
    |     (AA fracs + dipeptides + proteome props)
    |
    +---> ESM-2 mean-pooled embedding (1280-dim)
    |              |
    |       core_model (UDE/UTPC)
    |       ESMTempEncoder + UTPC ODE + constrained residual
    |              |
    |       normalised TPC shape  (peak = 1)  over user-defined temperature range
    |              |
    |  Medium JSON --> CarveMe GEM + COBRApy FBA --> peak growth rate (h-1)
    |              |
    Absolute TPC(T) = normalised_shape(T) × peak_rate   [h-1 at each temperature]
```

---

## Models and methods

### Stage 1 — OGT prediction (`OGT_predictor.py`)

The optimal growth temperature (OGT) is predicted by a **multilayer perceptron (MLP)**
trained on **425 protein-sequence-derived features** from 3131 genomes
(2869 Bacteria + 262 Archaea, GTDB taxonomy).
All features are derived directly from the amino acid sequences, so only a proteome
FASTA is required — no genome sequence, rRNA, tRNA, or codon data.

**Feature groups (425 total):**

| Feature group | Dimensions |
|---|---|
| Raw amino acid fractions (20 AAs) | 20 |
| Proteome properties (mean length, charged/hydrophobic fractions, IVYWREL ratios) | 5 |
| Dipeptide frequencies (20 × 20) | 400 |

**Architecture:** hidden layers (256, 128, 64), ReLU, Adam, adaptive LR, early stopping.  
**10-fold CV (n = 3131):** RMSE 5.17 °C | MAE 3.95 °C | R² 0.86

### Stage 2 — TPC shape prediction (`core_model.py` + `TPC_predictor.py`)

The normalised TPC shape is predicted by a **Universal Differential Equation (UDE)** model
combining two components:

**Encoder (`ESMTempEncoder_MLP`):**  
The proteome is represented as a mean-pooled ESM-2 embedding (1280-dim, model:
`esm2_t33_650M_UR50D`). A patch-based transformer encoder maps this embedding to a latent
vector z (64-dim), from which a parameter head outputs the UTPC physics parameters Pmax and E.

**Physics prior (UTPC ODE):**  
The predicted parameters seed a **Universal Temperature Performance Curve (UTPC)** based on
Eppley-style kinetics:

```
dμ/dT = -(Pmax / E) · exp((T - Topt) / E) · (T - Topt) / E
```

OGT is used directly as a **hard anchor** for Topt (no noise simulation).

**Residual correction (`ResidualMLP`):**  
A small residual MLP corrects the ODE trajectory, constrained so it cannot increase the
rate above OGT (enforced via `softplus` gating).

**Training schedule:** Warmup (25 ep) → Alternating θ / residual (4 cycles × 8 ep each)
→ Joint fine-tuning (20 ep). Losses: SmoothL1 data + monotonicity penalty + tail penalty.

### Stage 3 — Absolute scaling (`FBA_anchor_point.py`)

**CarveMe** reconstructs a genome-scale metabolic model (GEM) from the proteome FASTA.
**COBRApy FBA** is then solved under the user-specified medium; the optimal growth rate
(h⁻¹) anchors the normalised TPC shape to absolute units:

```
absolute_TPC(T) = normalised_shape(T) × FBA_peak_rate
```

If no metabolic model is provided the output remains normalised (peak = 1).

---

## File descriptions

### `code/core_model.py`
Training script for the UDE TPC shape model.

Key contents:
- `load_data(data_csv)` — loads the TPC dataset (CSV with embedded ESM-2 columns), filters
  to Bacteria/Archaea, fills missing OGT values, standardises ESM embeddings.
- `build_curves(...)` — groups the dataset by TPC_id and builds per-curve dicts
  (temperature array, normalised shape, embedding, OGT).
- `train_all(df, esm_cols)` — full training loop: Warmup → Alternating → Joint.
  Logs per-epoch losses to `Train/core_model_training_log.csv` and saves a loss plot.
- `save_model(...)` — saves `results/core_model_checkpoint.pt` and
  `results/core_model_scaler.pkl`.
- Neural network classes: `PositionalEncoding`, `ESMTempEncoder_MLP`, `ParamHead`,
  `ResidualMLP`, `UTPC_ODEFunc_Constrained`, `UDEModel_Constrained`.

### `code/TPC_predictor.py`
Inference script — predicts normalised TPC shapes from ESM-2 embeddings and OGT values.

Key contents:
- `load_model(checkpoint_path, scaler_path)` — loads the trained UDE weights and ESM scaler;
  returns `model, scaler, meta, device`.
- `predict_single(model, scaler, meta, device, esm_embedding, ogt_c, temperatures)` —
  predicts the normalised TPC for one organism. Returns `pred_shape`, `Pmax`, `ToptC`, `E`.
- `predict_from_csv(...)` — batch prediction from a CSV that already contains ESM columns
  and an OGT column.
- `plot_prediction(result, title, save_path)` — plots the predicted TPC curve.
- Neural network classes (identical to `core_model.py`, required for weight loading).

### `code/OGT_predictor.py`
Trains and applies the OGT MLP from 425 protein-sequence-derived features.

Key contents:
- `train_ogt_mlp(bacteria_csv, archaea_csv)` — 10-fold CV then final fit; automatically
  selects the 425 protein-compatible columns from the input CSVs; saves
  `results/ogt_mlp/mlp.pkl`, `scaler.pkl`, `feature_cols.pkl`, `cv_results.json`.
- `extract_protein_features(protein_fasta)` — reads a proteome FASTA and computes
  all 425 features (AA fractions, proteome properties, dipeptide frequencies).
- `predict_ogt_from_fasta(fasta_path, model_dir)` — end-to-end: proteome FASTA → features → OGT.
- `predict_ogt_from_csv(feature_csv, model_dir)` — batch OGT prediction from a
  pre-computed 425-feature CSV.

### `code/FBA_anchor_point.py`
Genome-scale metabolic model reconstruction and FBA.

Key contents:
- `reconstruct_gem(proteome_fasta, output_xml, universe)` — runs CarveMe to build a SBML GEM.
- `run_fba(gem_path, medium)` — loads the GEM with COBRApy, applies exchange-reaction bounds
  from the medium dict, solves FBA, returns the optimal growth rate (h⁻¹).
- `get_peak_growth_rate(fasta_path, medium, temperature_c, gem_path)` — high-level API:
  handles nucleotide vs. amino-acid FASTA detection, optional Prodigal call, and
  CarveMe + FBA in one call.

### `examples/example_ecoli.py`
Runnable demo for *E. coli* K-12 MG1655 (mesophile, OGT 37 C). Uses a placeholder random
ESM embedding; replace with the real embedding to get a meaningful prediction.

### `examples/example_thermus.py`
Runnable demo for *Thermus thermophilus* HB8 (thermophile, OGT 65 C).

### `examples/example_medium_ecoli.json`
M9 minimal medium for *E. coli* iJO1366: glucose, O₂, NH₄⁺, phosphate, sulfate, and
trace minerals. Exchange-reaction IDs follow the BiGG namespace.

### `examples/example_medium_thermus.json`
Defined medium for *T. thermophilus*: same carbon/nitrogen/mineral composition as the
*E. coli* medium.

### `results/`
Pre-trained model artifacts committed to the repository:
- `core_model_checkpoint.pt` — UDE encoder/head/residual weights + metadata (emb_len,
  n_patches, t_mean_k, t_std_k, esm_cols, hyperparams).
- `core_model_scaler.pkl` — `sklearn.StandardScaler` fitted on the ESM embeddings.
- `ogt_mlp/mlp.pkl` — final OGT MLP fitted on all 3131 genomes.
- `ogt_mlp/scaler.pkl` — `StandardScaler` fitted on the 425 features.
- `ogt_mlp/feature_cols.pkl` — ordered list of the 425 protein-level feature column names.
- `ogt_mlp/cv_results.json` — 10-fold CV metrics per fold.

### `Train/`
Training records generated during model fitting (not committed by default):
- `core_model_training_log.csv` — per-epoch loss breakdown (data / reg / mono / tail).
- `core_model_training_loss.png` — training loss curve.
- `ogt_mlp_cv_log.csv` — per-fold RMSE / MAE / R2.

---

## Usage: *E. coli* K-12 MG1655 example

Three inputs are required to produce a complete absolute-scale TPC:

| # | Input | What to prepare |
|---|---|---|
| 1 | **Proteome FASTA** | Amino acid sequences (or genome FASTA — auto-converted via Prodigal) |
| 2 | **Temperature range** | Min / max / step in °C |
| 3 | **Medium JSON** | Exchange reaction IDs + uptake rates |

### Step 0 — Install dependencies

```bash
pip install -r requirements.txt

# ESM-2 (for TPC shape prediction):
pip install fair-esm          # Facebook's official library (recommended)
# or:  pip install transformers

# FBA (for absolute growth rate scaling):
pip install cobra carveme
```

### Step 1 — Prepare inputs

**1a. Download the proteome** from NCBI RefSeq (accession `GCF_000005845.2`):

```bash
# Using NCBI datasets CLI (https://www.ncbi.nlm.nih.gov/datasets/):
datasets download genome accession GCF_000005845.2 --include protein
unzip ncbi_dataset.zip
# proteome FASTA: ncbi_dataset/data/GCF_000005845.2/protein.faa
```

> If you only have a genome FASTA, pass it directly — the pipeline calls Prodigal
> automatically to extract protein sequences.

**1b. Define the temperature range** — e.g. 5 to 80 °C in 1 °C steps (set via CLI flags
or directly in Python as `np.arange(5, 81, 1)`).

**1c. Prepare the medium JSON** — a file mapping BiGG exchange-reaction IDs to maximum
uptake rates (mmol gDW⁻¹ h⁻¹). An example for *E. coli* M9 minimal medium is provided at
`examples/example_medium_ecoli.json`:

```json
{
  "EX_glc__D_e": 10.0,
  "EX_o2_e":     20.0,
  "EX_nh4_e":    10.0,
  "EX_pi_e":     10.0,
  "EX_so4_e":    10.0
}
```

### Step 2 — Run the full pipeline (one command)

```bash
python code/TPC_predictor.py \
    --fasta   ncbi_dataset/data/GCF_000005845.2/protein.faa \
    --medium  examples/example_medium_ecoli.json \
    --temp_min 5 --temp_max 80 --temp_step 1 \
    --output  ecoli_tpc.csv
```

The script:
1. Extracts 425 protein features → predicts OGT with the MLP
2. Computes the ESM-2 proteome embedding → predicts the normalised TPC shape (UDE/UTPC)
3. Reconstructs a GEM with CarveMe and runs FBA → gets the peak growth rate
4. Multiplies shape × peak rate → writes the absolute TPC to `ecoli_tpc.csv`

Expected output:
```
[OGT] Predicted OGT = 36.8 C
[ESM] Embedding 4321 proteins with ESM-2 ...
[FBA] Growth rate = 0.9821 h-1

Results saved to: ecoli_tpc.csv
OGT used:   36.8 C
UTPC Pmax:  3.2415
UTPC E:     8.7632
Plot saved to: ecoli_tpc.png
```

`ecoli_tpc.csv`:

```
temperature_C,norm_shape,abs_growth_rate_per_h
5.0,0.012,0.0118
...
37.0,1.000,0.9821
...
80.0,0.001,0.0010
```

> **Skip FBA:** omit `--medium` to get a normalised TPC only (no CarveMe required).  
> **Known OGT:** pass `--ogt 37.0` to skip OGT prediction.

### Step 3 — Python API (advanced)

For programmatic use or batch processing:

```python
import numpy as np, json, sys
sys.path.insert(0, "code")
from TPC_predictor import run_pipeline

with open("examples/example_medium_ecoli.json") as f:
    medium = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

result = run_pipeline(
    fasta_path   = "ncbi_dataset/data/GCF_000005845.2/protein.faa",
    temperatures = np.arange(5, 81, 1, dtype=np.float32),
    medium       = medium,          # omit for normalised TPC only
    # ogt_c      = 37.0,            # uncomment to override OGT prediction
)

print(f"OGT: {result['ogt_c']:.1f} C")
print(f"Peak growth rate: {result['abs_growth_rate'].max():.4f} h-1")

import pandas as pd
pd.DataFrame({
    "temperature_C":        result["temperatures"],
    "norm_shape":           result["norm_shape"],
    "abs_growth_rate_per_h": result["abs_growth_rate"],
}).to_csv("ecoli_tpc.csv", index=False)
```

> **Note:** CarveMe requires a valid DIAMOND database and a compatible solver (CPLEX or
> GLPK). See https://carveme.readthedocs.io for installation details.

---

## Training your own models

### Retrain the OGT MLP

Prepare two CSVs — one for Bacteria, one for Archaea — each containing genomic feature
columns and an `OGT` column (degrees C). The script automatically selects only the
425 protein-compatible columns (genome-level and codon-usage columns are dropped):

```bash
python code/OGT_predictor.py train \
    --bacteria_csv data/calculated_features_bacteria.csv \
    --archaea_csv  data/calculated_features_archaea.csv
```

Artifacts saved to `results/ogt_mlp/`. Training log saved to `Train/ogt_mlp_cv_log.csv`.

### Retrain the core TPC shape model

Prepare a TPC dataset CSV containing columns `TPC_id`, `binomial_name`, `temperature`,
`mu`, `OGT`, `kingdom`, and pre-embedded ESM-2 columns `esm2_0` … `esm2_1279`:

```bash
python code/core_model.py --data data/your_tpc_dataset.csv
```

Artifacts saved to `results/`. Training log and loss plot saved to `Train/`.

---

## OGT model performance

| Split              | RMSE (°C) | MAE (°C) | R²   |
|--------------------|-----------|----------|------|
| 10-fold CV overall | 5.17      | 3.95     | 0.86 |
| Best fold          | 4.82      | 3.54     | 0.89 |
| Worst fold         | 5.76      | 4.34     | 0.81 |

Training set: 2869 Bacteria + 262 Archaea (GTDB taxonomy), 425 protein-sequence features.

---

## Citation

If you use this toolkit in your research, please cite:

- The **Hybrid-TPC-Model** for the UDE / UTPC architecture.
- Genomic feature data and OGT labels: TEMPURA database; Engqvist (2018) *PeerJ*.
- ESM-2: Lin et al. (2023) *Science* 379, 1123–1130.
- COBRApy: Ebrahim et al. (2013) *BMC Systems Biology* 7, 74.

---

## License

Academic Non-Commercial License — free for non-commercial academic and research use, with attribution required. Commercial use requires prior written permission. See [LICENSE](LICENSE) for full terms.
