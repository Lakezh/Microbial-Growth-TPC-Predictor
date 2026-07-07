# Microbial Growth TPC Predictor

A toolkit for predicting the full **Temperature-Performance Curve (TPC)** of microbial
growth, using a physics-informed deep learning model (UDE/UTPC) anchored by OGT prediction
and optionally scaled to absolute growth rates via FBA.

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
    |       |    |   core_model  (UDE)                 |
    |       |    |   ESMTempEncoder + UTPC ODE physics  |
    |       |    |   + constrained residual correction  |
    |       |    +-----+-------------------------------+
    |       |          | normalised TPC shape (peak = 1)
    |       |          v
    |       +--> codon / AA / dipeptide features (526-dim)
    |                  |
    |            +-----+----------+
    |            |  OGT_predictor |  sklearn MLP (256-128-64)
    |            +-----+----------+
    |                  | OGT (C)  <-- hard-anchors Topt in UTPC
    |
    |---[CarveMe GEM reconstruction]
    |           |
    |    +-------+----------+
    |    | FBA_anchor_point |  COBRApy FBA with user-defined medium
    |    +-------+----------+
    |            | peak growth rate (h-1)
    |
    Absolute TPC(T) = normalised_shape(T) x peak_rate
```

---

## Models and methods

### Stage 1 — OGT prediction (`OGT_predictor.py`)

The optimal growth temperature (OGT) is predicted by a **multilayer perceptron (MLP)**
trained on 526 genomic features extracted from 3131 genomes (2869 Bacteria + 262 Archaea,
GTDB taxonomy).

**Feature groups (526 total):**

| Feature group | Dimensions |
|---|---|
| Genome size | 1 |
| rRNA nucleotide fractions + MFE/len (5S / 16S / 23S) | 15 |
| tRNA nucleotide fractions + MFE/len | 5 |
| Genomic GC content | 1 |
| Proteome amino-acid fractions (raw + GC-normalised) | 40 |
| Proteome properties (mean length, charge ratios) | 4 |
| Codon usage (synonymous codon fractions) | 64 |
| Dipeptide frequencies | 400 |

**Architecture:** hidden layers (256, 128, 64), ReLU, Adam, early stopping.
**CV performance (10-fold, n = 3131):** RMSE 5.12 C | MAE 3.91 C | R2 0.87

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
Trains and applies the OGT MLP from 526 genomic features.

Key contents:
- `train_ogt_mlp(bacteria_csv, archaea_csv)` — 10-fold CV then final fit; saves
  `results/ogt_mlp/mlp.pkl`, `scaler.pkl`, `feature_cols.pkl`, `cv_results.json`.
- `extract_genomic_features(genome_fasta, tmp_dir)` — calls Prodigal and Barrnap to
  extract protein sequences and rRNA sequences, then computes all 526 features.
- `predict_ogt_from_fasta(fasta_path, model_dir)` — end-to-end: FASTA → features → OGT.
- `predict_ogt_from_csv(feature_csv, model_dir)` — batch OGT prediction from a
  pre-computed feature CSV.

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
- `ogt_mlp/scaler.pkl` — `StandardScaler` fitted on the 526 features.
- `ogt_mlp/feature_cols.pkl` — ordered list of the 526 feature column names.
- `ogt_mlp/cv_results.json` — 10-fold CV metrics per fold.

### `Train/`
Training records generated during model fitting (not committed by default):
- `core_model_training_log.csv` — per-epoch loss breakdown (data / reg / mono / tail).
- `core_model_training_loss.png` — training loss curve.
- `ogt_mlp_cv_log.csv` — per-fold RMSE / MAE / R2.

---

## Step-by-step usage example: *E. coli* K-12 MG1655

This walkthrough predicts the full TPC of *E. coli* K-12 MG1655 and, optionally, scales
it to absolute growth rates using FBA.

### Step 0 — Install dependencies

```bash
pip install -r requirements.txt

# For ESM-2 embedding extraction:
pip install fair-esm          # Facebook's official library (recommended)
# or:  pip install transformers

# For gene calling (requires a Linux/Mac environment or WSL):
# Prodigal: https://github.com/hyattpd/Prodigal
# Barrnap:  https://github.com/tseemann/barrnap

# For FBA (optional):
pip install cobra carveme
```

### Step 1 — Download the genome

Download the *E. coli* K-12 MG1655 genome from NCBI RefSeq:

```
Accession: GCF_000005845.2
File:      GCF_000005845.2_ASM584v2_genomic.fna
```

```bash
# Using NCBI datasets CLI (https://www.ncbi.nlm.nih.gov/datasets/):
datasets download genome accession GCF_000005845.2 --include genome
unzip ncbi_dataset.zip
# genome FASTA is at: ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna
```

### Step 2 — Predict OGT from genomic features

Extract 526 genomic features and predict OGT using the trained MLP:

```bash
python code/OGT_predictor.py predict \
    --fasta ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna \
    --model_dir results/ogt_mlp
```

Expected output:
```
Predicted OGT: 36.8 C
```

> If Prodigal or Barrnap are not available, you can supply the known OGT directly
> in Step 4 using `--ogt 37.0`.

### Step 3 — Compute ESM-2 proteome embedding

First, extract protein sequences with Prodigal:

```bash
prodigal \
    -i ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna \
    -a ecoli_proteins.faa \
    -p single -q
```

Then embed all proteins with ESM-2 and mean-pool:

```python
import esm, torch, numpy as np

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

# Read protein sequences
seqs = []
with open("ecoli_proteins.faa") as f:
    h, s = "", ""
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if h and s: seqs.append((h, s))
            h = line[1:].split()[0]; s = ""
        else:
            s += line
    if h and s: seqs.append((h, s))

# Embed in batches, mean-pool
embeddings = []
batch_size = 8
for i in range(0, len(seqs), batch_size):
    batch = [(h, s[:1022]) for h, s in seqs[i:i+batch_size]]
    _, _, tokens = batch_converter(batch)
    with torch.no_grad():
        out = model_esm(tokens, repr_layers=[33])
    for j, (_, s) in enumerate(batch):
        embeddings.append(out["representations"][33][j, 1:len(s)+1].mean(0).numpy())

esm_embedding = np.mean(embeddings, axis=0).astype(np.float32)  # shape (1280,)
np.save("ecoli_esm_embedding.npy", esm_embedding)
print(f"ESM embedding shape: {esm_embedding.shape}")
```

### Step 4 — Predict normalised TPC shape

```python
import numpy as np, sys
sys.path.insert(0, "code")
from TPC_predictor import load_model, predict_shape, plot_prediction

# Load model
model, scaler, meta, device = load_model()

# Load embedding and predicted OGT
esm_embedding = np.load("ecoli_esm_embedding.npy")
ogt_c         = 36.8   # from Step 2 (or use 37.0 as the known value)
temperatures  = np.arange(5, 75, 1, dtype=np.float32)

# Predict
result = predict_shape(model, scaler, meta, device,
                       esm_embedding=esm_embedding,
                       ogt_c=ogt_c,
                       temperatures=temperatures)

print(f"Topt:  {result['ToptC']:.1f} C")
print(f"Pmax:  {result['Pmax']:.4f}")
print(f"E:     {result['E']:.4f}")
print(f"Peak at: {temperatures[result['pred_shape'].argmax()]:.0f} C")

# Save CSV and plot
import pandas as pd
pd.DataFrame({"temperature_C": result["temperatures"],
              "norm_shape":    result["pred_shape"]}).to_csv("ecoli_tpc.csv", index=False)
plot_prediction(result, title="E. coli K-12 MG1655", save_path="ecoli_tpc.png")
```

Expected `ecoli_tpc.csv` (first few rows):

```
temperature_C,norm_shape
5.0,0.012
10.0,0.041
15.0,0.112
...
37.0,1.000
...
55.0,0.023
```

### Step 5 — (Optional) Scale to absolute growth rates with FBA

Provide the *E. coli* iJO1366 model and the M9 medium:

```python
import sys
sys.path.insert(0, "code")
from FBA_anchor_point import get_peak_growth_rate
import numpy as np, json

with open("examples/example_medium_ecoli.json") as f:
    medium = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

# Reconstruct GEM and run FBA
peak_rate = get_peak_growth_rate(
    fasta_path    = "ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
    medium        = medium,
    temperature_c = 37.0,
)
print(f"FBA peak growth rate: {peak_rate:.4f} h-1")

# Scale the normalised TPC
norm_shape = np.load("ecoli_tpc_shape.npy")  # or use result["pred_shape"] from Step 4
absolute_tpc = norm_shape * peak_rate
```

Expected output:
```
FBA peak growth rate: 0.9821 h-1
```

> **Note:** CarveMe requires a valid DIAMOND database and a compatible solver (CPLEX or
> GLPK). See https://carveme.readthedocs.io for installation.

### Step 6 — Run the bundled example

The above steps are pre-packaged in `examples/example_ecoli.py` (with a placeholder
embedding). Once you have a real ESM embedding, replace the `esm_embedding` variable and run:

```bash
python examples/example_ecoli.py
# outputs: examples/output/ecoli_tpc.csv
#          examples/output/ecoli_tpc.png
```

---

## Training your own models

### Retrain the OGT MLP

Prepare two CSV files — one for Bacteria, one for Archaea — each with 526 feature columns
and an `OGT` column (degrees C):

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
- Genomic feature data and OGT labels: TEMPURA database; Engqvist (2018) *PeerJ*.
- ESM-2: Lin et al. (2023) *Science* 379, 1123–1130.
- COBRApy: Ebrahim et al. (2013) *BMC Systems Biology* 7, 74.

---

## License

MIT
