#!/usr/bin/env python3
"""
OGT_predictor.py  --  Train and apply the OGT MLP predictor.

Uses mean-pooled ESM-2 proteome embeddings (1280-dim) as the sole input.
The MLP maps the embedding to a scalar optimal growth temperature (OGT) in degrees C.

Modes
-----
    Training (from a CSV containing ESM embedding columns and OGT labels):
        python code/OGT_predictor.py train --data data/tpc_dataset.csv

    Prediction from a FASTA file (requires Prodigal on PATH):
        python code/OGT_predictor.py predict --fasta genome.fna

    Prediction from a pre-computed embedding (.npy file, shape 1280):
        python code/OGT_predictor.py predict --embedding emb.npy

Training CSV format
-------------------
    Must contain columns  esm2_0 ... esm2_1279  and one of:
    OGT, ogt, optimal_growth_temperature, OGT_C  (values in degrees C).
    Additional columns (species, kingdom, etc.) are ignored.
"""

import json, argparse, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR  = Path(__file__).parent
REPO_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = REPO_DIR / "results" / "ogt_mlp"
TRAIN_DIR   = REPO_DIR / "Train"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

MLP_PATH    = RESULTS_DIR / "mlp.pkl"
SCALER_PATH = RESULTS_DIR / "scaler.pkl"
CV_PATH     = RESULTS_DIR / "cv_results.json"

ESM_DIM  = 1280
ESM_COLS = [f"esm2_{i}" for i in range(ESM_DIM)]

# ============================================================
# MLP hyperparameters
# ============================================================
MLP_PARAMS = dict(
    hidden_layer_sizes  = (256, 128, 64),
    activation          = "relu",
    solver              = "adam",
    alpha               = 1e-4,
    learning_rate_init  = 1e-3,
    max_iter            = 500,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,
    batch_size          = 64,
    random_state        = 42,
)

# ============================================================
# Training
# ============================================================

def _load_training_data(data_csv: Path):
    """Load a CSV with ESM columns and an OGT column. Returns X (n, 1280), y (n,)."""
    df = pd.read_csv(data_csv)

    ogt_col = next(
        (c for c in ["OGT", "ogt", "optimal_growth_temperature", "OGT_C"]
         if c in df.columns),
        None
    )
    if ogt_col is None:
        raise ValueError(
            f"No OGT column found in {data_csv}. "
            "Expected one of: OGT, ogt, optimal_growth_temperature, OGT_C")

    missing_esm = [c for c in ESM_COLS if c not in df.columns]
    if missing_esm:
        raise ValueError(
            f"{len(missing_esm)} ESM columns missing (e.g. {missing_esm[0]}). "
            "The CSV must contain columns esm2_0 ... esm2_1279.")

    df[ogt_col] = pd.to_numeric(df[ogt_col], errors="coerce")
    df = df.dropna(subset=[ogt_col] + ESM_COLS).reset_index(drop=True)

    # One row per unique organism (drop duplicate embeddings if dataset has multiple
    # temperature points per species)
    if "binomial_name" in df.columns:
        df = df.drop_duplicates(subset="binomial_name").reset_index(drop=True)
    elif "species" in df.columns:
        df = df.drop_duplicates(subset="species").reset_index(drop=True)

    X = df[ESM_COLS].values.astype(np.float32)
    y = df[ogt_col].values.astype(np.float32)

    print(f"Training data: {len(df)} organisms | ESM dim = {ESM_DIM}")
    return X, y


def train_ogt_mlp(data_csv: Path):
    """10-fold CV then final model on all data.

    Saves results/ogt_mlp/mlp.pkl, scaler.pkl, cv_results.json
    and Train/ogt_mlp_cv_log.csv.
    """
    X, y = _load_training_data(data_csv)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    print("\n10-fold cross-validation:")
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        scaler = StandardScaler().fit(X[tr])
        mlp    = MLPRegressor(**MLP_PARAMS).fit(scaler.transform(X[tr]), y[tr])
        y_pred = mlp.predict(scaler.transform(X[va]))

        rmse = float(np.sqrt(mean_squared_error(y[va], y_pred)))
        mae  = float(mean_absolute_error(y[va], y_pred))
        r2   = float(r2_score(y[va], y_pred))
        fold_results.append({"fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2})
        print(f"  Fold {fold:2d}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")

    rmse_arr = [d["RMSE"] for d in fold_results]
    mae_arr  = [d["MAE"]  for d in fold_results]
    r2_arr   = [d["R2"]   for d in fold_results]
    print(f"\nOverall: RMSE={np.mean(rmse_arr):.2f}+/-{np.std(rmse_arr):.2f}  "
          f"MAE={np.mean(mae_arr):.2f}  R2={np.mean(r2_arr):.4f}")

    cv_summary = {
        "n_samples":  int(len(y)),
        "n_features": ESM_DIM,
        "input":      "ESM-2 mean-pooled proteome embedding (1280-dim)",
        "rmse_mean":  float(np.mean(rmse_arr)),
        "rmse_std":   float(np.std(rmse_arr)),
        "mae_mean":   float(np.mean(mae_arr)),
        "r2_mean":    float(np.mean(r2_arr)),
        "folds":      fold_results,
        "mlp_params": MLP_PARAMS,
    }
    with open(CV_PATH, "w") as fh:
        json.dump(cv_summary, fh, indent=2)
    pd.DataFrame(fold_results).to_csv(TRAIN_DIR / "ogt_mlp_cv_log.csv", index=False)
    print(f"CV results saved: {CV_PATH}")

    print("\nFitting final model on all data ...")
    final_scaler = StandardScaler().fit(X)
    final_mlp    = MLPRegressor(**MLP_PARAMS).fit(final_scaler.transform(X), y)

    with open(MLP_PATH,    "wb") as fh: pickle.dump(final_mlp,    fh)
    with open(SCALER_PATH, "wb") as fh: pickle.dump(final_scaler, fh)
    print(f"Saved: {MLP_PATH}")
    print(f"Saved: {SCALER_PATH}")


# ============================================================
# Load saved OGT model
# ============================================================

def load_ogt_model(model_dir=None):
    """Load trained OGT MLP and its StandardScaler. Returns (mlp, scaler)."""
    d = Path(model_dir) if model_dir else RESULTS_DIR
    with open(d / "mlp.pkl",    "rb") as fh: mlp    = pickle.load(fh)
    with open(d / "scaler.pkl", "rb") as fh: scaler = pickle.load(fh)
    return mlp, scaler


# ============================================================
# Prediction from a pre-computed embedding
# ============================================================

def predict_ogt_from_embedding(esm_embedding, model_dir=None) -> float:
    """Predict OGT from a pre-computed 1280-dim ESM-2 proteome embedding.

    Parameters
    ----------
    esm_embedding : array-like of shape (1280,)
    model_dir     : path to results/ogt_mlp/ (uses default if None)

    Returns
    -------
    float -- predicted OGT in degrees C
    """
    mlp, scaler = load_ogt_model(model_dir)
    x = np.asarray(esm_embedding, dtype=np.float32).reshape(1, -1)
    return float(mlp.predict(scaler.transform(x))[0])


# ============================================================
# ESM-2 extraction helpers (self-contained, no circular import)
# ============================================================

def _read_fasta(path):
    seqs = []
    h = s = ""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if h and s: seqs.append((h, s))
                h = line[1:].split()[0]; s = ""
            else:
                s += line
    if h and s: seqs.append((h, s))
    return seqs


def _is_nucleotide(path):
    text = Path(path).read_text(errors="ignore").upper()
    body = "".join(l for l in text.splitlines() if not l.startswith(">"))
    if not body: return True
    return sum(body.count(c) for c in "ACGTN") / len(body) > 0.80


def _call_prodigal(genome_fasta, out_dir):
    import subprocess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protein_out = out_dir / "proteins.faa"
    r = subprocess.run(
        ["prodigal", "-i", str(genome_fasta), "-a", str(protein_out), "-p", "meta", "-q"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Prodigal failed:\n{r.stderr}")
    return protein_out


def _extract_esm_embedding(fasta_path, tmp_dir=None, max_proteins=2000):
    """Mean-pool ESM-2 over all proteins; return numpy array of shape (1280,).

    Accepts genome (nucleotide) or proteome (amino acid) FASTA.
    Requires fair-esm or transformers. Prodigal required for nucleotide input.
    """
    import torch
    fasta_path = Path(fasta_path)
    if tmp_dir is None:
        tmp_dir = fasta_path.parent / "_tmp_ogt"

    if _is_nucleotide(fasta_path):
        print("[OGT/ESM] Nucleotide FASTA -- running Prodigal ...")
        protein_fasta = _call_prodigal(fasta_path, tmp_dir)
    else:
        protein_fasta = fasta_path

    seqs = _read_fasta(protein_fasta)
    if not seqs:
        raise ValueError(f"No sequences found in {protein_fasta}")
    if len(seqs) > max_proteins:
        import random as _rnd
        seqs = _rnd.sample(seqs, max_proteins)
        print(f"[OGT/ESM] Subsampled to {max_proteins} proteins")

    print(f"[OGT/ESM] Embedding {len(seqs)} proteins with ESM-2 ...")
    embeddings = []
    try:
        import esm as esm_lib
        model_esm, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model_esm.eval()
        for i in range(0, len(seqs), 8):
            batch = [(h, s[:1022]) for h, s in seqs[i:i+8]]
            _, _, tokens = batch_converter(batch)
            with torch.no_grad():
                reps = model_esm(tokens, repr_layers=[33])["representations"][33]
            for j, (_, s) in enumerate(batch):
                embeddings.append(reps[j, 1:len(s[:1022])+1].mean(0).numpy())
    except ImportError:
        from transformers import AutoTokenizer, EsmModel
        tok   = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm_m = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval()
        for i in range(0, len(seqs), 4):
            batch_seqs = [s[:1022] for _, s in seqs[i:i+4]]
            inp  = tok(batch_seqs, return_tensors="pt",
                       padding=True, truncation=True, max_length=1024)
            with torch.no_grad():
                out = esm_m(**inp)
            mask = inp["attention_mask"].unsqueeze(-1).float()
            embeddings.extend(((out.last_hidden_state * mask).sum(1) / mask.sum(1)).numpy())

    return np.mean(embeddings, axis=0).astype(np.float32)


# ============================================================
# Prediction from FASTA
# ============================================================

def predict_ogt_from_fasta(fasta_path, model_dir=None, tmp_dir=None) -> float:
    """Predict OGT from a genome or proteome FASTA.

    Extracts the mean-pooled ESM-2 proteome embedding, then applies the
    trained MLP. Use predict_ogt_from_embedding() directly when the embedding
    is already available (e.g. reusing the TPC embedding avoids a second ESM pass).

    Parameters
    ----------
    fasta_path : str or Path -- nucleotide genome or amino acid proteome FASTA
    model_dir  : path to results/ogt_mlp/ (uses default if None)
    tmp_dir    : working directory for Prodigal intermediate files

    Returns
    -------
    float -- predicted OGT in degrees C
    """
    emb = _extract_esm_embedding(fasta_path, tmp_dir=tmp_dir)
    return predict_ogt_from_embedding(emb, model_dir=model_dir)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGT predictor: train or predict")
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train", help="Train OGT MLP from a CSV with ESM embeddings")
    train_p.add_argument("--data", required=True, type=Path,
                         help="CSV with esm2_0...esm2_1279 columns and an OGT column")

    pred_p = sub.add_parser("predict", help="Predict OGT from FASTA or embedding file")
    grp = pred_p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--fasta",     type=Path,
                     help="Genome (nucleotide) or proteome (amino acid) FASTA")
    grp.add_argument("--embedding", type=Path,
                     help="Pre-computed ESM-2 embedding as .npy file (shape 1280)")
    pred_p.add_argument("--model_dir", type=Path, default=RESULTS_DIR,
                         help="Directory containing mlp.pkl and scaler.pkl")
    pred_p.add_argument("--tmp_dir",   type=Path, default=None,
                         help="Working directory for Prodigal output")

    args = parser.parse_args()

    if args.mode == "train":
        train_ogt_mlp(args.data)

    elif args.mode == "predict":
        if args.fasta:
            ogt = predict_ogt_from_fasta(args.fasta, model_dir=args.model_dir,
                                         tmp_dir=args.tmp_dir)
        else:
            emb = np.load(args.embedding)
            ogt = predict_ogt_from_embedding(emb, model_dir=args.model_dir)
        print(f"Predicted OGT: {ogt:.1f} C")
