#!/usr/bin/env python3
"""
OGT_predictor.py  --  Train and apply the OGT MLP predictor.

The OGT (Optimal Growth Temperature) MLP is a sklearn MLPRegressor trained on
526 genomic features extracted from 3131 bacterial and archaeal genomes.

Feature groups (526 total)
--------------------------
    genome_size           1
    rRNA nucleotide fractions + MFE/len  (5S, 16S, 23S)  15
    tRNA nucleotide fractions + MFE/len                    5
    GC content (normalised)                                1
    Proteome AA fractions (raw + GC-normalised)           40
    Proteome properties (mean length, charge ratios)       4
    Codon usage (synonymous codon fractions)              64
    Dipeptide frequencies                                400
    ----------------------------------------------------------
    Total                                                526

Modes
-----
    Training (from pre-computed CSV files):
        python code/OGT_predictor.py --mode train
            --bacteria_csv data/calculated_features_bacteria.csv
            --archaea_csv  data/calculated_features_archaea.csv

    Prediction (from a FASTA file, requires Prodigal + Barrnap on PATH):
        python code/OGT_predictor.py --mode predict --fasta genome.fna

    Prediction (from a pre-computed feature CSV):
        python code/OGT_predictor.py --mode predict --feature_csv features.csv
"""

import sys, os, json, argparse, subprocess, warnings, pickle, re
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

MLP_PATH          = RESULTS_DIR / "mlp.pkl"
SCALER_PATH       = RESULTS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = RESULTS_DIR / "feature_cols.pkl"
CV_RESULTS_PATH   = RESULTS_DIR / "cv_results.json"

# ============================================================
# MLP hyperparameters
# ============================================================
MLP_PARAMS = dict(
    hidden_layer_sizes = (256, 128, 64),
    activation         = "relu",
    solver             = "adam",
    alpha              = 1e-4,
    learning_rate_init = 1e-3,
    max_iter           = 500,
    early_stopping     = True,
    validation_fraction= 0.1,
    n_iter_no_change   = 20,
    batch_size         = 64,
    random_state       = 42,
)

# ============================================================
# Training
# ============================================================

def load_training_data(bacteria_csv: Path, archaea_csv: Path):
    """Load and merge the two genome feature CSVs; return X, y, feature_cols."""
    frames = []
    for p, lbl in [(bacteria_csv, "bacteria"), (archaea_csv, "archaea")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Feature CSV not found: {p}")
        df = pd.read_csv(p)
        df["kingdom"] = lbl
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    ogt_col = None
    for c in ["OGT", "ogt", "optimal_growth_temperature", "OGT_C"]:
        if c in df.columns:
            ogt_col = c
            break
    if ogt_col is None:
        raise ValueError("No OGT column found in feature CSV (expected 'OGT' or 'ogt')")

    df[ogt_col] = pd.to_numeric(df[ogt_col], errors="coerce")
    df = df.dropna(subset=[ogt_col]).reset_index(drop=True)

    exclude = {"kingdom", ogt_col, "species", "genome_id", "taxid",
               "binomial_name", "source"}
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df[ogt_col].values.astype(np.float32)

    print(f"Training data: {len(df)} samples | {len(feature_cols)} features")
    return X, y, feature_cols


def train_ogt_mlp(bacteria_csv: Path, archaea_csv: Path):
    """Train OGT MLP with 10-fold CV, then fit a final model on all data.

    Saves mlp.pkl, scaler.pkl, feature_cols.pkl, cv_results.json
    and training log to Train/.
    """
    X, y, feature_cols = load_training_data(bacteria_csv, archaea_csv)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_list, mae_list, r2_list = [], [], []
    fold_results = []

    print("\n10-fold cross-validation:")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        mlp = MLPRegressor(**MLP_PARAMS)
        mlp.fit(X_tr_s, y_tr)
        y_pred = mlp.predict(X_va_s)

        rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
        mae  = float(mean_absolute_error(y_va, y_pred))
        r2   = float(r2_score(y_va, y_pred))
        rmse_list.append(rmse); mae_list.append(mae); r2_list.append(r2)
        fold_results.append({"fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2})
        print(f"  Fold {fold:2d}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")

    print(f"\nOverall: RMSE={np.mean(rmse_list):.2f}+/-{np.std(rmse_list):.2f}  "
          f"MAE={np.mean(mae_list):.2f}  R2={np.mean(r2_list):.4f}")

    cv_summary = {
        "n_samples": int(len(y)),
        "n_features": int(len(feature_cols)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std":  float(np.std(rmse_list)),
        "mae_mean":  float(np.mean(mae_list)),
        "r2_mean":   float(np.mean(r2_list)),
        "folds":     fold_results,
        "mlp_params": MLP_PARAMS,
    }
    with open(CV_RESULTS_PATH, "w") as fh:
        json.dump(cv_summary, fh, indent=2)
    pd.DataFrame(fold_results).to_csv(TRAIN_DIR / "ogt_mlp_cv_log.csv", index=False)
    print(f"CV results saved: {CV_RESULTS_PATH}")

    print("\nFitting final model on all data ...")
    final_scaler = StandardScaler().fit(X)
    X_s = final_scaler.transform(X)
    final_mlp = MLPRegressor(**MLP_PARAMS)
    final_mlp.fit(X_s, y)

    with open(MLP_PATH,          "wb") as fh: pickle.dump(final_mlp,    fh)
    with open(SCALER_PATH,       "wb") as fh: pickle.dump(final_scaler,  fh)
    with open(FEATURE_COLS_PATH, "wb") as fh: pickle.dump(feature_cols,  fh)
    print(f"Model saved: {MLP_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Feature cols saved: {FEATURE_COLS_PATH}")

# ============================================================
# Load saved OGT model
# ============================================================

def load_ogt_model(model_dir=None):
    d = Path(model_dir) if model_dir else RESULTS_DIR
    with open(d / "mlp.pkl",          "rb") as fh: mlp   = pickle.load(fh)
    with open(d / "scaler.pkl",        "rb") as fh: scl   = pickle.load(fh)
    with open(d / "feature_cols.pkl",  "rb") as fh: cols  = pickle.load(fh)
    return mlp, scl, cols

# ============================================================
# Genomic feature extraction from FASTA
# ============================================================

_CODONS = [
    a+b+c
    for a in "ACGT" for b in "ACGT" for c in "ACGT"
]

_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

_DIPEPTIDES = [a+b for a in _AA_ALPHABET for b in _AA_ALPHABET]


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    tot = sum(seq.count(c) for c in "ACGT")
    return gc / tot if tot > 0 else 0.0


def nucleotide_fracs(seq: str) -> dict:
    seq = seq.upper()
    n = len(seq) or 1
    return {c: seq.count(c) / n for c in "ACGU"}


def aa_fractions(protein_seqs: list) -> dict:
    concat = "".join(protein_seqs).upper()
    n = len(concat) or 1
    return {aa: concat.count(aa) / n for aa in _AA_ALPHABET}


def codon_usage(cds_seqs: list) -> dict:
    counts = {c: 0 for c in _CODONS}
    total  = 0
    for cds in cds_seqs:
        cds = cds.upper().replace("U", "T")
        for i in range(0, len(cds) - 2, 3):
            codon = cds[i:i+3]
            if codon in counts:
                counts[codon] += 1
                total += 1
    n = total or 1
    return {c: v / n for c, v in counts.items()}


def dipeptide_freq(protein_seqs: list) -> dict:
    counts = {dp: 0 for dp in _DIPEPTIDES}
    total  = 0
    for seq in protein_seqs:
        seq = seq.upper()
        for i in range(len(seq) - 1):
            dp = seq[i:i+2]
            if dp in counts:
                counts[dp] += 1
                total += 1
    n = total or 1
    return {dp: v / n for dp, v in counts.items()}


def read_fasta(fasta_path):
    seqs = []
    header = seq = ""
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header and seq:
                    seqs.append((header, seq))
                header = line[1:].split()[0]; seq = ""
            else:
                seq += line
    if header and seq:
        seqs.append((header, seq))
    return seqs


def run_barrnap(genome_fasta, tmp_dir):
    """Run barrnap to find rRNA genes; return dict of 5S/16S/23S sequences."""
    tmp_dir = Path(tmp_dir)
    gff_out = tmp_dir / "rrna.gff"
    cmd = ["barrnap", "--quiet", "--outseq", str(tmp_dir / "rrna.fna"),
           str(genome_fasta)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    rrna_fasta = tmp_dir / "rrna.fna"
    if not rrna_fasta.exists():
        return {}
    seqs = read_fasta(rrna_fasta)
    rrna = {"5S": [], "16S": [], "23S": []}
    for h, s in seqs:
        for k in rrna:
            if k in h:
                rrna[k].append(s)
    return rrna


def extract_genomic_features(genome_fasta, tmp_dir=None, prodigal_proteins=None):
    """Extract 526 genomic features from a genome FASTA.

    Parameters
    ----------
    genome_fasta     : path to nucleotide genome FASTA
    tmp_dir          : working directory for Prodigal / Barrnap output
    prodigal_proteins: pre-computed list of (header, protein_seq) from Prodigal

    Returns
    -------
    dict mapping feature_name -> float value
    """
    genome_fasta = Path(genome_fasta)
    if tmp_dir is None:
        tmp_dir = genome_fasta.parent / "_tmp_ogt"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    contigs = read_fasta(genome_fasta)
    genome_seq = "".join(s for _, s in contigs).upper()
    genome_size = len(genome_seq)

    gc = gc_fraction(genome_seq)

    if prodigal_proteins is None:
        protein_fasta = Path(tmp_dir) / "proteins.faa"
        cds_fasta     = Path(tmp_dir) / "cds.fna"
        cmd = ["prodigal", "-i", str(genome_fasta),
               "-a", str(protein_fasta), "-d", str(cds_fasta),
               "-p", "meta", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Prodigal failed:\n{result.stderr}")
        protein_seqs_raw = read_fasta(protein_fasta)
        cds_seqs_raw     = read_fasta(cds_fasta)
    else:
        protein_seqs_raw = prodigal_proteins
        cds_seqs_raw = []

    protein_seqs = [s for _, s in protein_seqs_raw]
    cds_seqs     = [s for _, s in cds_seqs_raw]

    aa_fracs = aa_fractions(protein_seqs)
    cod_use  = codon_usage(cds_seqs)
    dip_freq = dipeptide_freq(protein_seqs)

    mean_len     = float(np.mean([len(s) for s in protein_seqs])) if protein_seqs else 0.0
    charge_pos   = float(np.mean([(s.count("K") + s.count("R")) / max(1, len(s))
                                  for s in protein_seqs])) if protein_seqs else 0.0
    charge_neg   = float(np.mean([(s.count("D") + s.count("E")) / max(1, len(s))
                                  for s in protein_seqs])) if protein_seqs else 0.0
    net_charge   = charge_pos - charge_neg

    rrna = run_barrnap(genome_fasta, tmp_dir)

    features = {"genome_size": float(genome_size), "gc_content": gc}
    features["mean_protein_length"] = mean_len
    features["charge_positive"]     = charge_pos
    features["charge_negative"]     = charge_neg
    features["net_charge"]          = net_charge

    for rna_type in ["5S", "16S", "23S"]:
        seqs_rt = rrna.get(rna_type, [])
        seq_cat = "".join(seqs_rt).upper().replace("T", "U")
        nfracs  = nucleotide_fracs(seq_cat)
        for nuc in "ACGU":
            features[f"rrna_{rna_type}_{nuc}"] = nfracs[nuc]
        features[f"rrna_{rna_type}_len"]     = float(len(seq_cat))
        features[f"rrna_{rna_type}_gc"]      = gc_fraction(seq_cat)

    for aa, v in aa_fracs.items():
        features[f"aa_{aa}"] = v
        features[f"aa_{aa}_gc_corr"] = v * gc

    for codon, v in cod_use.items():
        features[f"codon_{codon}"] = v

    for dp, v in dip_freq.items():
        features[f"dp_{dp}"] = v

    return features


# ============================================================
# Predict OGT from FASTA
# ============================================================

def predict_ogt_from_features(feature_dict: dict, model_dir=None) -> float:
    """Apply trained OGT MLP to a dict of genomic features."""
    mlp, scl, cols = load_ogt_model(model_dir)
    X = np.array([[feature_dict.get(c, 0.0) for c in cols]], dtype=np.float32)
    X_s = scl.transform(X)
    return float(mlp.predict(X_s)[0])


def predict_ogt_from_fasta(fasta_path, model_dir=None, tmp_dir=None) -> float:
    """Extract features from a FASTA file and predict OGT."""
    features = extract_genomic_features(fasta_path, tmp_dir=tmp_dir)
    return predict_ogt_from_features(features, model_dir=model_dir)


def predict_ogt_from_csv(feature_csv: Path, model_dir=None) -> pd.DataFrame:
    """Predict OGT for every row in a pre-computed feature CSV."""
    mlp, scl, cols = load_ogt_model(model_dir)
    df = pd.read_csv(feature_csv)
    X = df[[c for c in cols if c in df.columns]].fillna(0.0).values.astype(np.float32)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} feature columns missing, filled with 0")
    X_s = scl.transform(X)
    df["predicted_OGT_C"] = mlp.predict(X_s)
    return df

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGT predictor: train or predict")
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train", help="Train OGT MLP from feature CSVs")
    train_p.add_argument("--bacteria_csv", required=True, type=Path)
    train_p.add_argument("--archaea_csv",  required=True, type=Path)

    pred_p = sub.add_parser("predict", help="Predict OGT from FASTA or feature CSV")
    group = pred_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta",       type=Path, help="Genome FASTA file")
    group.add_argument("--feature_csv", type=Path, help="Pre-computed feature CSV")
    pred_p.add_argument("--model_dir",  type=Path, default=RESULTS_DIR)
    pred_p.add_argument("--tmp_dir",    type=Path, default=None)
    pred_p.add_argument("--output",     type=Path, default=None)

    args = parser.parse_args()

    if args.mode == "train":
        train_ogt_mlp(args.bacteria_csv, args.archaea_csv)

    elif args.mode == "predict":
        if args.fasta:
            ogt = predict_ogt_from_fasta(args.fasta, model_dir=args.model_dir,
                                         tmp_dir=args.tmp_dir)
            print(f"Predicted OGT: {ogt:.1f} C")
        else:
            df_out = predict_ogt_from_csv(args.feature_csv, model_dir=args.model_dir)
            out = args.output or args.feature_csv.with_suffix("_ogt_pred.csv")
            df_out.to_csv(out, index=False)
            print(f"Predictions saved to: {out}")
            print(df_out[["predicted_OGT_C"]].describe())
