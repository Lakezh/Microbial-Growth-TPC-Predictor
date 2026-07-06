"""
Train OGT MLP
=============
Trains a Multi-Layer Perceptron to predict Optimal Growth Temperature (OGT)
from 526 genomic features (bacteria + archaea).

Two phases are executed:
  1. 10-fold cross-validation  — reports RMSE / MAE / R² on held-out data
  2. Final training on ALL data — saves model artifacts for deployment

Artifacts saved to ``--output_dir`` (default: ``../models/ogt_mlp/``):
  mlp.pkl          — trained sklearn MLPRegressor
  scaler.pkl       — StandardScaler fitted on all training data
  feature_cols.pkl — ordered list of the 526 feature column names
  cv_results.json  — per-fold and overall CV metrics

Usage
-----
python train/train_ogt.py \\
    --bacteria_csv data/calculated_features_bacteria.csv \\
    --archaea_csv  data/calculated_features_archaea.csv \\
    --output_dir   models/ogt_mlp
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Constants ────────────────────────────────────────────────────────────────
META_COLS = frozenset([
    "Unnamed: 0", "domain", "phylum", "class", "order",
    "family", "genus", "species", "OGT",
])

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    batch_size=64,
    random_state=42,
)

N_FOLDS    = 10
RANDOM_STATE = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(bacteria_csv: Path, archaea_csv: Path):
    df_b = pd.read_csv(bacteria_csv)
    df_a = pd.read_csv(archaea_csv)
    df   = pd.concat([df_b, df_a], ignore_index=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(np.float64)
    y = df["OGT"].values.astype(np.float64)

    print(f"Loaded {len(df)} samples "
          f"({len(df_b)} Bacteria + {len(df_a)} Archaea), "
          f"{len(feature_cols)} features")
    print(f"OGT range: [{y.min():.1f}, {y.max():.1f}] C")
    return X, y, feature_cols


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(X, y):
    kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_preds  = np.zeros(len(y))
    fold_rows  = []

    print(f"\n--- {N_FOLDS}-fold cross-validation ---")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        mlp = MLPRegressor(**MLP_PARAMS)
        mlp.fit(X_tr_s, y_tr)
        y_hat = mlp.predict(X_te_s)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        mae  = float(mean_absolute_error(y_te, y_hat))
        r2   = float(r2_score(y_te, y_hat))

        fold_rows.append({
            "fold": fold + 1,
            "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "n_iter":  int(mlp.n_iter_),
        })
        all_preds[te_idx] = y_hat
        print(f"  Fold {fold+1:2d}/{N_FOLDS}: "
              f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  (iters={mlp.n_iter_})")

    ov_rmse = float(np.sqrt(mean_squared_error(y, all_preds)))
    ov_mae  = float(mean_absolute_error(y, all_preds))
    ov_r2   = float(r2_score(y, all_preds))

    print(f"\n{'='*50}")
    print(f"Overall 10-fold CV:  RMSE={ov_rmse:.4f}  "
          f"MAE={ov_mae:.4f}  R2={ov_r2:.4f}")
    print(f"{'='*50}")

    overall = {"rmse": round(ov_rmse, 4),
               "mae":  round(ov_mae,  4),
               "r2":   round(ov_r2,   4)}
    return fold_rows, overall


# ── Final training + save ────────────────────────────────────────────────────

def train_and_save(X, y, feature_cols, output_dir: Path, fold_rows, overall):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Training final model on all data ---")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    mlp    = MLPRegressor(**MLP_PARAMS)
    mlp.fit(X_s, y)
    print(f"  Converged in {mlp.n_iter_} iterations")

    with open(output_dir / "mlp.pkl", "wb") as fh:
        pickle.dump(mlp, fh)
    with open(output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open(output_dir / "feature_cols.pkl", "wb") as fh:
        pickle.dump(feature_cols, fh)

    cv_results = {
        "model":        "MLPRegressor",
        "hidden_layers": list(MLP_PARAMS["hidden_layer_sizes"]),
        "n_folds":       N_FOLDS,
        "fold_results":  fold_rows,
        "overall":       overall,
    }
    with open(output_dir / "cv_results.json", "w") as fh:
        json.dump(cv_results, fh, indent=2)

    print(f"\nArtifacts saved to: {output_dir}/")
    for f in ["mlp.pkl", "scaler.pkl", "feature_cols.pkl", "cv_results.json"]:
        print(f"  {f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 on Windows consoles that default to cp932
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train OGT MLP predictor")
    parser.add_argument("--bacteria_csv", required=True,
                        help="Path to calculated_features_bacteria.csv")
    parser.add_argument("--archaea_csv",  required=True,
                        help="Path to calculated_features_archaea.csv")
    parser.add_argument("--output_dir",   default="models/ogt_mlp",
                        help="Directory where model artifacts are saved")
    args = parser.parse_args()

    X, y, feature_cols = load_data(
        Path(args.bacteria_csv), Path(args.archaea_csv)
    )
    fold_rows, overall = cross_validate(X, y)
    train_and_save(X, y, feature_cols, Path(args.output_dir), fold_rows, overall)


if __name__ == "__main__":
    main()
