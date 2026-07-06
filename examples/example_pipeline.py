"""
Example: Full MGTP Pipeline
============================
Demonstrates two usage scenarios:

  A. Without FBA — outputs normalised TPC shape (peak = 1)
  B. With FBA    — outputs absolute TPC in h-1

Before running:
  1. Train the OGT model:
       python train/train_ogt.py \\
           --bacteria_csv <path>/calculated_features_bacteria.csv \\
           --archaea_csv  <path>/calculated_features_archaea.csv \\
           --output_dir   models/ogt_mlp

  2. Place the TPC PINN artifacts in models/tpc_pinn/:
         models/tpc_pinn/checkpoint.pt
         models/tpc_pinn/esm_scaler.pkl

  3. (Optional, for FBA) place an SBML model, e.g. iJO1366.xml, in models/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's directory
# ---------------------------------------------------------------------------
HERE       = Path(__file__).parent.parent          # repo root
OGT_DIR    = HERE / "models" / "ogt_mlp"
TPC_DIR    = HERE / "models" / "tpc_pinn"
FBA_MODEL  = HERE / "models" / "iJO1366.xml"       # optional
MEDIUM_FILE = HERE / "examples" / "example_medium_ecoli.json"
OUT_DIR    = HERE / "examples" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenario A — normalised TPC (no FBA)
# ---------------------------------------------------------------------------
def scenario_a():
    print("=" * 60)
    print("Scenario A: Normalised TPC (no FBA anchor)")
    print("=" * 60)

    from mgtp import MGTPipeline

    pipe = MGTPipeline(
        ogt_model_dir = OGT_DIR,
        tpc_model_dir = TPC_DIR,
    )
    print(pipe)

    # Replace with real genomic features (must match feature_cols.pkl)
    import pickle
    with open(OGT_DIR / "feature_cols.pkl", "rb") as fh:
        feat_cols = pickle.load(fh)
    n_feat = len(feat_cols)

    import pandas as pd
    rng  = np.random.default_rng(0)
    fake_features = pd.DataFrame(
        rng.standard_normal((1, n_feat)), columns=feat_cols
    )

    # Replace with real mean-pooled ESM-2 embedding
    emb_dim  = pipe.tpc_predictor.emb_len
    fake_esm = rng.standard_normal(emb_dim).astype(np.float32)

    T_range = np.arange(10, 70, 1, dtype=np.float32)

    T, gr, ogt = pipe.predict(
        esm_embedding    = fake_esm,
        genomic_features = fake_features,
        temperature_range = T_range,
    )

    print(f"\nPredicted OGT : {ogt:.2f} C")
    print(f"Peak normalised rate at OGT : {gr[np.argmin(np.abs(T - ogt))]:.4f}")
    _plot_tpc(T, gr, ogt, title="Scenario A — Normalised TPC",
              ylabel="Normalised growth rate",
              out_path=OUT_DIR / "scenario_a_tpc.png")


# ---------------------------------------------------------------------------
# Scenario B — absolute TPC with FBA anchor
# ---------------------------------------------------------------------------
def scenario_b():
    if not FBA_MODEL.exists():
        print("\nScenario B skipped — FBA model not found at:", FBA_MODEL)
        return

    print("\n" + "=" * 60)
    print("Scenario B: Absolute TPC with FBA peak anchor")
    print("=" * 60)

    from mgtp import MGTPipeline

    pipe = MGTPipeline(
        ogt_model_dir  = OGT_DIR,
        tpc_model_dir  = TPC_DIR,
        fba_model_path = FBA_MODEL,
    )
    print(pipe)

    # Load medium from JSON file
    with open(MEDIUM_FILE) as fh:
        medium = json.load(fh)
    print(f"Medium loaded: {len(medium)} exchange reactions")

    import pickle
    with open(OGT_DIR / "feature_cols.pkl", "rb") as fh:
        feat_cols = pickle.load(fh)

    import pandas as pd
    rng  = np.random.default_rng(0)
    fake_features = pd.DataFrame(
        rng.standard_normal((1, len(feat_cols))), columns=feat_cols
    )
    fake_esm = rng.standard_normal(pipe.tpc_predictor.emb_len).astype(np.float32)

    # Supply a known OGT (37 C for E. coli) instead of predicting it
    T, gr, ogt = pipe.predict(
        esm_embedding    = fake_esm,
        ogt_override     = 37.0,
        medium           = medium,
        temperature_range = np.arange(10, 55, 1, dtype=np.float32),
    )

    print(f"\nOGT (override) : {ogt:.1f} C")
    print(f"Peak growth rate (FBA): {gr.max():.4f} h-1")
    _plot_tpc(T, gr, ogt, title="Scenario B — Absolute TPC (FBA anchor)",
              ylabel="Growth rate (h-1)",
              out_path=OUT_DIR / "scenario_b_tpc.png")


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_tpc(T, gr, ogt, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T, gr, "-o", markersize=3, color="royalblue", linewidth=1.8,
            label="Predicted TPC")
    ax.axvline(ogt, color="tomato", linestyle="--", linewidth=1.4,
               label=f"OGT = {ogt:.1f} C")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scenario_a()
    scenario_b()
    print("\nDone.")
