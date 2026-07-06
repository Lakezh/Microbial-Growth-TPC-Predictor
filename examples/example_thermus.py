#!/usr/bin/env python3
"""
Example: Thermus thermophilus HB8 (thermophile, OGT ~65 C)

Demonstrates TPC prediction for an extreme thermophile.
Replace `esm_embedding` with the real ESM-2 embedding from the genome.

Pre-requisites
--------------
    pip install torch scikit-learn numpy pandas matplotlib
"""

import sys
from pathlib import Path
import numpy as np

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "code"))

from TPC_predictor import load_model, predict_shape

# ============================================================
# Simulated inputs -- replace with real values
# ============================================================

CHECKPOINT = REPO_DIR / "results" / "core_model_checkpoint.pt"
SCALER     = REPO_DIR / "results" / "core_model_scaler.pkl"

print("Loading core model ...")
model, scaler, meta, device = load_model(CHECKPOINT, SCALER)

EMB_LEN = meta["emb_len"]

# Replace with real ESM-2 mean-pooled embedding for T. thermophilus HB8.
# Genome: GCF_000091545.1 (NCBI)
np.random.seed(1)
esm_embedding = np.random.randn(EMB_LEN).astype(np.float32)

ogt_c        = 65.0   # T. thermophilus optimal growth temperature
temperatures = np.arange(20, 95, 1, dtype=np.float32)

# ============================================================
# Prediction
# ============================================================
result = predict_shape(model, scaler, meta, device, esm_embedding, ogt_c, temperatures)

print(f"\nThermus thermophilus HB8 (thermophile)")
print(f"  OGT used:  {result['ToptC']:.1f} C")
print(f"  UTPC Pmax: {result['Pmax']:.4f}")
print(f"  UTPC E:    {result['E']:.4f}")
print(f"  Temp range: {temperatures[0]:.0f} -- {temperatures[-1]:.0f} C")
print(f"  Peak normalised rate at: "
      f"{temperatures[np.argmax(result['pred_shape'])]:.0f} C")

# ============================================================
# Save and plot
# ============================================================
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = REPO_DIR / "examples" / "output"
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "temperature_C": result["temperatures"],
    "norm_shape":    result["pred_shape"],
}).to_csv(out_dir / "thermus_tpc.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(result["temperatures"], result["pred_shape"],
        color="firebrick", lw=2, label="Normalised growth rate")
ax.axvline(ogt_c, color="darkorange", ls="--", lw=1.5, label=f"OGT = {ogt_c} C")
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("Normalised growth rate")
ax.set_title("Thermus thermophilus HB8 -- TPC prediction")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "thermus_tpc.png", dpi=150)
plt.close()

print(f"\nOutputs saved to: {out_dir}")
print("  thermus_tpc.csv")
print("  thermus_tpc.png")
