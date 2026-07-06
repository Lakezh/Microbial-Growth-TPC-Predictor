#!/usr/bin/env python3
"""
Example: E. coli K-12 MG1655 (mesophile, OGT ~37 C)

This example demonstrates TPC prediction without FBA (normalised output).
Replace `esm_embedding` and `ogt_c` with values computed from a real genome.

Pre-requisites
--------------
    pip install torch scikit-learn numpy pandas matplotlib
    (optional for FBA): pip install cobra carveme
"""

import sys
from pathlib import Path
import numpy as np

# Add code/ to Python path so we can import without installation
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
print(f"Model expects ESM embedding of length {EMB_LEN}")

# Replace this with the mean-pooled ESM-2 embedding for E. coli K-12 MG1655.
# To compute it:
#   python code/TPC_predictor.py --fasta GCF_000005845.2_ASM584v2_genomic.fna --ogt 37.0
np.random.seed(0)
esm_embedding = np.random.randn(EMB_LEN).astype(np.float32)

ogt_c        = 37.0   # E. coli optimal growth temperature
temperatures = np.arange(5, 75, 1, dtype=np.float32)

# ============================================================
# Prediction
# ============================================================
result = predict_shape(model, scaler, meta, device, esm_embedding, ogt_c, temperatures)

print(f"\nE. coli K-12 MG1655")
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
}).to_csv(out_dir / "ecoli_tpc.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(result["temperatures"], result["pred_shape"],
        color="steelblue", lw=2, label="Normalised growth rate")
ax.axvline(ogt_c, color="tomato", ls="--", lw=1.5, label=f"OGT = {ogt_c} C")
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("Normalised growth rate")
ax.set_title("E. coli K-12 MG1655 -- TPC prediction")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / "ecoli_tpc.png", dpi=150)
plt.close()

print(f"\nOutputs saved to: {out_dir}")
print("  ecoli_tpc.csv")
print("  ecoli_tpc.png")
