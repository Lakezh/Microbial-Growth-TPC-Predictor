#!/usr/bin/env python3
"""
Example: E. coli K-12 MG1655 (mesophile, OGT ~37 C)

Inputs required
---------------
    esm_embedding  --  mean-pooled ESM-2 vector for the proteome (1280-dim numpy array)
    ogt_c          --  predicted OGT from OGT_predictor.py, or known value

To get the ESM-2 embedding, run ESM-2 (facebook/esm2_t33_650M_UR50D) over all
protein sequences in the proteome and mean-pool the per-residue representations.

To predict OGT from genomic features:
    python code/OGT_predictor.py predict --fasta GCF_000005845.2.fna

This example uses a random vector as a placeholder for the real embedding.
"""

import sys
import numpy as np
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "code"))

from TPC_predictor import load_model, predict_single, plot_prediction

# ============================================================
# Load model
# ============================================================
model, scaler, meta, device = load_model()
EMB_LEN = meta["emb_len"]

# ============================================================
# Inputs  (replace with real values)
# ============================================================
np.random.seed(0)
esm_embedding = np.random.randn(EMB_LEN).astype(np.float32)  # real ESM-2 embedding here

ogt_c        = 37.0
temperatures = np.arange(5, 75, 1, dtype=np.float32)

# ============================================================
# Predict
# ============================================================
result = predict_single(model, scaler, meta, device,
                        esm_embedding=esm_embedding,
                        ogt_c=ogt_c,
                        temperatures=temperatures)

print("E. coli K-12 MG1655")
print(f"  OGT: {result['ToptC']:.1f} C")
print(f"  Pmax: {result['Pmax']:.4f}  E: {result['E']:.4f}")
print(f"  Peak at: {temperatures[result['pred_shape'].argmax()]:.0f} C")

# ============================================================
# Save
# ============================================================
import pandas as pd
out_dir = REPO_DIR / "examples" / "output"
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"temperature_C": result["temperatures"],
              "norm_shape":    result["pred_shape"]}
             ).to_csv(out_dir / "ecoli_tpc.csv", index=False)

plot_prediction(result,
                title="E. coli K-12 MG1655 -- TPC prediction",
                save_path=str(out_dir / "ecoli_tpc.png"))

print(f"Saved to {out_dir}/")
