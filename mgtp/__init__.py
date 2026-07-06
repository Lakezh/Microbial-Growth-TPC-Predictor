"""
MGTP — Microbial Growth TPC Predictor
======================================
A three-stage pipeline that predicts the full Temperature-Performance Curve (TPC)
for a microbial organism given:
  1. Genomic features  →  OGT (Optimal Growth Temperature) via MLP
  2. ESM embeddings + OGT  →  Normalised TPC shape via PINN/UDE
  3. Medium composition  →  Peak growth rate via FBA (optional anchor)

Quick start
-----------
>>> from mgtp import MGTPipeline
>>> pipe = MGTPipeline(
...     ogt_model_dir  = "models/ogt_mlp",
...     tpc_model_dir  = "models/tpc_pinn",
...     fba_model_path = "models/iJO1366.xml",   # optional
... )
>>> T, rate, ogt = pipe.predict(
...     genomic_features = feature_df,
...     esm_embedding    = esm_vec,
...     medium           = {"EX_glc__D_e": 10, "EX_o2_e": 20},
...     temperature_range = np.arange(10, 65, 2),
... )
"""

from .ogt_predictor import OGTPredictor
from .tpc_shape import TPCShapePredictor, load_tpc_model
from .fba_anchor import FBAAnchor
from .pipeline import MGTPipeline

__version__ = "0.1.0"
__all__ = ["OGTPredictor", "TPCShapePredictor", "load_tpc_model",
           "FBAAnchor", "MGTPipeline"]
