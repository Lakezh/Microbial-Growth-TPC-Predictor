"""
OGT Predictor
=============
Loads the pre-trained sklearn MLP and StandardScaler to predict the Optimal
Growth Temperature (OGT, °C) from 526 genomic features.

The 526 features cover:
  - Genome size
  - rRNA nucleotide composition (5S / 16S / 23S) and MFE/length
  - tRNA nucleotide composition
  - Proteome amino-acid fractions (raw and genome-GC-normalised)
  - Proteome dipeptide frequencies
  - Codon usage (ORF-level synonymous codon fractions)

Training data: 2 869 Bacteria + 262 Archaea (total 3 131 genomes).
CV performance (10-fold): RMSE = 5.12 °C, MAE = 3.91 °C, R² = 0.87.
"""

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Columns that are metadata rather than numeric features
_META_COLS = frozenset([
    "Unnamed: 0", "domain", "phylum", "class", "order",
    "family", "genus", "species", "OGT",
])


class OGTPredictor:
    """
    Predict OGT from genomic features.

    Parameters
    ----------
    model_dir : path-like
        Directory containing ``mlp.pkl``, ``scaler.pkl``,
        and ``feature_cols.pkl``.
    """

    def __init__(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        with open(model_dir / "scaler.pkl", "rb") as fh:
            self._scaler = pickle.load(fh)
        with open(model_dir / "mlp.pkl", "rb") as fh:
            self._model = pickle.load(fh)
        with open(model_dir / "feature_cols.pkl", "rb") as fh:
            self.feature_cols: list = pickle.load(fh)

    # ------------------------------------------------------------------
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict OGT for one or more organisms.

        Parameters
        ----------
        features : DataFrame or ndarray
            If DataFrame, must contain all columns listed in
            ``self.feature_cols``.  If ndarray, shape must be
            ``(n_samples, n_features)`` in the same column order.

        Returns
        -------
        ogt : ndarray, shape (n_samples,)
            Predicted OGT in °C.
        """
        X = self._to_matrix(features)
        return self._model.predict(self._scaler.transform(X))

    def predict_single(self, feature_row: Union[pd.Series, np.ndarray]) -> float:
        """Return the scalar OGT prediction for a single organism."""
        if isinstance(feature_row, pd.Series):
            X = feature_row[self.feature_cols].values.astype(np.float64).reshape(1, -1)
        else:
            X = np.asarray(feature_row, dtype=np.float64).reshape(1, -1)
        return float(self._model.predict(self._scaler.transform(X))[0])

    # ------------------------------------------------------------------
    def _to_matrix(self, features):
        if isinstance(features, pd.DataFrame):
            return features[self.feature_cols].values.astype(np.float64)
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.feature_cols)
