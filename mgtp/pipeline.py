"""
MGTP Pipeline
=============
Orchestrates the three-stage TPC prediction workflow:

  Stage 1 — OGT Prediction
    Genomic features (526-dim)  →  MLP  →  OGT (°C)

  Stage 2 — TPC Shape
    ESM embedding + OGT  →  PINN/UDE  →  Normalised TPC shape (peak = 1)

  Stage 3 — FBA Peak Anchor  (optional)
    Medium conditions  →  FBA  →  Peak growth rate (h⁻¹)

Final output:
    growth_rate(T) = normalised_shape(T) × peak_rate
    (if FBA is disabled, peak_rate = 1.0 and the output is the normalised shape)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .ogt_predictor import OGTPredictor
from .tpc_shape import TPCShapePredictor
from .fba_anchor import FBAAnchor


class MGTPipeline:
    """
    Full TPC prediction pipeline.

    Parameters
    ----------
    ogt_model_dir : path-like
        Directory with the trained OGT MLP artifacts
        (``mlp.pkl``, ``scaler.pkl``, ``feature_cols.pkl``).
    tpc_model_dir : path-like
        Directory with the trained TPC PINN artifacts
        (``checkpoint.pt``, ``esm_scaler.pkl``).
    fba_model_path : path-like, optional
        Path to a SBML genome-scale metabolic model.
        If ``None``, FBA anchoring is disabled and the output is normalised.

    Examples
    --------
    Without FBA (normalised output):

    >>> pipe = MGTPipeline("models/ogt_mlp", "models/tpc_pinn")
    >>> T, rate, ogt = pipe.predict(
    ...     genomic_features=feat_df,
    ...     esm_embedding=esm_vec,
    ...     temperature_range=np.arange(10, 65, 2),
    ... )

    With FBA (absolute growth rates, h⁻¹):

    >>> pipe = MGTPipeline(
    ...     "models/ogt_mlp", "models/tpc_pinn",
    ...     fba_model_path="models/iJO1366.xml",
    ... )
    >>> medium = {"EX_glc__D_e": 10.0, "EX_o2_e": 20.0, "EX_nh4_e": 10.0}
    >>> T, rate, ogt = pipe.predict(
    ...     genomic_features=feat_df,
    ...     esm_embedding=esm_vec,
    ...     medium=medium,
    ...     temperature_range=np.arange(10, 65, 2),
    ... )
    """

    def __init__(
        self,
        ogt_model_dir:  Union[str, Path],
        tpc_model_dir:  Union[str, Path],
        fba_model_path: Optional[Union[str, Path]] = None,
    ):
        self.ogt_predictor  = OGTPredictor(ogt_model_dir)
        self.tpc_predictor  = TPCShapePredictor(tpc_model_dir)
        self.fba_anchor     = (FBAAnchor(fba_model_path)
                               if fba_model_path is not None else None)
        self._fba_enabled   = self.fba_anchor is not None

    # ------------------------------------------------------------------
    def predict(
        self,
        esm_embedding:    np.ndarray,
        genomic_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        ogt_override:     Optional[float] = None,
        medium:           Optional[Dict[str, float]] = None,
        temperature_range: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the full prediction pipeline for one organism.

        Parameters
        ----------
        esm_embedding : ndarray, shape (emb_len,)
            Mean-pooled ESM-2 embedding of the proteome.
        genomic_features : DataFrame or ndarray, optional
            526-dimensional genomic feature vector used to predict OGT.
            Required unless ``ogt_override`` is supplied.
        ogt_override : float, optional
            Skip OGT prediction and use this value directly (°C).
        medium : dict, optional
            COBRApy-format medium for FBA anchor.  Required when a
            metabolic model was supplied at construction time.
        temperature_range : ndarray, optional
            Temperature query points in °C.  Defaults to
            ``[OGT - 30, OGT + 20]`` in 1 °C steps.

        Returns
        -------
        temperatures : ndarray
            Temperature points (°C).
        growth_rates : ndarray
            Predicted growth rate at each temperature.
            Normalised (peak = 1) when FBA is disabled;
            absolute (h⁻¹) when FBA is enabled.
        ogt : float
            Predicted (or overridden) OGT in °C.
        """
        # ── Stage 1: OGT ────────────────────────────────────────────────
        if ogt_override is not None:
            ogt = float(ogt_override)
        elif genomic_features is not None:
            ogt = float(self.ogt_predictor.predict(genomic_features)[0])
        else:
            raise ValueError(
                "Either genomic_features or ogt_override must be provided."
            )

        # ── Build temperature array ──────────────────────────────────────
        if temperature_range is None:
            temperature_range = np.arange(
                max(0.0, ogt - 30), ogt + 21, 1.0, dtype=np.float32
            )
        temps = np.asarray(temperature_range, dtype=np.float32)

        # ── Stage 2: TPC shape ───────────────────────────────────────────
        norm_shape, params = self.tpc_predictor.predict(
            esm_embedding=esm_embedding,
            ogt_c=ogt,
            temperatures=temps,
        )

        # ── Stage 3: FBA peak anchor ─────────────────────────────────────
        if self._fba_enabled:
            if medium is None:
                raise ValueError(
                    "A medium dict is required when an FBA model is loaded."
                )
            peak_rate   = self.fba_anchor.get_peak_growth_rate(medium)
            growth_rates = norm_shape * peak_rate
        else:
            peak_rate   = 1.0
            growth_rates = norm_shape

        return np.sort(temps), growth_rates, ogt

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        records: list,
    ) -> list:
        """
        Predict TPC for multiple organisms.

        Parameters
        ----------
        records : list of dict
            Each element is a keyword-argument dict for :meth:`predict`.

        Returns
        -------
        list of dict, each with keys:
            ``temperatures``, ``growth_rates``, ``ogt``
        """
        results = []
        for i, rec in enumerate(records):
            T, gr, ogt = self.predict(**rec)
            results.append({"temperatures": T, "growth_rates": gr, "ogt": ogt})
        return results

    # ------------------------------------------------------------------
    @property
    def fba_enabled(self) -> bool:
        """Whether FBA anchoring is active."""
        return self._fba_enabled

    def __repr__(self) -> str:
        fba = (Path(self.fba_anchor._model_path).name
               if self._fba_enabled else "disabled")
        return (
            f"MGTPipeline(\n"
            f"  ogt_features = {self.ogt_predictor.n_features},\n"
            f"  esm_dim      = {self.tpc_predictor.emb_len},\n"
            f"  fba          = {fba}\n"
            f")"
        )
