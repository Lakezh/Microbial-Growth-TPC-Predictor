"""
FBA Peak Anchor
===============
Uses COBRApy to run Flux Balance Analysis (FBA) under user-specified medium
conditions and returns the maximum growth rate, which anchors the absolute
peak of the TPC.

Rationale
---------
The PINN outputs a *normalised* TPC (peak = 1).  The absolute scale depends
on the organism's metabolic capacity under a given medium composition.  FBA
under that medium — at steady state, optimal temperature — provides a
physiologically grounded peak rate:

    growth_rate(T) = normalised_shape(T) × fba_peak_rate

The medium is specified as exchange-reaction upper bounds (same format used
by COBRApy's ``model.medium``).  A positive value allows uptake of the
corresponding metabolite.

Example
-------
>>> anchor = FBAAnchor("models/iJO1366.xml")
>>> peak   = anchor.get_peak_growth_rate(
...     medium={"EX_glc__D_e": 10.0, "EX_o2_e": 20.0, "EX_nh4_e": 10.0}
... )
>>> print(f"Peak growth rate: {peak:.4f} h-1")
"""

from pathlib import Path
from typing import Union, Dict, Optional


class FBAAnchor:
    """
    FBA-based peak growth-rate estimator.

    Parameters
    ----------
    model_path : path-like
        Path to a SBML-format genome-scale metabolic model (e.g. ``iJO1366.xml``).
    objective_id : str, optional
        Reaction ID of the biomass objective.  ``None`` uses the model's
        pre-set objective (default).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        objective_id: Optional[str] = None,
    ):
        try:
            import cobra
        except ImportError as exc:
            raise ImportError(
                "COBRApy is required for FBA.  "
                "Install it with:  pip install cobra"
            ) from exc

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Metabolic model not found: {self._model_path}")

        self._cobra_model = cobra.io.read_sbml_model(str(self._model_path))
        if objective_id is not None:
            self._cobra_model.objective = objective_id

    # ------------------------------------------------------------------
    def get_peak_growth_rate(self, medium: Dict[str, float]) -> float:
        """
        Run FBA with the given medium and return the optimal growth rate.

        Parameters
        ----------
        medium : dict
            Mapping from exchange-reaction ID to maximum uptake rate
            (mmol gDW⁻¹ h⁻¹).  Only reactions listed here are opened;
            all others are closed.

            Example::

                {
                    "EX_glc__D_e": 10.0,   # glucose uptake
                    "EX_o2_e":     20.0,   # oxygen uptake
                    "EX_nh4_e":    10.0,   # ammonium
                    "EX_pi_e":      5.0,   # phosphate
                }

        Returns
        -------
        float
            Maximum growth rate in h⁻¹.

        Raises
        ------
        RuntimeError
            If FBA returns a non-optimal solution.
        """
        # Work inside a context manager so the original model is unchanged
        with self._cobra_model as m:
            m.medium = medium
            solution = m.optimize()

        if solution.status != "optimal":
            raise RuntimeError(
                f"FBA returned non-optimal status '{solution.status}'. "
                "Check that the medium is feasible."
            )
        return float(solution.objective_value)

    # ------------------------------------------------------------------
    def get_exchange_reactions(self) -> list:
        """Return a list of exchange reaction IDs in the model."""
        return [r.id for r in self._cobra_model.exchanges]

    def summary(self) -> str:
        m = self._cobra_model
        return (
            f"Metabolic model: {self._model_path.name}\n"
            f"  Reactions  : {len(m.reactions)}\n"
            f"  Metabolites: {len(m.metabolites)}\n"
            f"  Genes      : {len(m.genes)}\n"
            f"  Objective  : {m.objective.to_json()['expression']}"
        )
