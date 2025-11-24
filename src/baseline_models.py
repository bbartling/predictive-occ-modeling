"""
Legacy import wrapper for predictive occupancy modelling functions.

The probability‑based modelling functions have been relocated to
``predictive_occ.model``.  This module re‑exports those functions so
that existing imports of ``src.baseline_models`` continue to work.
Deprecated mean‑based functions remain available but raise
``NotImplementedError`` to alert callers that they are no longer part
of the API.
"""

from predictive_occ.model import (
    infer_step_minutes,
    build_probability_table,
    export_probability_schedule,
    compare_to_std,
    build_mean_pivot,
    evaluate_baseline_models,
    flatten_mean_pivot,
)

__all__ = [
    "infer_step_minutes",
    "build_probability_table",
    "export_probability_schedule",
    "compare_to_std",
    "build_mean_pivot",
    "evaluate_baseline_models",
    "flatten_mean_pivot",
]