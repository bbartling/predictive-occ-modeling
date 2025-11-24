"""
Predictive Occupancy Modeling
=============================

This top‑level ``src`` package serves as a thin wrapper around the
``predictive_occ`` library.  Historically, the modelling code lived
directly in this package, but it has been refactored into a proper
subpackage to support cleaner imports and reuse across notebooks,
scripts and API layers.

For backward compatibility the most commonly used functions remain
available from this namespace:

* ``load_occupancy_data`` and ``clean_occupancy_counts`` for loading and
  preprocessing CSV data.
* ``adf_stationary`` and ``kpss_stationary`` for stationarity tests.
* ``infer_step_minutes``, ``build_probability_table`` and
  ``export_probability_schedule`` for building a probability‑based model.

New code should import these functions from ``predictive_occ`` rather
than from ``src`` directly.
"""

from .data_processing import load_occupancy_data, clean_occupancy_counts  # type: ignore[F401]
from .stationarity import adf_stationary, kpss_stationary  # type: ignore[F401]
from .baseline_models import (
    infer_step_minutes,
    build_probability_table,
    export_probability_schedule,
)  # type: ignore[F401]

__all__ = [
    "load_occupancy_data",
    "clean_occupancy_counts",
    "adf_stationary",
    "kpss_stationary",
    "infer_step_minutes",
    "build_probability_table",
    "export_probability_schedule",
]