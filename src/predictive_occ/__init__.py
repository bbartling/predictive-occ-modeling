"""
Core library for predictive occupancy modeling.

This subpackage contains the reusable components for loading occupancy data,
performing basic cleaning, testing for stationarity and building a simple
probability‑based model.  The functions defined here are completely
independent of any API or CLI wrapper.  They can be imported from
notebooks, scripts or other applications.

For backwards compatibility the root ``src`` package re‑exports a subset of
these functions.  New code should prefer to import from
``predictive_occ`` directly, e.g.::

    from predictive_occ.io import load_occupancy_data
    from predictive_occ.model import build_probability_table
    from predictive_occ.stationarity import adf_stationary

"""

from .io import load_occupancy_data, clean_occupancy_counts
from .stationarity import adf_stationary, kpss_stationary
from .model import (
    infer_step_minutes,
    build_probability_table,
    export_probability_schedule,
    compare_to_std,
)

__all__ = [
    "load_occupancy_data",
    "clean_occupancy_counts",
    "adf_stationary",
    "kpss_stationary",
    "infer_step_minutes",
    "build_probability_table",
    "export_probability_schedule",
    "compare_to_std",
]