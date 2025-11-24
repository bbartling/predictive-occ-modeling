"""
Predictive Occupancy Modeling
=============================

This package contains reusable modules for working with time‑series
occupancy data.  The goal of the project is to make it easy to load
arbitrary datasets, check if they exhibit stationarity, build simple
baseline models, evaluate those models and generate schedule lookup
tables that can be consumed by Building Automation Systems (BAS).

The functions defined in the submodules are designed to be generic and
should work on any dataset that includes a time stamp column and a
numeric occupancy measurement.  See the top‑level ``README.md`` for
details on how to use these utilities.
"""

from .data_processing import load_occupancy_data, clean_occupancy_counts
from .stationarity import adf_stationary, kpss_stationary
from .baseline_models import (
    infer_step_minutes,
    build_probability_table,
    export_probability_schedule,
)

# Note: plotting utilities and mean-based baselines have been removed.
# They are no longer imported or re-exported from this package.  The
# probability-based baseline is now the sole modelling approach.

__all__ = [
    "load_occupancy_data",
    "clean_occupancy_counts",
    "adf_stationary",
    "kpss_stationary",
    "infer_step_minutes",
    "build_probability_table",
    "export_probability_schedule",
]