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
    build_mean_pivot,
    build_probability_table,
    evaluate_baseline_models,
    export_probability_schedule,
)
from .plotting import (
    plot_time_series,
    plot_histogram,
    plot_average_by_hour,
    plot_weekday_vs_weekend,
    plot_probability_heatmap,
    plot_raw_vs_clean,
)

__all__ = [
    "load_occupancy_data",
    "clean_occupancy_counts",
    "adf_stationary",
    "kpss_stationary",
    "infer_step_minutes",
    "build_mean_pivot",
    "build_probability_table",
    "evaluate_baseline_models",
    "export_probability_schedule",
    "plot_time_series",
    "plot_histogram",
    "plot_average_by_hour",
    "plot_weekday_vs_weekend",
    "plot_probability_heatmap",
    "plot_raw_vs_clean",
]