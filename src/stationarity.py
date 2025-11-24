"""
Legacy import wrapper for stationarity testing functions.

The stationarity tests have been relocated to ``predictive_occ.stationarity``.
This module re‑exports those functions so that existing imports of
``src.stationarity`` continue to work.
"""

from predictive_occ.stationarity import adf_stationary, kpss_stationary

__all__ = ["adf_stationary", "kpss_stationary"]