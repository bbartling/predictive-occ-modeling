"""
Legacy import wrapper for predictive occupancy I/O functions.

This file exists for backwards compatibility.  The core I/O functions
have been moved to the ``predictive_occ.io`` module.  Importing
``load_occupancy_data`` or ``clean_occupancy_counts`` from this module
will forward to the new implementation.
"""

from predictive_occ.io import load_occupancy_data, clean_occupancy_counts

__all__ = ["load_occupancy_data", "clean_occupancy_counts"]