"""Configuration Utilities
===========================

This module provides a lightweight helper for loading configuration
files used by the predictive occupancy modelling pipeline.  The
configuration is expected to be provided as a JSON document on disk.
Keys in the JSON can override the command‑line defaults supplied to
the pipeline.  Using a configuration file allows datasets with
different column names, time zones or cleaning parameters to be
processed without modifying the source code or rewriting the
command‑line invocation each time.

Example ``config.json``::

    {
      "data_path": "data/occupancy_sample.csv",
      "time_col": "time",
      "occ_col": "occ",
      "tz": "UTC",
      "deadband": 1.0,
      "prob_threshold": 0.5,
      "plots_dir": "plots",
      "schedule_path": "data/final_schedule.csv"
    }

Place the file anywhere on disk and point the ``--config`` option
to it.  Any values present in the configuration will override the
corresponding command‑line defaults.  Unknown keys are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Filepath pointing to a JSON document.

    Returns
    -------
    dict
        Parsed key/value pairs from the JSON file.  If the file
        cannot be read or parsed, an empty dictionary is returned.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure keys are strings for consistency
        return {str(k): v for k, v in data.items()}
    except Exception:
        # Any error results in an empty configuration.  The pipeline
        # will fall back to command‑line defaults in this case.
        return {}
