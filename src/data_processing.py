"""
Data Processing Utilities
=========================

This module provides helper functions for loading and preprocessing
occupancy time‑series data.  The functions are designed to work with
generic CSV files where at least two columns are available: a
timestamp and a numeric occupancy measurement.  Additional feature
columns such as day‑of‑week and minute‑of‑day are derived to support
modeling and plotting.

The default column names assume a dataset similar to the example
provided in this repository (``occupancy_sample.csv``), but callers
can override them via keyword arguments.
"""

from __future__ import annotations

import pandas as pd
from typing import Union, Optional


def load_occupancy_data(
    path: Union[str, "os.PathLike"],
    time_col: str = "time",
    occ_col: str = "occ",
    parse_dates: bool = True,
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """Load a CSV file containing occupancy measurements.

    Parameters
    ----------
    path : str or PathLike
        Path to a CSV file with at least a timestamp and an occupancy
        column.
    time_col : str, default ``"time"``
        Name of the column containing timestamps.  Times are parsed
        using :func:`pandas.to_datetime` and optionally localized to
        ``tz``.
    occ_col : str, default ``"occ"``
        Name of the column containing the numeric occupancy counts.
    parse_dates : bool, default ``True``
        If ``True`` then the timestamp column will be converted to
        :class:`pandas.DatetimeIndex`.  If ``False`` the timestamps
        remain strings.
    tz : str or ``None``, default ``"UTC"``
        If provided, localize naive timestamps to the specified
        timezone.  If ``None``, timestamps remain naive.

    Returns
    -------
    :class:`pandas.DataFrame`
        A dataframe sorted by timestamp with additional columns
        ``dow`` (day‑of‑week) and ``minute_of_day`` (minutes since
        midnight).
    """
    df = pd.read_csv(path)
    if parse_dates:
        # Convert the time column to datetime.  We parse with ``utc=True``
        # so that any naive timestamps are assumed to be UTC.  If the
        # input already contains offset information, pandas will
        # normalise it to UTC.  A timezone conversion step is then
        # applied if ``tz`` is not "UTC".
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        if tz is not None:
            # Convert from UTC to the requested timezone.  If tz="UTC"
            # this is a no‑op.
            df[time_col] = df[time_col].dt.tz_convert(tz)
    df = df.sort_values(time_col).reset_index(drop=True)

    # Derive helper columns for grouping and visualization
    df["dow"] = df[time_col].dt.dayofweek
    df["minute_of_day"] = df[time_col].dt.hour * 60 + df[time_col].dt.minute
    return df


def clean_occupancy_counts(
    df: pd.DataFrame,
    occ_col: str = "occ",
    deadband: float = 1.0,
    make_binary: bool = True,
) -> pd.DataFrame:
    """Apply basic cleaning logic to a series of occupancy counts.

    The raw sensor data may contain negative counts or small positive
    values due to noise.  The cleaning strategy implemented here
    follows the example notebook and scripts in this repository:

      * Negative values are clipped to zero.
      * Counts less than or equal to ``deadband`` are set to zero.
      * A binary occupancy flag is optionally created.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe containing at least the occupancy column.
    occ_col : str, default ``"occ"``
        Name of the column containing raw occupancy counts.
    deadband : float, default ``1.0``
        Threshold below which counts are assumed to be noise and set
        to zero.  For example, if ``deadband`` is 1.0 then counts of
        1 or below are considered unoccupied.
    make_binary : bool, default ``True``
        If ``True`` a new column ``is_occupied`` (boolean) and
        ``is_occupied_int`` (integer) are added.

    Returns
    -------
    :class:`pandas.DataFrame`
        A copy of ``df`` with an additional ``clean_count`` column and
        optional binary columns.
    """
    cleaned = df.copy()
    # Step 1: clip negative values to zero
    cleaned["clean_count"] = cleaned[occ_col].apply(lambda x: max(0.0, x))
    # Step 2: apply deadband to suppress low counts
    cleaned["clean_count"] = cleaned["clean_count"].apply(
        lambda x: 0.0 if x <= deadband else x
    )
    if make_binary:
        cleaned["is_occupied"] = cleaned["clean_count"] > 0.0
        cleaned["is_occupied_int"] = cleaned["is_occupied"].astype(int)
    return cleaned