"""
I/O and preprocessing utilities for predictive occupancy modeling.

This module provides helper functions for loading and preprocessing
occupancy time‑series data.  It is a direct refactor of the former
``data_processing`` module; its public API remains unchanged.  The
functions defined here are intended to be imported from the
``predictive_occ.io`` namespace.
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
    rename_columns: bool = True,
) -> pd.DataFrame:
    """Load a CSV file containing occupancy measurements.

    This helper reads a CSV file and returns a tidy DataFrame with
    standardised column names and a handful of derived features.  It
    supports datasets where the timestamp and occupancy columns are
    named differently by allowing callers to pass ``time_col`` and
    ``occ_col``.  When ``rename_columns`` is true (the default) the
    timestamp column will be renamed to ``"time"`` and the occupancy
    column renamed to ``"occ"``.  All downstream functions in this
    package assume these canonical names, so enabling renaming avoids
    the need to pass column names repeatedly.

    Parameters
    ----------
    path : str or PathLike
        Path to a CSV file with at least a timestamp and an occupancy
        column.
    time_col : str, default ``"time"``
        Name of the column containing timestamps in the input CSV.
    occ_col : str, default ``"occ"``
        Name of the column containing the numeric occupancy counts in
        the input CSV.
    parse_dates : bool, default ``True``
        If ``True`` then the timestamp column will be converted to
        :class:`pandas.DatetimeIndex`.  If ``False`` the timestamps
        remain strings.
    tz : str or ``None``, default ``"UTC"``
        If provided, localise naive timestamps to the specified
        timezone.  If ``None``, timestamps remain naive.  This
        argument is ignored if ``parse_dates`` is ``False``.
    rename_columns : bool, default ``True``
        If ``True`` the columns specified by ``time_col`` and
        ``occ_col`` will be renamed to ``"time"`` and ``"occ"``
        respectively in the returned DataFrame.  Downstream functions
        rely on these canonical names.  If ``False`` the original
        names are preserved and callers must pass the column names
        explicitly to other functions.

    Returns
    -------
    :class:`pandas.DataFrame`
        A dataframe sorted by timestamp with additional columns
        ``dow`` (day‑of‑week) and ``minute_of_day`` (minutes since
        midnight).  When ``rename_columns`` is ``True`` the DataFrame
        will contain canonical column names ``"time"`` and ``"occ"``.
    """
    df = pd.read_csv(path)
    if parse_dates:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        if tz is not None:
            df[time_col] = df[time_col].dt.tz_convert(tz)
    if rename_columns:
        if time_col != "time":
            df.rename(columns={time_col: "time"}, inplace=True)
        if occ_col != "occ":
            df.rename(columns={occ_col: "occ"}, inplace=True)
        tcol = "time"
        ocol = "occ"
    else:
        tcol = time_col
        ocol = occ_col
    df = df.sort_values(tcol).reset_index(drop=True)
    df["dow"] = df[tcol].dt.dayofweek
    df["minute_of_day"] = df[tcol].dt.hour * 60 + df[tcol].dt.minute
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
    cleaned["clean_count"] = cleaned[occ_col].apply(lambda x: max(0.0, x))
    cleaned["clean_count"] = cleaned["clean_count"].apply(
        lambda x: 0.0 if x <= deadband else x
    )
    if make_binary:
        cleaned["is_occupied"] = cleaned["clean_count"] > 0.0
        cleaned["is_occupied_int"] = cleaned["is_occupied"].astype(int)
    return cleaned


__all__ = ["load_occupancy_data", "clean_occupancy_counts"]