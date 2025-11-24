"""
Probability‑Based Model for Predictive Occupancy
===============================================

This module contains the core logic for building a simple,
explainable occupancy model based on empirical probabilities.  It is
adapted directly from the original ``baseline_models`` module and
provides the same public functions:

* ``infer_step_minutes`` – infer the most common sampling interval in
  minutes.
* ``build_probability_table`` – compute the probability of occupancy
  for each day‑of‑week and time slot.
* ``export_probability_schedule`` – convert the probability table into a
  binary schedule matrix and optionally write it to CSV.
* ``compare_to_std`` – compute the RMSE between a model profile and a
  standard week profile (used for evaluation in notebooks).

Deprecated functions for mean‑based baselines (``build_mean_pivot``,
``evaluate_baseline_models``, ``flatten_mean_pivot``) remain as
``NotImplementedError`` to alert callers that they are no longer part
of the API.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def infer_step_minutes(df: pd.DataFrame, time_col: str = "time") -> int:
    """Infer the most common sampling interval in minutes.

    Given a series of timestamps, compute the difference between
    consecutive samples and return the mode of the differences in
    minutes.  If no differences are found, a default of 15 minutes
    is returned.
    """
    diffs = df[time_col].sort_values().diff().dropna()
    if diffs.empty:
        return 15
    step = int(diffs.mode().iloc[0].total_seconds() // 60)
    return max(step, 1)


def build_mean_pivot(
    df: pd.DataFrame,
    occ_col: str = "occ",
    time_col: str = "time",
    step_minutes: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """Construct a pivot table of mean occupancy by day‑of‑week and time slot.

    This function is deprecated and no longer implemented.  The project
    relies exclusively on the probability‑based baseline, which does
    not use mean counts.  Calling this function will raise an
    exception to alert callers that it has been removed.
    """
    raise NotImplementedError(
        "build_mean_pivot has been removed. Use the probability model instead."
    )


def build_probability_table(
    df: pd.DataFrame,
    occ_threshold: float,
    step_minutes: int,
    occ_col: str = "occ",
) -> pd.DataFrame:
    """Compute the probability of occupancy for each slot.

    This function first derives a binary label ``y`` indicating whether
    the raw count exceeds ``occ_threshold``.  It then groups by
    day‑of‑week and time slot to calculate the mean of ``y`` (which is
    equivalent to the probability of occupancy).

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe containing occupancy counts and derived fields
        ``dow`` and ``minute_of_day``.
    occ_threshold : float
        Counts less than or equal to this threshold are treated as
        unoccupied (0), otherwise occupied (1).
    step_minutes : int
        Sampling interval used to bucket the data into slots.
    occ_col : str, default ``"occ"``
        Name of the occupancy count column.

    Returns
    -------
    :class:`pandas.DataFrame`
        Table with columns ``dow``, ``slot`` and ``p_occ`` representing
        the probability of being occupied for each day and slot.
    """
    data = df.copy()
    data["slot"] = (data["minute_of_day"] // step_minutes) * step_minutes
    data["y"] = (data[occ_col] > occ_threshold).astype(int)
    prob_table = (
        data.groupby(["dow", "slot"])["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "p_occ"})
    )
    return prob_table


def evaluate_baseline_models(
    df_test: pd.DataFrame,
    mean_table: pd.DataFrame,
    prob_table: pd.DataFrame,
    occ_threshold: float,
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare the accuracy of the mean and probability baselines.

    This function is deprecated and no longer implemented.  The
    project no longer compares mean and probability baselines,
    because mean‑based baselines have been removed.  Calling this
    function will raise an exception to alert callers that it has
    been removed.
    """
    raise NotImplementedError(
        "evaluate_baseline_models has been removed. Use the probability model instead."
    )


def flatten_mean_pivot(
    pivot: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a mean pivot table into long form for merging.

    The :func:`build_mean_pivot` function returns a pivot table with
    rows labelled by slot and columns labelled by day‑of‑week.  This
    helper is now deprecated and no longer implemented.
    """
    raise NotImplementedError(
        "flatten_mean_pivot has been removed. Use the probability model instead."
    )


def export_probability_schedule(
    prob_table: pd.DataFrame,
    output_path: str | None,
    prob_threshold: float = 0.5,
    step_minutes: int = 15,
) -> pd.DataFrame:
    """Create a matrix lookup table and write it to a CSV file.

    The probability table contains rows indexed by ``dow`` and
    ``slot``.  The returned matrix has rows representing the hour of
    day (in fractional hours) and columns representing day of week.
    Each value is 0 or 1 depending on whether the probability of
    occupancy exceeds ``prob_threshold``.  The output is suitable for
    ingestion by a Building Automation System (BAS).

    Parameters
    ----------
    prob_table : :class:`pandas.DataFrame`
        Table with columns ``dow``, ``slot`` and ``p_occ``.
    output_path : str
        Path to write the resulting CSV file.
    prob_threshold : float, default ``0.5``
        Threshold used to convert probabilities into binary decisions.
    step_minutes : int, default ``15``
        The slot length in minutes.  This is used to convert ``slot``
        values into fractional hour indices.

    Returns
    -------
    :class:`pandas.DataFrame`
        The pivoted decision matrix with rows labelled by fractional
        hours and columns by day of week.
    """
    tbl = prob_table.copy()
    tbl["decision"] = (tbl["p_occ"] > prob_threshold).astype(int)
    tbl["hour"] = tbl["slot"] / 60.0
    matrix = (
        tbl
        .pivot(index="hour", columns="dow", values="decision")
        .fillna(0)
        .astype(int)
    )
    if output_path:
        matrix.to_csv(output_path)
    return matrix


def compare_to_std(
    pivot: pd.DataFrame,
    std: pd.DataFrame,
) -> float:
    """Compute the RMSE between a model profile and a standard week profile.

    A "standard week" profile summarises a typical occupancy pattern
    for each time slot and is provided in ``std_week.csv``.  This
    function interpolates the standard profile to match the number of
    slots in the model pivot table and then computes the root mean
    squared error (RMSE) between the two profiles.

    Parameters
    ----------
    pivot : :class:`pandas.DataFrame`
        A pivot table of mean occupancy counts with slot indices.
    std : :class:`pandas.DataFrame`
        Dataframe with a column ``median_occ`` representing the
        standard week occupancy profile.

    Returns
    -------
    float
        Root mean squared error between the model profile and the
        interpolated standard profile.
    """
    model_profile = pivot.mean(axis=1)
    std_interp = np.interp(
        model_profile.index,
        np.linspace(0, 24 * 60, len(std)),
        std["median_occ"],
    )
    return float(np.sqrt(((model_profile.values - std_interp) ** 2).mean()))


__all__ = [
    "infer_step_minutes",
    "build_probability_table",
    "export_probability_schedule",
    "compare_to_std",
    "build_mean_pivot",
    "evaluate_baseline_models",
    "flatten_mean_pivot",
]