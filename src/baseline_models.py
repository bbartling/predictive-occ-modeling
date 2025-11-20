"""
Baseline Models for Occupancy Prediction
=======================================

This module implements simple, explainable baselines for predicting
binary occupancy states from historical time‑series data.  The
approach is intentionally lightweight so that it can be run on
resource‑constrained Building Automation Systems (BAS) after the
training phase.  Two baseline strategies are provided:

``mean_count_baseline``
    For each discrete time slot (for example 15‑minute intervals),
    compute the average occupancy count.  Apply a deadband to
    convert the continuous average into a binary occupied/unoccupied
    decision.  This model is sensitive to sporadic spikes in the
    underlying data but is useful for comparison.

``probability_baseline``
    First convert raw occupancy counts into a binary variable (1 if
    the count exceeds a configurable threshold, 0 otherwise).  Then
    compute, for each time slot, the historical probability of being
    occupied (i.e. the fraction of samples in that slot that are
    occupied).  Finally threshold these probabilities to produce a
    binary schedule.  Because this model uses frequency rather than
    magnitude, it is more robust to sensor noise and is typically
    preferred when the data are stationary.

The functions below are largely adapted from the original project
scripts but have been refactored into a more modular form so that
they can be reused for arbitrary datasets.
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

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe with at least columns ``occ_col``, ``time_col``, ``dow`` and
        ``minute_of_day``.
    occ_col : str, default ``"occ"``
        Name of the occupancy count column.
    time_col : str, default ``"time"``
        Name of the timestamp column.
    step_minutes : int or ``None``
        Sampling interval in minutes.  If ``None`` the interval will be
        inferred from the data via :func:`infer_step_minutes`.

    Returns
    -------
    pivot : :class:`pandas.DataFrame`
        Pivot table where rows are slots (minutes past midnight) and
        columns are day‑of‑week (0=Monday).  Values are mean occupancy.
    step_minutes : int
        The sampling interval used to build the pivot table.
    """
    if step_minutes is None:
        step_minutes = infer_step_minutes(df, time_col=time_col)
    data = df.copy()
    data["slot"] = (data["minute_of_day"] // step_minutes) * step_minutes
    grouped = (
        data.groupby(["dow", "slot"])[occ_col]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="slot", columns="dow", values=occ_col)
    # Ensure a full 24‑hour range with no gaps
    all_slots = np.arange(0, 24 * 60, step_minutes)
    pivot = pivot.reindex(all_slots)
    # Interpolate missing values to avoid NaNs at the start/end
    pivot = pivot.interpolate(limit_direction="both")
    return pivot, step_minutes


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

    For each record in ``df_test`` compute the predicted occupancy
    using both baselines, then compare those predictions with the
    ground truth label ``y``.  Accuracy is defined as the fraction of
    correct predictions.

    Parameters
    ----------
    df_test : :class:`pandas.DataFrame`
        Test dataset with columns ``dow``, ``slot`` and binary label
        ``y`` indicating actual occupancy.
    mean_table : :class:`pandas.DataFrame`
        Table with columns ``dow``, ``slot`` and ``mean_occ`` produced
        by :func:`build_mean_pivot` (flattened into long format).
    prob_table : :class:`pandas.DataFrame`
        Table with columns ``dow``, ``slot`` and ``p_occ`` produced by
        :func:`build_probability_table`.
    occ_threshold : float
        Deadband threshold used to convert mean counts into binary
        predictions.
    prob_threshold : float, default ``0.5``
        Decision threshold applied to probabilities to obtain binary
        predictions.

    Returns
    -------
    :class:`pandas.DataFrame`
        A table with one row per model, containing the accuracy.
    """
    # Merge predictions onto the test set
    merged = (
        df_test
        .merge(mean_table, on=["dow", "slot"], how="left")
        .merge(prob_table, on=["dow", "slot"], how="left")
    )
    y_true = merged["y"].astype(int)
    # Mean baseline: threshold continuous mean
    y_pred_mean = (merged["mean_occ"] > occ_threshold).astype(int).fillna(0)
    # Probability baseline
    y_pred_prob = (merged["p_occ"] > prob_threshold).astype(int).fillna(0)
    acc_mean = (y_pred_mean == y_true).mean()
    acc_prob = (y_pred_prob == y_true).mean()
    return pd.DataFrame({
        "model": ["baseline_threshold", "probability_model"],
        "accuracy": [acc_mean, acc_prob],
    })


def flatten_mean_pivot(
    pivot: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a mean pivot table into long form for merging.

    The :func:`build_mean_pivot` function returns a pivot table with
    rows labelled by slot and columns labelled by day‑of‑week.  This
    helper unpivots the table into a long format with columns
    ``dow``, ``slot`` and ``mean_occ`` so that it can be merged with
    other tables for evaluation.

    Parameters
    ----------
    pivot : :class:`pandas.DataFrame`
        Pivot table of mean occupancy counts.

    Returns
    -------
    :class:`pandas.DataFrame`
        Long format table with columns ``dow``, ``slot`` and
        ``mean_occ``.
    """
    long = (
        pivot
        .reset_index()
        .melt(id_vars="slot", var_name="dow", value_name="mean_occ")
        .dropna(subset=["mean_occ"])
        .astype({"dow": int})
    )
    return long


def export_probability_schedule(
    prob_table: pd.DataFrame,
    output_path: str,
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
    # Interpolate the standard profile over the model's slot range
    std_interp = np.interp(
        model_profile.index,
        np.linspace(0, 24 * 60, len(std)),
        std["median_occ"],
    )
    rmse = np.sqrt(((model_profile.values - std_interp) ** 2).mean())
    return float(rmse)