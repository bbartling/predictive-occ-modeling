"""
Baseline Models for Occupancy Prediction (Probability-Only)
===========================================================

This module implements a single, explainable, **probability-based**
baseline model for predicting binary occupancy from historical
time-series data.

The method computes the empirical probability of occupancy for each
discrete time slot (e.g., 15-minute intervals) across multiple weeks
of historical data. These probabilities are thresholded using
``prob_threshold`` to generate a final occupied/unoccupied schedule.

This approach is:

- **non-parametric and fully explainable**
- **robust to noisy sensors**
- **lightweight enough for edge/BAS execution**

All former count-based mean models have been removed in favor of the
probability-only baseline.
"""

import pandas as pd
import numpy as np


def build_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the empirical probability of occupancy for each
    (day_of_week, minute_of_day) combination.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
        - 'dow' : integer day of week (0=Monday)
        - 'minute_of_day' : integer minute (0–1439)
        - 'is_occupied_int' : 0 or 1 occupancy observation

    Returns
    -------
    pandas.DataFrame
        With columns ['dow', 'minute_of_day', 'probability'] where
        each row represents the historical probability of being
        occupied at that specific weekly time slot.
    """
    grouped = df.groupby(["dow", "minute_of_day"])["is_occupied_int"].mean()
    result = grouped.reset_index()
    result = result.rename(columns={"is_occupied_int": "probability"})
    return result


def export_probability_schedule(
    df_prob: pd.DataFrame, *, prob_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Convert the probability table into a final binary occupancy schedule
    by thresholding.

    Parameters
    ----------
    df_prob : pandas.DataFrame
        Output of build_probability_table(). Must contain:
        - 'dow'
        - 'minute_of_day'
        - 'probability'

    prob_threshold : float, optional
        Values >= threshold map to 1 (occupied); otherwise 0.

    Returns
    -------
    pandas.DataFrame
        With columns ['dow', 'minute_of_day', 'schedule_occ'].
    """
    df = df_prob.copy()
    df["schedule_occ"] = (df["probability"] >= prob_threshold).astype(int)
    return df
