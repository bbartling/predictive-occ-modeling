"""
Plotting Utilities
===================

This module centralizes the plotting routines used throughout the
predictive occupancy modeling project.  Each function accepts a
Pandas DataFrame and a path or file name where the resulting figure
should be saved.  The goal is to produce consistent, well‑labelled
visualizations that aid in understanding the occupancy patterns and
model behaviour.

The plots loosely mirror those from the original proof‑of‑concept
scripts but are generalized so that they can be applied to any
appropriately formatted dataset.
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(df: pd.DataFrame, occ_col: str, out_path: str) -> None:
    """Plot the hourly average occupancy over time.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe containing at least a datetime index or a ``time``
        column and the occupancy column.
    occ_col : str
        Name of the occupancy column to plot.
    out_path : str
        File path where the figure will be saved.  The directory
        structure will be created if necessary.
    """
    hourly = df.set_index("time").resample("1h").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly[occ_col].plot(ax=ax)
    ax.set_title("Hourly Average Occupancy Over Time (Raw)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Occupancy")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_histogram(df: pd.DataFrame, occ_col: str, out_path: str) -> None:
    """Plot a histogram of the occupancy distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df[occ_col].hist(bins=30, ax=ax)
    ax.set_title("Occupancy Distribution (Raw)")
    ax.set_xlabel("Occupancy")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_average_by_hour(df: pd.DataFrame, occ_col: str, out_path: str) -> None:
    """Plot the average occupancy by hour of day."""
    df = df.copy()
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    hourly_mean = df.groupby("hour")[occ_col].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    hourly_mean.plot(ax=ax)
    ax.set_title("Average Occupancy by Hour of Day (Raw)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Occupancy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_weekday_vs_weekend(df: pd.DataFrame, occ_col: str, out_path: str) -> None:
    """Compare average hourly occupancy for weekdays and weekends."""
    df = df.copy()
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["is_weekend"] = df["dow"] >= 5
    weekday = df[~df["is_weekend"]]
    weekend = df[df["is_weekend"]]
    weekday_hourly = weekday.groupby("hour")[occ_col].mean()
    weekend_hourly = weekend.groupby("hour")[occ_col].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    weekday_hourly.plot(ax=ax, label="Weekday (Mon–Fri)", linewidth=2)
    weekend_hourly.plot(ax=ax, label="Weekend (Sat–Sun)", linewidth=2, linestyle="--")
    ax.set_title("Weekday vs Weekend Occupancy Profile (Raw)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Occupancy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_probability_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Plot a heatmap of occupancy probability by day and hour.

    Expects the DataFrame to contain ``dow`` (day of week) and
    ``hour_int`` (integer hour) columns, along with a binary occupancy
    column ``is_occupied_int``.
    """
    pivot_prob = df.groupby(["dow", "hour_int"])["is_occupied_int"].mean().unstack()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_prob.index = days
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_prob, cmap="RdYlGn_r", annot=False, fmt=".1f", vmin=0, vmax=1, ax=ax)
    ax.set_title("Probability of Occupancy (Binary, Cleaned)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_raw_vs_clean(
    df: pd.DataFrame,
    raw_col: str,
    clean_col: str,
    out_path: str,
) -> None:
    """Overlay the raw and cleaned occupancy signals for comparison."""
    hourly_raw = df.set_index("time").resample("1h")[raw_col].mean()
    hourly_clean = df.set_index("time").resample("1h")[clean_col].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly_raw.index, hourly_raw, label="Raw Sensor Data", color="gray", alpha=0.4)
    ax.plot(hourly_clean.index, hourly_clean, label="Cleaned Model Input", color="green", linewidth=1.5)
    ax.set_title("Sensor Noise vs. Cleaned Model Input")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average People Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)