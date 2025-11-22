"""
Command‑line Entry Point
========================

This script orchestrates the entire predictive occupancy pipeline.  It
can be invoked from the command line and will:

1. Load a time‑series dataset from CSV.
2. Perform stationarity tests on the occupancy signal using both
   Augmented Dickey–Fuller (ADF) and KPSS methods.
3. Clean the raw occupancy counts using a deadband filter.
4. Generate exploratory plots and save them into a user‑specified
   directory.
5. Train baseline models on a training split of the data and
   evaluate their accuracy on a held‑out test set.
6. Export a schedule lookup table (matrix) based on the
   probability baseline so it can be consumed by a Building
   Automation System (BAS).

The pipeline is configurable via command‑line arguments.  For
example, to process the included sample dataset and write outputs
into the ``plots/`` and ``data/`` directories, run:

.. code-block:: bash

    python -m src.main --data data/occupancy_sample.csv --plots_dir plots --schedule_path data/final_schedule.csv

This script emphasizes reproducibility and transparency.  All
intermediate results and diagnostic metrics are written to standard
output so that users can verify the behaviour of the models on their
own datasets.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_processing import load_occupancy_data, clean_occupancy_counts
from .stationarity import adf_stationary, kpss_stationary
from .baseline_models import (
    build_mean_pivot,
    flatten_mean_pivot,
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


def run_pipeline(
    data_path: str,
    plots_dir: str,
    schedule_path: str,
    deadband: float = 1.0,
    prob_threshold: float = 0.5,
    time_col: str = "time",
    occ_col: str = "occ",
    tz: str = "UTC",
) -> None:
    """Execute the predictive occupancy pipeline on a given dataset.

    This function orchestrates the end‑to‑end modelling process.  It
    loads the dataset, tests for stationarity, cleans the occupancy
    counts, generates plots, trains and evaluates baseline models and
    exports a final schedule.  Parameters controlling the names of
    columns and the timezone allow it to operate on a wide variety
    of CSV files.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the raw time‑series data.
    plots_dir : str
        Directory where figures will be saved.  It will be created
        if it does not already exist.
    schedule_path : str
        Path to write the final BAS schedule (CSV format).
    deadband : float, default 1.0
        Threshold applied during cleaning.  Counts less than or
        equal to this value are treated as unoccupied.
    prob_threshold : float, default 0.5
        Decision threshold used when converting occupancy
        probabilities into binary values for the exported schedule.
    time_col : str, default "time"
        Name of the timestamp column in the input CSV.
    occ_col : str, default "occ"
        Name of the occupancy count column in the input CSV.
    tz : str, default "UTC"
        Timezone used to localise the timestamps.  Only relevant
        when ``parse_dates`` is True in :func:`load_occupancy_data`.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    print("Loading data from:", data_path)
    df = load_occupancy_data(
        data_path,
        time_col=time_col,
        occ_col=occ_col,
        parse_dates=True,
        tz=tz,
        rename_columns=True,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2. Stationarity tests on raw occupancy counts
    print("\n---- STATIONARITY TESTS ----")
    adf_stat, adf_p, _ = adf_stationary(df["occ"].values)
    print(f"ADF Test: p-value = {adf_p:.4f} → {'Stationary' if adf_stat else 'Non-stationary'}")
    kpss_stat, kpss_p, _ = kpss_stationary(df["occ"].values)
    print(f"KPSS Test: p-value = {kpss_p:.4f} → {'Stationary' if kpss_stat else 'Non-stationary'}")

    if adf_stat and kpss_stat:
        print("The occupancy series appears to be stationary based on both tests.")
    else:
        print(
            "Warning: The occupancy series may not be strictly stationary.\n"
            "Baseline models will still be trained, but consider differencing or detrending for more advanced models."
        )

    # ------------------------------------------------------------------
    # 3. Cleaning
    # The ``clean_occupancy_counts`` function works on the canonical
    # ``occ`` column created by ``load_occupancy_data``.  If you set
    # ``rename_columns`` to False then you must pass the original
    # occupancy column name here.
    df_clean = clean_occupancy_counts(df, occ_col="occ", deadband=deadband, make_binary=True)
    df_clean["hour_int"] = df_clean["time"].dt.hour

    # ------------------------------------------------------------------
    # 4. Plotting
    print("\nGenerating plots...")
    os.makedirs(plots_dir, exist_ok=True)
    plot_time_series(df_clean, occ_col="occ", out_path=os.path.join(plots_dir, "hourly_occupancy_timeseries.png"))
    plot_histogram(df_clean, occ_col="occ", out_path=os.path.join(plots_dir, "occupancy_histogram.png"))
    plot_average_by_hour(df_clean, occ_col="occ", out_path=os.path.join(plots_dir, "avg_occupancy_by_hour.png"))
    plot_weekday_vs_weekend(df_clean, occ_col="occ", out_path=os.path.join(plots_dir, "weekday_vs_weekend_occupancy.png"))
    # Heatmap uses binary cleaned occupancy
    plot_probability_heatmap(df_clean, out_path=os.path.join(plots_dir, "occupancy_probability_heatmap.png"))
    # Raw vs clean comparison
    plot_raw_vs_clean(df_clean, raw_col="occ", clean_col="clean_count", out_path=os.path.join(plots_dir, "raw_vs_clean_timeseries.png"))
    print(f"All plots saved to: {plots_dir}")

    # ------------------------------------------------------------------
    # 5. Train/test split for baseline evaluation
    step_minutes: int
    pivot_full, step_minutes = build_mean_pivot(df_clean, occ_col="clean_count", time_col="time", step_minutes=None)
    # Use full data to build probability table for final schedule
    prob_table_full = build_probability_table(df_clean, occ_threshold=0.0, step_minutes=step_minutes, occ_col="clean_count")

    # Prepare labelled dataset for evaluation (binary y)
    df_labelled = df_clean.copy()
    df_labelled["slot"] = (df_labelled["minute_of_day"] // step_minutes) * step_minutes
    df_labelled["y"] = (df_labelled["clean_count"] > 0.0).astype(int)

    # Train/test split (70/30)
    split_idx = int(len(df_labelled) * 0.70)
    df_train = df_labelled.iloc[:split_idx].copy()
    df_test = df_labelled.iloc[split_idx:].copy()

    # Build mean and probability tables on the training set
    pivot_train, _ = build_mean_pivot(df_train, occ_col="clean_count", time_col="time", step_minutes=step_minutes)
    mean_table_train = flatten_mean_pivot(pivot_train)
    prob_table_train = build_probability_table(df_train, occ_threshold=0.0, step_minutes=step_minutes, occ_col="clean_count")

    print("\n---- BASELINE TRAINING & COMPARISON ----")
    results = evaluate_baseline_models(df_test, mean_table_train, prob_table_train, occ_threshold=deadband, prob_threshold=prob_threshold)
    print(results)

    # Plot bar chart of accuracies
    fig, ax = plt.subplots()
    ax.bar(results["model"], results["accuracy"], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline Model Accuracy (No ML)")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "baseline_model_accuracy.png"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Export final schedule
    print("\n---- EXPORTING FINAL SCHEDULE ----")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(schedule_path), exist_ok=True)
    matrix = export_probability_schedule(
        prob_table_full, output_path=schedule_path, prob_threshold=prob_threshold, step_minutes=step_minutes
    )
    print(f"Final schedule saved to: {schedule_path}")

    # Optional: compare the mean profile to a standard week profile if available
    std_path = os.path.join(os.path.dirname(data_path), "std_week.csv")
    if os.path.exists(std_path):
        std_week = pd.read_csv(std_path)
        from .baseline_models import compare_to_std  # imported lazily to avoid circular import at top
        rmse = compare_to_std(pivot_full, std_week)
        print(f"RMSE vs std_week profile: {rmse:.3f}")

    # Print a friendly message about the underlying question
    print("\nWhat is the probability that a zone is occupied or unoccupied in the near term future?")
    print(
        "This pipeline estimates that probability using historical patterns.  "
        "The exported lookup table provides a simple way for a BAS to answer that question at runtime."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predictive occupancy modelling pipeline.  This tool loads an "
            "occupancy time‑series, tests for stationarity, cleans and "
            "visualises the data, trains baseline models and exports a "
            "schedule suitable for Building Automation Systems."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "occupancy_sample.csv"),
        help=(
            "Path to the input CSV containing occupancy data.  This argument "
            "can be overridden via a configuration file."
        ),
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Directory to save generated plots."
    )
    parser.add_argument(
        "--schedule_path",
        type=str,
        default=os.path.join("data", "final_schedule.csv"),
        help="Path to save the final BAS schedule CSV."
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=1.0,
        help=(
            "Deadband threshold for cleaning occupancy counts.  Counts less "
            "than or equal to this value are treated as unoccupied."
        ),
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for converting probabilities into binary decisions."
        ),
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="time",
        help=(
            "Name of the timestamp column in the input CSV.  This is only "
            "used if no configuration file is supplied."
        ),
    )
    parser.add_argument(
        "--occ_col",
        type=str,
        default="occ",
        help=(
            "Name of the occupancy column in the input CSV.  This is only "
            "used if no configuration file is supplied."
        ),
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="UTC",
        help="Timezone for localising timestamps (e.g. 'America/Chicago').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON configuration file.  Values in the "
            "configuration will override command‑line defaults for "
            "data_path, time_col, occ_col, tz, deadband, prob_threshold, "
            "plots_dir and schedule_path."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load configuration file if specified.  Values in the config
    # override the corresponding command‑line arguments.  Missing
    # keys are ignored.  This allows a JSON document to serve as
    # a lightweight dataset descriptor for bespoke CSV files.
    cfg = {}
    if args.config:
        from .config import load_config
        cfg = load_config(args.config)
        if cfg:
            print(f"Loaded configuration from {args.config}")
        else:
            print(f"Warning: failed to load configuration from {args.config}, falling back to defaults")

    # Determine effective parameters, falling back to CLI defaults
    data_path = cfg.get("data_path", args.data)
    plots_dir = cfg.get("plots_dir", args.plots_dir)
    schedule_path = cfg.get("schedule_path", args.schedule_path)
    deadband = cfg.get("deadband", args.deadband)
    prob_threshold = cfg.get("prob_threshold", args.prob_threshold)
    time_col = cfg.get("time_col", args.time_col)
    occ_col = cfg.get("occ_col", args.occ_col)
    tz = cfg.get("tz", args.tz)

    run_pipeline(
        data_path=data_path,
        plots_dir=plots_dir,
        schedule_path=schedule_path,
        deadband=deadband,
        prob_threshold=prob_threshold,
        time_col=time_col,
        occ_col=occ_col,
        tz=tz,
    )


if __name__ == "__main__":  # pragma: no cover
    main()