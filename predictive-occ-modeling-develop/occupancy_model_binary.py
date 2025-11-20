import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROB_THRESHOLD = 0.5  # threshold for deciding occupied from probability model
OCC_DEADBAND = 1.0    # counts <= this are treated as unoccupied in baseline

# =========================================================
# 1. DATA LOADING & HELPERS
# =========================================================

def load_data(base: str) -> pd.DataFrame:
    """Load occupancy data and add time fields."""
    df = pd.read_csv(os.path.join(base, "all_occupancy_data.csv"))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    df["dow"] = df["time"].dt.dayofweek         # 0 = Monday
    df["minute_of_day"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    return df


def infer_step_minutes(df: pd.DataFrame) -> int:
    """Infer sampling step in minutes (e.g. 15 min)."""
    diffs = df["time"].sort_values().diff().dropna()
    if diffs.empty:
        return 15
    step = int(diffs.mode().iloc[0].total_seconds() // 60)
    return max(step, 1)


# =========================================================
# 2. BASELINE MODEL & LIVE QUERY
# =========================================================

def build_baseline_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Build baseline model (Mean Pivot Table) on *raw counts*.

    Returns
    -------
    pivot : DataFrame
        index = slot (minute_of_day bucket)
        columns = dow (0..6)
        values = mean occ
    step_minutes : int
        sampling interval in minutes
    """
    step_minutes = infer_step_minutes(df)
    df = df.copy()
    df["slot"] = (df["minute_of_day"] // step_minutes) * step_minutes

    grouped = df.groupby(["dow", "slot"])["occ"].mean().reset_index()
    pivot = grouped.pivot(index="slot", columns="dow", values="occ")

    # Ensure full 0..1440 window
    all_slots = np.arange(0, 24 * 60, step_minutes)
    pivot = pivot.reindex(all_slots)
    pivot = pivot.interpolate(limit_direction="both")
    return pivot, step_minutes


def predict_schedule(pivot: pd.DataFrame, threshold: float = OCC_DEADBAND) -> pd.DataFrame:
    """
    Convert mean-count pivot to binary schedule (0/1) using a deadband threshold.
    """
    schedule = (pivot > threshold).astype(int)
    return schedule


def _slot_from_timestamp(ts: pd.Timestamp, step_minutes: int) -> Tuple[int, int]:
    """Return (dow, slot_minutes) for a given timestamp."""
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    dow = ts.dayofweek
    minute_of_day = ts.hour * 60 + ts.minute
    slot = (minute_of_day // step_minutes) * step_minutes
    return dow, slot


def get_predicted_state_at(timestamp: pd.Timestamp,
                           schedule: pd.DataFrame,
                           step_minutes: int) -> int:
    """
    Return predicted occupancy state (0/1) at a specific timestamp
    using the mean-count baseline schedule.
    """
    dow, slot = _slot_from_timestamp(timestamp, step_minutes)
    # Snap to nearest slot if needed
    if slot not in schedule.index:
        slot = int(schedule.index[(schedule.index - slot).abs().argmin()])
    if dow not in schedule.columns:
        return 0
    return int(schedule.loc[slot, dow])


def minutes_to_next_transition(
    timestamp: pd.Timestamp,
    schedule: pd.DataFrame,
    step_minutes: int,
    max_days_ahead: int = 7
) -> Optional[Tuple[float, str, pd.Timestamp]]:
    """
    Find minutes until next start/stop event in the baseline schedule.
    """
    ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else ts.tz_localize("UTC")
    current_state = get_predicted_state_at(ts, schedule, step_minutes)
    max_steps = int((max_days_ahead * 24 * 60) / step_minutes)

    for step in range(1, max_steps + 1):
        future_ts = ts + pd.Timedelta(minutes=step * step_minutes)
        future_state = get_predicted_state_at(future_ts, schedule, step_minutes)
        if future_state != current_state:
            transition_type = "start" if future_state == 1 else "end"
            return step * step_minutes, transition_type, future_ts
    return None


# =========================================================
# 3. STD WEEK COMPARISON
# =========================================================

def load_std_week(base: str) -> pd.DataFrame:
    path = os.path.join(base, "std_week.csv")
    return pd.read_csv(path)


def compare_to_std(pivot: pd.DataFrame, std: pd.DataFrame) -> float:
    """Compare live model to std_week profile using RMSE."""
    model_profile = pivot.mean(axis=1)
    std_interp = np.interp(
        model_profile.index,
        np.linspace(0, 24 * 60, len(std)),
        std["median_occ"],
    )
    return np.sqrt(((model_profile.values - std_interp) ** 2).mean())


# =========================================================
# 4. PROBABILITY BASELINE (NO ML)
# =========================================================

def add_slot_and_label(df: pd.DataFrame,
                       occ_threshold: float,
                       step_minutes: int) -> pd.DataFrame:
    """
    Add 'slot' and binary label y based on raw occ > occ_threshold.
    """
    df = df.copy()
    df["slot"] = (df["minute_of_day"] // step_minutes) * step_minutes
    df["y"] = (df["occ"] > occ_threshold).astype(int)
    return df


def train_mean_model(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Mean model on continuous occ (used only for comparison).
    """
    return (
        df_train
        .groupby(["dow", "slot"])["occ"]
        .mean()
        .reset_index()
        .rename(columns={"occ": "mean_occ"})
    )


def train_prob_model(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Probability model on binary y (this is the main "Generic Week" schedule).
    """
    return (
        df_train
        .groupby(["dow", "slot"])["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "p_occ"})
    )


def evaluate_baselines(df_test: pd.DataFrame,
                       mean_table: pd.DataFrame,
                       prob_table: pd.DataFrame,
                       occ_threshold: float = OCC_DEADBAND,
                       prob_thresh: float = PROB_THRESHOLD) -> pd.DataFrame:
    """
    Compare:
      - threshold on mean_occ (continuous baseline)
      - probability model p_occ (main generic-week schedule)
    """
    merged = (
        df_test
        .merge(mean_table, on=["dow", "slot"], how="left")
        .merge(prob_table, on=["dow", "slot"], how="left")
    )

    y_true = merged["y"].astype(int)

    # Mean-count threshold baseline
    y_pred_mean = (merged["mean_occ"] > occ_threshold).astype(int).fillna(0)

    # Probability baseline
    y_pred_prob = (merged["p_occ"] > prob_thresh).astype(int).fillna(0)

    acc_mean = (y_pred_mean == y_true).mean()
    acc_prob = (y_pred_prob == y_true).mean()

    return pd.DataFrame({
        "model": ["baseline_threshold", "probability_model"],
        "accuracy": [acc_mean, acc_prob],
    })


# =========================================================
# 5. EXPORT UTILITY (PROBABILITY MODEL ONLY)
# =========================================================

def export_probability_schedule(prob_table: pd.DataFrame,
                                output_path: str,
                                prob_threshold: float = PROB_THRESHOLD):
    """
    Exports the probability model as the Final Master Schedule CSV.

    - Uses p_occ > prob_threshold → decision (0/1)
    - Pivots into matrix: rows = hour, cols = dow (0..6)
    """
    prob_table = prob_table.copy()
    prob_table["decision"] = (prob_table["p_occ"] > prob_threshold).astype(int)
    prob_table["hour"] = prob_table["slot"] / 60.0

    matrix = (
        prob_table
        .pivot(index="hour", columns="dow", values="decision")
        .fillna(0)
        .astype(int)
    )
    matrix.to_csv(output_path)
    return matrix


# =========================================================
# MAIN
# =========================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    occ_threshold = OCC_DEADBAND  # deadband for "occupied" based on raw count

    print("Loading data...")
    df = load_data(base)

    # --- 1. LIVE DASHBOARD DEMO (Using Mean Baseline) ---
    print("\n--- 1. LIVE DASHBOARD DEMO (Baseline) ---")
    pivot, step_minutes = build_baseline_model(df)
    schedule = predict_schedule(pivot, threshold=occ_threshold)

    now = df["time"].iloc[-1]  # Use last timestamp in file as "Now"
    state_str = "OCCUPIED" if get_predicted_state_at(now, schedule, step_minutes) == 1 else "UNOCCUPIED"
    print(f"Reference time: {now} (UTC)")
    print(f"Current State: {state_str}")

    res = minutes_to_next_transition(now, schedule, step_minutes)
    if res:
        mins, trans, ts = res
        print(f"Next Transition: {trans.upper()} in {mins:.0f} mins at {ts} (UTC)")

    # --- 2. Std Week Comparison (Optional) ---
    std_path = os.path.join(base, "std_week.csv")
    if os.path.exists(std_path):
        rmse = compare_to_std(pivot, load_std_week(base))
        print(f"RMSE vs std_week: {rmse:.3f}")

    # --- 3. Probability Baseline Training & Comparison ---
    print("\n--- 2. BASELINE TRAINING & COMPARISON (NO ML) ---")
    df_labeled = add_slot_and_label(df, occ_threshold, step_minutes)
    split_idx = int(len(df_labeled) * 0.70)
    df_train = df_labeled.iloc[:split_idx].copy()
    df_test = df_labeled.iloc[split_idx:].copy()

    print(f"Training rows: {len(df_train)}, Test rows: {len(df_test)}")

    mean_tbl = train_mean_model(df_train)
    prob_tbl = train_prob_model(df_train)

    results = evaluate_baselines(df_test, mean_tbl, prob_tbl,
                                 occ_threshold=occ_threshold,
                                 prob_thresh=PROB_THRESHOLD)
    print(results)

    # Simple bar chart for documentation / QA
    charts_dir = os.path.join(base, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plt.figure()
    plt.bar(results["model"], results["accuracy"], color=["#1f77b4", "#ff7f0e"])
    plt.ylim(0, 1.0)
    plt.title("Baseline Model Accuracy (No ML)")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(charts_dir, "baseline_model_accuracy.png"))
    plt.close()

    # --- 4. EXPORTING MASTER SCHEDULE (PROBABILITY MODEL ONLY) ---
    print("\n--- 3. EXPORTING MASTER SCHEDULE (Probability Model) ---")
    out_path = os.path.join(base, "final_predicted_schedule.csv")
    export_probability_schedule(prob_tbl, out_path, prob_threshold=PROB_THRESHOLD)
    print(f" Schedule saved to: {out_path}")
    print(f" Using probability baseline with threshold={PROB_THRESHOLD}")

if __name__ == "__main__":
    main()
