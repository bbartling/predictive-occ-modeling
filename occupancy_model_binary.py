import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# Core model helpers (baseline + schedule features)
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


def build_baseline_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Build baseline model:
    - index: minute_of_day (0..1439, step ~15)
    - columns: day of week (0..6)
    - values: mean occupancy
    Returns (pivot, step_minutes).
    """
    step_minutes = infer_step_minutes(df)

    # Snap minutes to discrete bins (e.g. 0,15,30,...)
    df = df.copy()
    df["slot"] = (df["minute_of_day"] // step_minutes) * step_minutes

    grouped = df.groupby(["dow", "slot"])["occ"].mean().reset_index()
    pivot = grouped.pivot(index="slot", columns="dow", values="occ")

    # Ensure full 0..1440 window in given step
    all_slots = np.arange(0, 24 * 60, step_minutes)
    pivot = pivot.reindex(all_slots)
    pivot = pivot.interpolate(limit_direction="both")  # fill any gaps

    return pivot, step_minutes


def predict_schedule(pivot: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Convert occupancy pivot to a binary schedule.
    1 = occupied, 0 = unoccupied.
    Same shape as pivot (slot x dow).
    """
    schedule = (pivot > threshold).astype(int)
    return schedule


# ---------------------------------------------------------
# Query functions: state + minutes to next start/end
# ---------------------------------------------------------

def _slot_from_timestamp(ts: pd.Timestamp, step_minutes: int) -> Tuple[int, int]:
    """Return (dow, slot_minutes) for a given timestamp."""
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    dow = ts.dayofweek
    minute_of_day = ts.hour * 60 + ts.minute
    slot = (minute_of_day // step_minutes) * step_minutes
    return dow, slot


def get_predicted_state_at(
    timestamp: pd.Timestamp,
    schedule: pd.DataFrame,
    step_minutes: int
) -> int:
    """
    Return predicted occupancy state at a given timestamp:
    1 = occupied, 0 = unoccupied.
    """
    dow, slot = _slot_from_timestamp(timestamp, step_minutes)

    # Wrap slot into index range if needed
    if slot not in schedule.index:
        # snap to nearest index
        slot = int(schedule.index[(schedule.index - slot).abs().argmin()])

    if dow not in schedule.columns:
        return 0

    state = int(schedule.loc[slot, dow])
    return state


def minutes_to_next_transition(
    timestamp: pd.Timestamp,
    schedule: pd.DataFrame,
    step_minutes: int,
    max_days_ahead: int = 7
) -> Optional[Tuple[float, str, pd.Timestamp]]:
    """
    Find minutes until next change in occupancy (start or end).
    """
    ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else ts.tz_localize("UTC")

    current_state = get_predicted_state_at(ts, schedule, step_minutes)

    # We simulate forward in discrete steps
    max_steps = int((max_days_ahead * 24 * 60) / step_minutes)

    for step in range(1, max_steps + 1):
        future_ts = ts + pd.Timedelta(minutes=step * step_minutes)
        future_state = get_predicted_state_at(future_ts, schedule, step_minutes)

        if future_state != current_state:
            transition_type = "start" if future_state == 1 else "end"
            minutes_ahead = step * step_minutes
            return minutes_ahead, transition_type, future_ts

    return None


# ---------------------------------------------------------
# std_week comparison
# ---------------------------------------------------------

def load_std_week(base: str) -> pd.DataFrame:
    path = os.path.join(base, "std_week.csv")
    return pd.read_csv(path)


def compare_to_std(pivot: pd.DataFrame, std: pd.DataFrame) -> float:
    """
    Compare live model (pivot) to std_week median profile using RMSE.
    """
    model_profile = pivot.mean(axis=1)  # average across all days
    # Interpolate std to same index range (0..24h)
    std_interp = np.interp(
        model_profile.index,
        np.linspace(0, 24 * 60, len(std)),
        std["median_occ"]
    )
    rmse = np.sqrt(((model_profile.values - std_interp) ** 2).mean())
    return rmse


# =========================================================
# Classification labels and simple models
# =========================================================

def add_slot_and_label(df: pd.DataFrame, occ_threshold: float, step_minutes: int) -> pd.DataFrame:
    """
    Add discrete slot (minute_of_day bucket) and binary label y = occ > threshold.
    """
    df = df.copy()
    df["slot"] = (df["minute_of_day"] // step_minutes) * step_minutes
    df["y"] = (df["occ"] > occ_threshold).astype(int)
    return df


def train_mean_model(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Train the baseline mean-occupancy model.
    Returns a table with columns [dow, slot, mean_occ].
    """
    grouped = (
        df_train.groupby(["dow", "slot"])["occ"]
        .mean()
        .reset_index()
        .rename(columns={"occ": "mean_occ"})
    )
    return grouped


def train_prob_model(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Train the probability model based on historical frequency of occupancy (y = 1).
    Returns a table with columns [dow, slot, p_occ].
    """
    grouped = (
        df_train.groupby(["dow", "slot"])["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "p_occ"})
    )
    return grouped


# ---------------------------------------------------------
# Random Forest model (sklearn)
# ---------------------------------------------------------

def train_random_forest(df_train: pd.DataFrame):
    """
    Train a lightweight RandomForestClassifier on simple time features.
    Features: [dow, minute_of_day].
    """
    X = df_train[["dow", "minute_of_day"]].values
    y = df_train["y"].values
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced" # Added based on previous recommendation for imbalanced data
    )
    rf.fit(X, y)
    return rf


def predict_random_forest(model, df_test: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(df_test), dtype=int)
    X = df_test[["dow", "minute_of_day"]].values
    return model.predict(X)


# ---------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------

def evaluate_models(df_test: pd.DataFrame,
                    mean_table: pd.DataFrame,
                    prob_table: pd.DataFrame,
                    rf_model,
                    occ_threshold: float = 2.0,
                    prob_decision_threshold: float = 0.5) -> pd.DataFrame:
    """
    Evaluate the three approaches on held-out test data and return a results DataFrame.
    Models:
      - baseline_threshold: mean_occ > occ_threshold
      - probability_model: p_occ > prob_decision_threshold
      - random_forest: RF classifier on [dow, minute_of_day]
    """
    # Merge mean/prob tables
    merged = df_test.merge(mean_table, on=["dow", "slot"], how="left")
    merged = merged.merge(prob_table, on=["dow", "slot"], how="left")

    y_true = merged["y"]

    # Baseline mean-threshold model
    y_hat_mean = (merged["mean_occ"] > occ_threshold).astype(int).fillna(0).astype(int)
    acc_mean = (y_hat_mean == y_true).mean()

    # Probability model
    y_hat_prob = (merged["p_occ"] > prob_decision_threshold).astype(int).fillna(0).astype(int)
    acc_prob = (y_hat_prob == y_true).mean()

    # Random Forest
    y_hat_rf = predict_random_forest(rf_model, merged)
    acc_rf = (y_hat_rf == y_true).mean()

    results = pd.DataFrame({
        "model": [
            "baseline_threshold",
            "probability_model",
            "random_forest"
        ],
        "accuracy": [acc_mean, acc_prob, acc_rf],
    })
    return results


# =========================================================
# Main: log demo, model comparison, and simple plot
# =========================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    occ_threshold = 1.0 # Updated to 1.0 based on your "clean ghosts" strategy

    print("Loading data...")
    df = load_data(base)
    print(f"Rows: {len(df)}, days: {df['time'].dt.date.nunique()}")

    # Build baseline pivot for schedule-style queries and RMSE check
    print("Building baseline model...")
    pivot, step_minutes = build_baseline_model(df)

    print("Building binary schedule (occupied/unoccupied)...")
    schedule = predict_schedule(pivot, threshold=occ_threshold)

    # "Now" = last timestamp in dataset
    now = df["time"].iloc[-1]
    current_state = get_predicted_state_at(now, schedule, step_minutes)
    state_str = "OCCUPIED" if current_state == 1 else "UNOCCUPIED"
    print(f"\nReference time: {now} (UTC)")
    print(f"Predicted current state: {state_str}")

    result = minutes_to_next_transition(now, schedule, step_minutes, max_days_ahead=7)

    if result is None:
        print("No schedule transition found in the next 7 days.")
    else:
        minutes_ahead, transition_type, ts_transition = result
        print(
            f"Next predicted transition: {transition_type.upper()} "
            f"in {minutes_ahead:.0f} minutes at {ts_transition} (UTC)"
        )

    # Optional: if std_week.csv exists, print RMSE
    std_path = os.path.join(base, "std_week.csv")
    if os.path.exists(std_path):
        std = load_std_week(base)
        rmse = compare_to_std(pivot, std)
        print(f"\nRMSE vs std_week profile: {rmse:.3f}")

    # -------------------------------------------------
    # Train/test split for comparing classifiers
    # -------------------------------------------------
    print("\nPreparing train/test split for model comparison...")
    step_minutes_data = infer_step_minutes(df)
    df_labeled = add_slot_and_label(df, occ_threshold=occ_threshold, step_minutes=step_minutes_data)

    # Time-based split: first 2/3 train, last 1/3 test
    split_idx = int(len(df_labeled) * 0.67)
    df_train = df_labeled.iloc[:split_idx].copy()
    df_test = df_labeled.iloc[split_idx:].copy()

    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    # Train simple models
    mean_table = train_mean_model(df_train)
    prob_table = train_prob_model(df_train)
    rf_model = train_random_forest(df_train)

    # Evaluate three approaches (No LSTM)
    results = evaluate_models(
        df_test,
        mean_table,
        prob_table,
        rf_model,
        occ_threshold=occ_threshold,
        prob_decision_threshold=0.5,
    )

    print("\n--- Model Comparison on Held-out Test Data ---")
    for _, row in results.iterrows():
        print(f"{row['model']}: accuracy = {row['accuracy']:.3f}")

    best_row = results.iloc[results["accuracy"].idxmax()]
    print(f"\nWinner: {best_row['model']} (accuracy = {best_row['accuracy']:.3f})")

    # Simple bar chart of model accuracies
    charts_dir = os.path.join(base, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    plt.figure()
    plt.bar(results["model"], results["accuracy"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel("Accuracy")
    plt.title("Occupancy Model Comparison (Held-out Test Data)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "model_comparison_accuracy.png"))
    plt.close()

    print(f"Model comparison plot saved to {os.path.join(charts_dir, 'model_comparison_accuracy.png')}")


if __name__ == "__main__":
    main()