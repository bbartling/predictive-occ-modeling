import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple

# ---------------------------------------------------------
# Core model helpers
# ---------------------------------------------------------

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
        # if something weird, default to 0
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
    Find minutes until next change in occupancy (start or end), from given timestamp.

    Returns:
        (minutes_ahead, transition_type, transition_timestamp)
        where transition_type is "start" (0 -> 1) or "end" (1 -> 0),
        or None if no transition found within max_days_ahead.
    """
    ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")

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
# Optional: std_week comparison
# ---------------------------------------------------------

def load_std_week(base: str) -> pd.DataFrame:
    path = os.path.join(base, "std_week.csv")
    return pd.read_csv(path)


def compare_to_std(pivot: pd.DataFrame, std: pd.DataFrame) -> float:
    """
    Compare live model (pivot) to std_week median profile using RMSE.
    We collapse days by averaging across columns, then compare against std.
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

# ---------------------------------------------------------
# Main: log-only demo
# ---------------------------------------------------------

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print("Loading data...")
    df = load_data(base)
    print(f"Rows: {len(df)}, days: {df['time'].dt.date.nunique()}")

    print("Building baseline model...")
    pivot, step_minutes = build_baseline_model(df)

    print("Building binary schedule (occupied/unoccupied)...")
    schedule = predict_schedule(pivot, threshold=2.0)

    # "Now" can be any timestamp; for demo use the last timestamp in dataset
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

    # Optional: you *could* add a quick plot here (heatmap of schedule),
    # but for now we just log to console per your preference.


if __name__ == "__main__":
    main()
