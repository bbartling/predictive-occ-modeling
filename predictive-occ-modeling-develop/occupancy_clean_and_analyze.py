import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
    Occupancy Analysis Script.
    
    1. Loads and cleans data (Deadband logic: counts <= 1 are 0).
    2. Generates standard EDA plots (Weekday vs Weekend, etc.).
    3. Generates Model-ready plots (Heatmap, Raw vs Clean).
    4. Saves everything to 'charts/'.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "all_occupancy_data.csv")
    charts_dir = os.path.join(base_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    # ---------------------------------------------------------
    # 1. FEATURE ENGINEERING & CLEANING
    # ---------------------------------------------------------
    df["date"] = df["time"].dt.date
    # Decimal hour for plotting (e.g. 9.5 = 9:30 AM)
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    # Integer hour for Heatmap grouping (e.g. 9)
    df["hour_int"] = df["time"].dt.hour 
    df["dow"] = df["time"].dt.dayofweek  # 0 = Monday
    df["is_weekend"] = df["dow"] >= 5

    # --- The "Deadband" Logic ---
    # Step A: Fix sensor errors (negatives -> 0)
    df["clean_count"] = df["occ"].apply(lambda x: max(0, x))
    
    # Step B: Apply Deadband (Counts of 1.0 or less become 0)
    df["clean_count"] = df["clean_count"].apply(lambda x: 0 if x <= 1.0 else x)

    # Step C: Create Binary Target (True if > 0 after cleaning)
    df["is_occupied"] = df["clean_count"] > 0
    df["is_occupied_int"] = df["is_occupied"].astype(int)

    # ---------------------------------------------------------
    # 2. PRINT BASIC STATS (No ROI)
    # ---------------------------------------------------------
    print("---- BASIC INFO ----")
    print("Start:", df["time"].min())
    print("End:", df["time"].max())
    print("Rows:", len(df))
    print("Unique days:", df["date"].nunique())
    print()
    print("Occupancy describe (Raw):")
    print(df["occ"].describe())
    print()
    print("Fraction of zero occupancy (Raw):", (df["occ"] == 0).mean())

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    
    # --- Plot 1: Hourly Average Occupancy (Raw) ---
    hourly = df.set_index("time").resample("1h").mean(numeric_only=True)
    
    plt.figure(figsize=(10, 5))
    hourly["occ"].plot()
    plt.title("Hourly Average Occupancy Over Time (Raw)")
    plt.xlabel("Time")
    plt.ylabel("Average Occupancy")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "hourly_occupancy_timeseries.png"))
    plt.close()

    # --- Plot 2: Histogram (Raw) ---
    plt.figure(figsize=(8, 5))
    df["occ"].hist(bins=30)
    plt.title("Occupancy Distribution (Raw)")
    plt.xlabel("Occupancy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "occupancy_histogram.png"))
    plt.close()

    # --- Plot 3: Average Occupancy by Hour of Day (Raw) ---
    hourly_mean = df.groupby("hour")["occ"].mean()
    plt.figure(figsize=(8, 5))
    hourly_mean.plot()
    plt.title("Average Occupancy by Hour of Day (Raw)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Occupancy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_occupancy_by_hour.png"))
    plt.close()

    # --- Plot 4: Weekday vs Weekend Profile (Raw) ---
    # Separating data
    weekday = df[~df["is_weekend"]]
    weekend = df[df["is_weekend"]]

    # Grouping by decimal hour
    weekday_hourly = weekday.groupby("hour")["occ"].mean()
    weekend_hourly = weekend.groupby("hour")["occ"].mean()

    plt.figure(figsize=(10, 6))
    weekday_hourly.plot(label="Weekday (Mon-Fri)", linewidth=2)
    weekend_hourly.plot(label="Weekend (Sat-Sun)", linewidth=2, linestyle="--")
    plt.title("Weekday vs Weekend Occupancy Profile (Raw)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Occupancy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "weekday_vs_weekend_occupancy.png"))
    plt.close()

    # --- Plot 5: Heatmap of Probability (Cleaned/Binary) ---
    # Group by Day of Week and Integer Hour
    pivot_prob = df.groupby(["dow", "hour_int"])["is_occupied_int"].mean().unstack()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_prob.index = days

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_prob, cmap="RdYlGn_r", annot=False, fmt=".1f", vmin=0, vmax=1)
    plt.title("Probability of Occupancy (Binary, Cleaned)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "occupancy_probability_heatmap.png"))
    plt.close()

    # --- Plot 6: Raw vs Cleaned Time Series (Comparison) ---
    plt.figure(figsize=(12, 6))
    # Raw Data (Faded)
    plt.plot(hourly.index, hourly["occ"], label="Raw Sensor Data", color="gray", alpha=0.4)
    # Cleaned Data (Sharp)
    hourly_clean = df.set_index("time").resample("1h")["clean_count"].mean()
    plt.plot(hourly_clean.index, hourly_clean, label="Cleaned Model Input (Deadband Applied)", color="green", linewidth=1.5)
    
    plt.title("Sensor Noise vs. Cleaned Model Input")
    plt.xlabel("Date")
    plt.ylabel("Avg People Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "raw_vs_clean_timeseries.png"))
    plt.close()

    print(f"All charts saved to: {charts_dir}")

if __name__ == "__main__":
    main()