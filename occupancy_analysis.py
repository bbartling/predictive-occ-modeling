import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Simple occupancy EDA script.

    - Reads `all_occupancy_data.csv` in the same folder.
    - Parses timestamps.
    - Adds hour-of-day, day-of-week, weekend flags.
    - Saves charts into the `charts/` directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "all_occupancy_data.csv")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Parse timestamps (works fine on Windows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    # Feature engineering
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["dow"] = df["time"].dt.dayofweek  # 0 = Monday
    df["is_weekend"] = df["dow"] >= 5

    print("---- BASIC INFO ----")
    print("Start:", df["time"].min())
    print("End:", df["time"].max())
    print("Rows:", len(df))
    print("Unique days:", df["date"].nunique())
    print()
    print("Occupancy describe:")
    print(df["occ"].describe())
    print()
    print("Fraction of zero occupancy:", (df["occ"] == 0).mean())

    # Make charts directory
    charts_dir = os.path.join(base_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1) Hourly average occupancy over time
    hourly = df.set_index("time").resample("1H").mean(numeric_only=True)
    plt.figure()
    hourly["occ"].plot()
    plt.title("Hourly Average Occupancy Over Time")
    plt.xlabel("Time")
    plt.ylabel("Average Occupancy")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "hourly_occupancy_timeseries.png"))
    plt.close()

    # 2) Occupancy histogram
    plt.figure()
    df["occ"].hist(bins=30)
    plt.title("Occupancy Distribution")
    plt.xlabel("Occupancy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "occupancy_histogram.png"))
    plt.close()

    # 3) Average occupancy by hour of day
    hourly_mean = df.groupby("hour")["occ"].mean()
    plt.figure()
    hourly_mean.plot()
    plt.title("Average Occupancy by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Occupancy")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_occupancy_by_hour.png"))
    plt.close()

    # 4) Weekday vs weekend occupancy profiles
    weekday = df[~df["is_weekend"]]
    weekend = df[df["is_weekend"]]

    weekday_hourly = weekday.groupby("hour")["occ"].mean()
    weekend_hourly = weekend.groupby("hour")["occ"].mean()

    plt.figure()
    weekday_hourly.plot(label="Weekday")
    weekend_hourly.plot(label="Weekend")
    plt.title("Weekday vs Weekend Occupancy Profile")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Occupancy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "weekday_vs_weekend_occupancy.png"))
    plt.close()

    print(f"Charts saved to: {charts_dir}")

if __name__ == "__main__":
    main()
