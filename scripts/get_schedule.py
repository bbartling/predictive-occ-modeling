"""
Utility script to fetch the full occupancy schedule and write it to CSV.

This script sends a GET request to the `/schedule` endpoint of a running
predictive occupancy server, then writes the returned matrix to a CSV
file.  The first column contains the fractional hours (e.g. 0.0,
0.25, …) and the next seven columns correspond to days of the week
(0=Monday, 6=Sunday).

Example usage:

```bash
python get_schedule.py --url http://localhost:8000/schedule --output predicted_schedule.csv
```
"""

import argparse
import csv
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch occupancy schedule via REST API")
    parser.add_argument("--url", default="http://localhost:8000/schedule", help="Endpoint URL for the schedule API")
    parser.add_argument("--output", default="predicted_schedule.csv", help="Path to write the schedule CSV")
    args = parser.parse_args()

    try:
        response = requests.get(args.url)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error contacting {args.url}: {exc}")
        return
    data = response.json()
    hours = data.get("hours")
    schedule = data.get("schedule")
    if hours is None or schedule is None:
        print("Invalid schedule response")
        return
    # Write to CSV
    header = ["hour"] + [str(i) for i in range(7)]
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for hour, row in zip(hours, schedule):
            writer.writerow([hour] + row)
    print(f"Schedule saved to {args.output}")


if __name__ == "__main__":
    main()