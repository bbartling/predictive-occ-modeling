"""
Utility script to train the predictive occupancy model via the REST API.

Run this script with Python to send a JSON payload to the `/train`
endpoint of a running predictive‑occupancy server.  The script
expects the server to be listening on http://localhost:8000 by
default.  Adjust the URL and JSON parameters as necessary.

Example usage:

```bash
python train_model.py --data data/occupancy_sample.csv --deadband 1.0 --prob_threshold 0.5
```
"""

import argparse
import json
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the predictive occupancy model via REST API")
    parser.add_argument("--url", default="http://localhost:8000/train", help="Endpoint URL for the train API")
    parser.add_argument("--data", required=True, help="Path to the CSV file accessible to the server")
    parser.add_argument("--time_col", default="time", help="Name of the timestamp column in the CSV")
    parser.add_argument("--occ_col", default="occ", help="Name of the occupancy count column in the CSV")
    parser.add_argument("--tz", default="UTC", help="Timezone to interpret timestamps")
    parser.add_argument("--deadband", type=float, default=1.0, help="Deadband threshold for cleaning")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Probability threshold for schedule generation")
    args = parser.parse_args()

    payload = {
        "data_path": args.data,
        "time_col": args.time_col,
        "occ_col": args.occ_col,
        "tz": args.tz,
        "deadband": args.deadband,
        "prob_threshold": args.prob_threshold,
    }

    try:
        response = requests.post(args.url, json=payload)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error contacting {args.url}: {exc}")
        return
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


if __name__ == "__main__":
    main()