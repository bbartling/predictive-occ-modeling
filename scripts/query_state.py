"""
Utility script to query the current occupancy state from the REST API.

This script sends a GET request to the `/state` endpoint of a running
predictive occupancy server.  You can optionally pass an ISO 8601
timestamp via the `--at` argument to evaluate the state at a
specific moment.  If omitted, the server uses its own current time.

Example usage:

```bash
python query_state.py --url http://localhost:8000/state
python query_state.py --url http://localhost:8000/state --at 2025-11-22T15:45:00Z
```
"""

import argparse
import json
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Query occupancy state via REST API")
    parser.add_argument("--url", default="http://localhost:8000/state", help="Endpoint URL for the state API")
    parser.add_argument("--at", default=None, help="ISO 8601 timestamp (UTC) to evaluate. Defaults to now.")
    args = parser.parse_args()

    params = {}
    if args.at:
        params["at"] = args.at

    try:
        response = requests.get(args.url, params=params)
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