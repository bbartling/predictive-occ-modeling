# IoT POC App

## Docker

The simplest way to run the API on an edge device is inside a
container.  A `Dockerfile` lives alongside `pyproject.toml` in the
project root.  **Be sure to execute the following commands from the
directory containing `Dockerfile` and `pyproject.toml`**—otherwise
Docker will not find the Python project.

Build:

```bash

docker stop predictive-occ-api
docker rm predictive-occ-api
docker build -t predictive-occ-api .
```

Run:
```bash

docker run --name predictive-occ-api \
  --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  predictive-occ-api
```

*Visit the Swagger API docs:*
➡️ **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

Restart:
```bash
docker restart predictive-occ-api
```

Prune:
```bash
docker system prune -a --volumes
```


---

## Testing scripts
Start the API (either via Docker or locally as described above).  To
exercise the endpoints without writing your own `curl` commands,
the repository provides three helper scripts in the `scripts/`
directory:

* **`train_model.py`** – sends a POST request to `/train` with a JSON
  payload pointing to a CSV on disk.  For example, after starting
  the server you can train using the sample data with:

  ```bash
  python scripts/train_model.py --data data/occupancy_sample.csv
  ```

  This will return a JSON summary of the training job.

* **`query_state.py`** – calls `/state` and prints the current
  occupancy state and next transition.  Optionally pass `--at
  YYYY-MM-DDTHH:MM:SSZ` to evaluate at a specific timestamp:

  ```bash
  python scripts/query_state.py
  python scripts/query_state.py --at 2025-11-22T15:45:00Z
  ```

* **`get_schedule.py`** – calls `/schedule` and writes the returned
  matrix to a CSV file.  For instance:

  ```bash
  python scripts/get_schedule.py --output my_schedule.csv
  ```

The scripts use the `requests` library to call the API.  If it
is not already installed as part of your environment, install it
via `pip install requests`.  They default to `http://localhost:8000` but
accept a `--url` argument if you run the server on a different host
or port.

## Configuring deadband and probability threshold

Two parameters govern how the model interprets your data:

* **Deadband (`deadband`)** – used during cleaning.  Raw counts
  less than or equal to the deadband are considered unoccupied and
  set to zero.  Higher counts are considered occupied.  For
  instance, with `deadband = 1.0` both 0 and 1 count as unoccupied,
  while counts ≥ 2 count as occupied.

* **Probability threshold (`prob_threshold`)** – used when
  constructing the final schedule.  After the historical probability
  of occupancy is computed for each slot, any probability greater
  than or equal to this threshold results in the slot being marked
  occupied.  Lower probabilities result in an unoccupied slot.

Adjust these values in the training request to tune sensitivity and
occupancy aggressiveness for your application.


## API Reference

### `POST /train`

Train a probability‑based occupancy model from a CSV file.  The
request body must be JSON conforming to the following schema:

```json
{
  "data_path": "data/occupancy_sample.csv",
  "time_col": "time",          // optional – default "time"
  "occ_col": "occ",            // optional – default "occ"
  "tz": "UTC",                 // optional – default "UTC"
  "deadband": 1.0,             // optional – default 1.0
  "prob_threshold": 0.5        // optional – default 0.5
}
```

On success the API returns a summary including the number of records
processed, the inferred sampling interval in minutes and p‑values
from the Augmented Dickey–Fuller (ADF) and KPSS stationarity tests【502360774470064†L199-L212】【502360774470064†L243-L253】.
If the CSV cannot be loaded or parsed a 400 response is returned.

### `GET /state`

Retrieve the current occupancy state and next transition.  An
optional `at` query parameter accepts an ISO 8601 timestamp in UTC
(`YYYY‑MM‑DDThh:mm:ssZ`) to evaluate the state at a particular
moment.  If omitted the server uses its current time.  The
response contains:

* `reference_time` – the time the state was evaluated (UTC).  
* `current_state` – either `OCCUPIED` or `UNOCCUPIED`.  
* `probability` – the historical probability of occupancy for the
  current time slot.  
* `next_transition` – an object describing when the state will next
  change, including the transition type (`START` or `END`), the
  number of minutes until it occurs and the UTC timestamp of the
  transition.

If no model has been trained yet the endpoint returns a 400 error.

### `GET /schedule`

Return the full weekly schedule as a JSON payload.  The response
includes the slot length (`step_minutes`), a list of fractional hours
(`hours`) corresponding to the row index and a nested list (`schedule`)
whose inner lists contain seven integers (one per day of week,
0 = Monday) indicating occupancy (1) or vacancy (0).  This matrix is
equivalent to the CSV exported in the original CLI implementation
and can be used by a BAS to perform simple schedule lookups.

## Training a model (example)

Suppose you have a 15‑minute occupancy dataset at `data/occupancy_sample.csv`.
To train a model and view its summary using curl:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/occupancy_sample.csv", "deadband": 1.0, "prob_threshold": 0.5}'
```

The response might look like this:

```json
{
  "message": "Model trained successfully",
  "records": 20160,
  "step_minutes": 15,
  "adf_p": 0.0000,
  "adf_stationary": true,
  "kpss_p": 0.1000,
  "kpss_stationary": true
}
```

This indicates that the occupancy signal appears stationary based on
both tests and that the model uses 15‑minute slots.  Internally the
probability of occupancy is calculated for each of the 96 slots in a
day and for each day of the week, and the binary schedule is
constructed by comparing those probabilities to the `prob_threshold`.

## Querying occupancy and next transition

After training you can query the current state.  By default the
server uses its own clock:

```bash
curl http://localhost:8000/state
```

The JSON response contains the reference time, whether the zone is
currently occupied and details of the next state change.  For
example:

```json
{
  "reference_time": "2025-11-22T15:30:00+00:00",
  "current_state": "OCCUPIED",
  "probability": 0.72,
  "next_transition": {
    "type": "END",
    "in_minutes": 90,
    "at": "2025-11-22T17:00:00+00:00"
  }
}
```

This means that at 15:30 the model predicts occupancy and expects it
to end 90 minutes later (at 17:00 UTC).  You can also supply a
specific timestamp via `?at=YYYY-MM-DDTHH:MM:SSZ` to evaluate the
state in the past or future.

## Retrieving the full schedule

To fetch the complete schedule matrix:

```bash
curl http://localhost:8000/schedule
```

The response includes the slot size and the matrix.  You can
convert this JSON into a CSV if required.  Each row corresponds to
a time of day (fractional hours) and each column corresponds to a
day of week (0 = Monday, 6 = Sunday).  A value of 1 means the zone
is predicted occupied at that time; 0 means unoccupied.


