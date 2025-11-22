# Predictive Occupancy Modeling

![Insert Snip Here](https://github.com/bbartling/predictive-occ-modeling/blob/develop/BAS_Of_The_Future.png)

> **What is the probability that an HVAC zone is occupied or unoccupied in the near term future?**


This repository provides a framework for evaluating and modeling **occupancy patterns in Building Automation Systems (BAS)** using historical occupancy-sensor time series. The purpose is to make it easy to ingest BAS occupancy data, assess whether the time series is sufficiently stationary using statistics, build lightweight baseline models, evaluate their performance, and ultimately export a predicted-schedule lookup table that a BAS can consume. In short, this repo is designed to **test BAS occupancy sensor data and prototype concepts for future BAS integration.**


> The repository contains two datasets: one with occupancy data from people-counting sensors and another with hourly weather data. These datasets are unrelated and are used for stationary-data tests for learning purposes, demonstrating that datasets like weather— which are naturally non-stationary — cannot be used in the modeling process.

## Why stationarity matters

A stationary time series has a constant mean, variance and
autocorrelation over time.  Many forecasting
techniques assume that the data are stationary; if they are not,
models can produce unreliable results.  To assess stationarity we
employ two complementary tests:

* **Augmented Dickey–Fuller (ADF) test:** the null hypothesis is
  that the series contains a unit root (i.e. it is non‑stationary).
  A small p‑value allows us to reject non‑stationarity and conclude
  that the series is stationary.
* **Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test:** the null
  hypothesis is that the series is stationary around a deterministic
  trend.  A large p‑value (greater than 0.05) indicates stationarity
  while a small p‑value suggests non‑stationarity.

If both tests agree that the occupancy signal is stationary then a
simple probability‑of‑occupancy model often outperforms more complex
techniques.  Stationary signals are common in buildings with highly
repetitive weekly occupancy patterns, and in such cases the
probability model is robust to sensor noise because it uses the
historical frequency of occupancy rather than raw counts.

## Notebooks

The [notebooks](https://github.com/bbartling/predictive-occ-modeling/tree/develop/notebooks) directory contains several example Jupyter notebooks that demonstrate key concepts such as stationarity testing, baseline modeling, and lag analysis. Some of the datasets used in these notebooks are Kaggle-sourced CSV files related to occupancy and thermal comfort. The notebooks incorporate both the occupancy dataset and a weather dataset to highlight the differences between stationary and non-stationary signals.

## Model Used

This project uses a probability-based, time-slot occupancy model built directly from historical data. Instead of training a machine-learning algorithm, the method divides the week into discrete time buckets (such as 15-minute intervals across a seven-day cycle) and computes the empirical probability that each bucket is occupied. For every time slot, the system examines how often the space was occupied in past weeks, converts these frequencies into a probability between zero and one, and applies a configurable threshold (`prob_threshold`) to generate a final binary occupied or unoccupied schedule. Because the model is non-parametric, fully explainable, and based entirely on observed frequencies, it remains extremely stable, resilient to noisy sensor inputs, and lightweight enough to run on constrained Building Automation Systems or edge devices. From a statistical perspective, the method estimates the empirical distribution of occupancy over repeating weekly cycles, producing a reliable schedule whenever the underlying time-series data ***is stationary***.


---


<details>
<summary>Python Setup, Docker, CLI and Configuration for IoT Edge - TODO</summary>

* NOT DONE

### Installation

This project is packaged as a standard Python module using a
``pyproject.toml``.  Clone the repository and install it in editable
mode so that changes are reflected without re‑installation:

```bash
git clone <this‑repo>
cd predictive-occ-modeling-develop/predictive-occ-modeling-develop
pip install .
```

Dependencies are declared in ``pyproject.toml`` and include
``pandas``, ``numpy``, ``matplotlib``, ``seaborn`` and
``statsmodels``.

After installation a command‑line script named
``predictive-occ-modeling`` becomes available.  You can invoke it
from anywhere without modifying your ``PYTHONPATH``.

### Configuration file

Datasets often differ in the names of their timestamp and occupancy
columns or need different cleaning thresholds.  To avoid passing
numerous flags each time you run the pipeline, you can supply a
JSON configuration file via the ``--config`` option.  Keys in the
file override CLI defaults.  A minimal example looks like this:

```json
{
  "data_path": "data/occupancy_sample.csv",
  "time_col": "time",
  "occ_col": "occ",
  "tz": "UTC",
  "deadband": 1.0,
  "prob_threshold": 0.5,
  "plots_dir": "plots",
  "schedule_path": "data/final_schedule.csv"
}
```

Save the JSON as ``config.json`` (or any filename) and run:

```bash
predictive-occ-modeling --config config.json
```

If no configuration file is provided you can still pass
options on the command line.  The most commonly used arguments are:

* ``--data`` – path to your CSV file (default ``data/occupancy_sample.csv``)
* ``--time_col`` – name of the timestamp column (default ``time``)
* ``--occ_col`` – name of the occupancy counts column (default ``occ``)
* ``--tz`` – timezone for localising timestamps (default ``UTC``)
* ``--deadband`` – cleaning threshold below which counts are treated as unoccupied
* ``--prob_threshold`` – threshold for converting probabilities into a binary schedule
* ``--plots_dir`` – directory to save generated figures ***Also leave key blank to skip plotting***
* ``--schedule_path`` – where to write the final schedule lookup table

### Usage

To process the bundled sample dataset without a config file:

Mac/Linux

```bash
predictive-occ-modeling \
  --data data/occupancy_sample.csv \
  --plots_dir plots \
  --schedule_path data/final_schedule.csv \
  --deadband 1.0 \
  --prob_threshold 0.5
```

PowerShell

```powershell
predictive-occ-modeling `
  --data "data/occupancy_sample.csv" `
  --plots_dir "plots" `
  --schedule_path "data/final_schedule.csv" `
  --deadband 1.0 `
  --prob_threshold 0.5

```

```bash
predictive-occ-modeling --config my_dataset_config.json
```

On Windows (PowerShell) the syntax is analogous, replacing line
continuation characters as appropriate.

The script performs the following steps:

1. **Load and inspect the data.** It prints the range of timestamps
   and conducts Augmented Dickey–Fuller (ADF) and KPSS stationarity
   tests on the raw occupancy signal.
2. **Clean the data.** Negative counts are clipped to zero and
   counts less than or equal to ``deadband`` are set to zero.  A
   binary occupancy signal is derived for modelling.
3. **Generate plots.** Figures are saved into ``plots_dir`` including
   a time‑series overview, histogram, average hourly profile,
   weekday/weekend comparison, probability heatmap and a raw vs
   cleaned overlay.
4. **Train baseline models.** A simple threshold baseline and a
   probability baseline are trained on a 70 % split and tested on
   the remaining 30 %.  Accuracies are reported in the console and
   saved as a bar chart.
5. **Export a schedule.** The probability baseline trained on the
   full dataset produces a lookup table (matrix) saved at
   ``schedule_path``.  Rows correspond to fractional hours and
   columns to days of the week (0=Monday).  A value of 1 means
   occupied; 0 means unoccupied.

At the end of the run you will see output summarising the model
training and schedule generation.  For example:

```text
Loading data from: data/occupancy_sample.csv

---- STATIONARITY TESTS ----
ADF Test: p-value = 0.0000 → Stationary
KPSS Test: p-value = 0.1000 → Stationary
The occupancy series appears to be stationary based on both tests.

Generating plots...
All plots saved to: plots

---- BASELINE TRAINING & COMPARISON ----
                model  accuracy
0  baseline_threshold  0.759499
1   probability_model  0.752840

---- EXPORTING FINAL SCHEDULE ----
Final schedule saved to: data/final_schedule.csv
RMSE vs std_week profile: 9.715

What is the probability that a zone is occupied or unoccupied in the near term future?
This pipeline estimates that probability using historical patterns.  The exported lookup table provides a simple way for a BAS to answer that question at runtime.
```

### Deadband versus Probability Threshold

* ``deadband`` (data cleaning) – If the raw count is ≤ ``deadband``
  it is treated as zero occupancy; counts above the threshold are
  treated as occupied.  For example, with ``deadband = 1.0`` counts
  of 0 and 1 are mapped to unoccupied, while counts ≥ 2 are mapped
  to occupied.
* ``prob_threshold`` (schedule generation) – After computing the
  historical probability of occupancy for each slot, the
  ``prob_threshold`` converts that probability into a final binary
  schedule.  Probabilities ≥ ``prob_threshold`` become 1 (occupied);
  otherwise 0.

### Docker support

If you wish to run the pipeline or the notebooks in an isolated
container, a ``Dockerfile`` is provided.  Build and run the image
like so:

```bash
# Build the image
docker build -t predictive-occ-modeling .
# Run the command‑line interface (prints help by default)
docker run --rm predictive-occ-modeling --help
```

When launched with Jupyter, navigate to the ``notebooks`` directory
to explore the example analyses.  The CLI script remains available
inside the container as the ``predictive-occ-modeling`` command.

</details>
---

<details>
<summary>Conceptual idea of extending the framework - TODO</summary>

To predict occupancy on live building control systems, operations technology (OT) such as **Building Automation Systems (BAS)** typically does not natively support data modeling or complex analysis libraries like Python, Pandas, NumPy, and certainly not machine-learning frameworks. Historically, these modeling capabilities have been provided by specialized Smart Building IoT platforms. However, some vendors—such as Delta Controls with their RED5 lineup—are beginning to blur this boundary and bring more advanced analytics capability directly into the BAS layer.

> The ultimate goal is to use the current computer time, compare it to the modeled probability of the HVAC zone becoming occupied or unoccupied, and determine the next transition—including whether the zone will switch to OCC or UNOCC and the number of minutes until that change occurs—similar to the output shown below:

```text
Reference time: 2024-11-21 17:45:00+00:00 (UTC)
Current State: OCCUPIED
Next Transition: END in 975 mins at 2024-11-22 10:00:00+00:00 (UTC)
```

Some modern BAS platforms ***could be capable*** of **parsing and ingesting CSV files**. Our `final_predicted_schedule.csv` leverages this capability; it is a static **Matrix Lookup Table** that a BAS platform can read. By comparing the current time to the table's structure, the system can instantly retrieve a **modeled predictive occupancy value** (0 or 1), which can then be used to control HVAC scheduling and optimal start algorithms.

The output file itself is based on the **Probability Model**. This model converts the historical likelihood of occupancy for every 15-minute interval into a binary decision. This conversion uses a **50% confidence threshold** (set by the variable `prob_threshold=0.5`). This threshold can be easily adjusted within the `occupancy_model_binary.py` script (e.g., to `0.85`) if a more conservative (stricter) control strategy is required.

  * **Rows (Index):** Time of day in decimal hours (e.g., `0.0` is Midnight, `14.5` is 2:30 PM).
  * **Columns (Header):** Day of the week, where `0` = Monday and `6` = Sunday.
  * **Values:** `1` for Occupied, `0` for Unoccupied.

### Example Matrix Lookup Table in CSV format

```csv
hour,0,1,2,3,4,5,6
0.0,0,1,1,1,1,0,1
0.25,1,1,1,1,1,0,0
0.5,0,1,1,1,1,0,0
0.75,1,1,1,1,1,0,1
...
```

This matrix functions as a lightweight lookup table for a live Building Automation System (BAS) or IoT if it can read and parse the CSV file. The **BAS controller then has to check rounded current time to the nearest 15-minute increment**. The value at that intersection (the 0 or 1) is the ***Predicted Occupancy State*** for that specific 15-minute slot. We can then focus purely on the Future Look-Ahead logic for optimal start algorithms. (This is also open for debate or other use cases too!)

### Example BAS Scripting Pseudocode for HVAC sequencing

> What is the probablitiy that the building will be occupied in 3 hours from now? Do we need to warm up or cool down the building??!

```lua
TIME_OFFSET_HOURS = 3

PREDICTED_STATE = Check_Static_Schedule(Current_Time + TIME_OFFSET_HOURS)

IF PREDICTED_STATE == OCCUPIED THEN
    Run_Optimal_Start_Routine()
ELSE
    Maintain_Night_Setback()
END IF
```



</details>


---

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.