# predictive-occ-modeling

Concept idea to predictive people occupancy for use in HVAC building automation industry.

The `all_occupancy_data.csv` file contains the full raw dataset, representing the net occupancy coming from three separate people-counter devices. Because it’s real field data, it includes some counting errors, and you’ll notice occasional negative values in the evenings when the counters lose sync. The `std_week.csv` file is a cleaned, summarized version created by taking the median value for each time slot across the entire dataset. By using medians instead of raw counts, the typical-week profile naturally filters out the negative errors and produces a stable “canonical week” that better reflects the true occupancy pattern.


## Project Structure

```text
occupancy_project/
├── all_occupancy_data.csv   # Raw occupancy data (15-min intervals)
├── std_week.csv             # Reference "standard week" (not used in script yet)
├── occupancy_analysis.py    # Main analysis script
└── charts/                  # Output charts are saved here
```

## Requirements

Install Python Packages
```bash
pip install pandas matplotlib
```

Run Script
```bash
python occupancy_analysis.py
```

The script will:
- Print basic stats about the dataset.
- Create a `charts/` folder (if it doesn't exist yet).
- Save the following PNG files into `charts/`:
   - `hourly_occupancy_timeseries.png`
   - `occupancy_histogram.png`
   - `avg_occupancy_by_hour.png`
   - `weekday_vs_weekend_occupancy.png`

---

## 📊 Exploratory Occupancy Plots

This project generates several charts to help understand occupancy patterns in time and magnitude.

- **Hourly Average Occupancy Over Time**  
  Shows the hourly mean occupancy across the full dataset, highlighting overall usage patterns, gaps in data, and any long-term drift or seasonal effects.

  ![Hourly Average Occupancy Over Time](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/hourly_occupancy_timeseries.png)

- **Average Occupancy by Hour of Day**  
  Aggregates all days together to show a “typical day” profile, useful for building a baseline schedule or optimal start logic.

  ![Average Occupancy by Hour of Day](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/avg_occupancy_by_hour.png)

- **Weekday vs Weekend Occupancy Profile**  
  Compares average hourly occupancy for weekdays versus weekends, revealing differences in schedule, peak times, and potential savings from different control strategies.

  ![Weekday vs Weekend Occupancy Profile](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/weekday_vs_weekend_occupancy.png)

- **Occupancy Distribution**  
  Histogram of all occupancy values, showing how often the space is empty, lightly used, or heavily occupied. This is helpful for threshold selection (e.g., occupied vs unoccupied) and for spotting outliers.

  ![Occupancy Distribution](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/occupancy_histogram.png)

---

Here’s a tightened-up version you can drop into the README:


## Occupancy Predictor using Pandas, NumPy, and RMSE-based accuracy checks

Run the script:

```bash
python occupancy_model.py
```

The script will:

* Learn a baseline occupied/unoccupied schedule from historical data (day-of-week × time-of-day).
* Predict the current occupancy state (OCC/UNOCC) for a given reference time.
* Compute how many minutes remain until the next predicted occupancy **start** or **end** and log the transition time.
* Optionally compare the learned profile to `std_week.csv` and print an RMSE “distance” to your standard week.

Example output:

```bash
> python occupancy_model.py
Loading data...
Rows: 8508, days: 91
Building baseline model...
Building binary schedule (occupied/unoccupied)...

Reference time: 2024-11-21 17:45:00+00:00 (UTC)
Predicted current state: OCCUPIED
Next predicted transition: END in 900 minutes at 2024-11-22 08:45:00+00:00 (UTC)

RMSE vs std_week profile: 9.755
```

---

## Future?
1) Try in conjuction with an optimal start algorithm which require `Predicted current state` and `Next predicted transition` as inputs.
2) The current model predicts **occupied vs unoccupied** by taking the historical average occupancy for the same day-of-week and time-of-day and applying a simple threshold, producing a hard 0/1 result with no probability. Adding a probability layer—such as the historical fraction of times that slot was occupied—would give a confidence measure for each prediction, and you could compare which approach performs better by testing both methods against a held-out portion of the dataset and measuring accuracy or RMSE on true occupancy states. 
3) Probability methods could be started with lightweight historical-frequency probabilities, then move to simple logistic regression, and finally compare them to more advanced ML models like random forests or LSTMs to see which provides the most reliable occupancy confidence.


---

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.