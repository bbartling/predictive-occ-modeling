# predictive-occ-modeling

Concept idea to predictive people occupancy for use in HVAC building automation industry. These are supervised binary classification models—specifically a machine-learning Random Forest and simple statistical baselines—that analyze historical time patterns to predict a "True/False" occupancy state.

1) Baseline Threshold: Simple average > threshold.
2) Probability Model: Historical frequency of "occupied" status.
3) Random Forest: A decision tree approach that learns time-of-day and day-of-week patterns.

The `all_occupancy_data.csv` file contains the full raw dataset, representing the net occupancy coming from three separate people-counter devices inside a building. Because it’s real field data, it includes some counting errors, and you’ll notice occasional negative values in the evenings when the counters lose sync. The `std_week.csv` file is a cleaned, summarized version created by taking the median value for each time slot across the entire dataset. By using medians instead of raw counts, the typical-week profile naturally filters out the negative errors and produces a stable “canonical week” that better reflects the true occupancy pattern.


## Requirements

Install Python Packages
```bash
pip install pandas matplotlib scikit-learn
```

## 🚀 How to Run

### Step 1: Clean & Analyze

This script cleans the raw data, applies the deadband filter, and visualizes the results.

```bash
python occupancy_clean_and_analyze.py
```

**Output Log:**

```text
Loading data from: C:\Users\ben\Documents\predictive-occ-modeling\all_occupancy_data.csv
---- BASIC INFO ----
Start: 2024-08-22 00:00:00+00:00
End: 2024-11-21 17:45:00+00:00
Rows: 8508
Unique days: 91

Occupancy describe:
count    8508.000000
mean        5.538434
std         7.188722
min        -9.000000
25%         0.000000
50%         3.000000
75%         9.000000
max        32.000000
Name: occ, dtype: float64

Fraction of zero occupancy: 0.22167371885284437
Charts saved to: C:\Users\ben\Documents\predictive-occ-modeling\charts
```
Fraction of zero occupancy reveals that the sensors reported exactly 0 people 22% of the time in the dataset and 78% of time the sensors reported > 0 people (even just 1 person).

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

### Step 2: Train & Predict

This script trains the models, compares them, and outputs the "Winner."

1) The baseline approach assumes a stationary weekly periodicity in building usage. It constructs a predictive model by projecting historical time-series data onto a discretized weekly domain, where the binary occupancy state for any given interval $t$ is determined by thresholding the ensemble average of all historical observations at $t$. I.E., creating a grid of "Time Buckets" where the model looks at every 'Monday at 8:00 AM' in your history and calculates the average crowd size for that specific time slot.

2) The probability model refines the baseline approach by converting raw counts into binary states before aggregation, shifting the analytical focus from crowd magnitude to occupancy frequency. Instead of averaging the number of people, it calculates the empirical likelihood that a specific 'Time Bucket' (e.g., Mondays at 8:00 AM) will be active based on historical consistency. This provides a robust enhancement by mitigating the skewing effect of sporadic high-volume events, ensuring the generated schedule reflects the reliability of presence rather than the density of the crowd.

3) The Random Forest classifier advances the modeling strategy by employing an ensemble of decision trees to capture non-linear interactions between temporal features, such as the specific conditional logic that distinguishes a 'Tuesday Morning' from a 'Saturday Night.' Unlike the baseline approach which relies on global averaging, this method recursively partitions the data into granular segments based on Day-of-Week and Minute-of-Day, allowing it to learn complex, localized boundaries for occupancy. This results in a robust binary schedule that mitigates overfitting by aggregating the 'votes' of hundreds of independent trees to determine the most probable state for any given timestamp.


Run command:  
```bash
python occupancy_model_binary.py
```

**Output Log:**

```text
Loading data...
Rows: 8508, days: 91
Building baseline model...
Building binary schedule (occupied/unoccupied)...

Reference time: 2024-11-21 17:45:00+00:00 (UTC)
Predicted current state: OCCUPIED
Next predicted transition: END in 975 minutes at 2024-11-22 10:00:00+00:00 (UTC)

RMSE vs std_week profile: 9.755

Preparing train/test split for model comparison...
Train samples: 5700, Test samples: 2808

--- Model Comparison on Held-out Test Data ---
baseline_threshold: accuracy = 0.745
probability_model: accuracy = 0.762
random_forest: accuracy = 0.770

Winner: random_forest (accuracy = 0.770)
Model comparison plot saved to C:\Users\ben\Documents\predictive-occ-modeling\charts\model_comparison_accuracy.png
PS C:\Users\ben\Documents\predictive-occ-modeling> 
```

---


## 🏆 Model Comparison Plot

![Model Plots](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/model_comparison_accuracy.png)

---


## 📂 `predicted_schedule.csv` output file

This file serves as the **Master Schedule Matrix** generated by the model. It converts the predictive analysis into a static lookup table that can be easily ingested by a BAS controller (JACE, Supervisor, or Edge Device).

### **Data Structure**

  * **Rows (`hour`):** Time of day in **Decimal Hours**.
      * *Example:* `0.0` = Midnight, `8.5` = 8:30 AM, `14.75` = 2:45 PM.
  * **Columns (`0` - `6`):** Days of the week.
      * `0` = Monday, `6` = Sunday.
  * **Values:** The Binary Control Command.
      * `0`: **Unoccupied** (Setback / System Off).
      * `1`: **Occupied** (Comfort Mode / System On).

### **Integration Example (Pseudo-code)**

To use this in a controller, simply lookup the value based on current time:

```python
# Current Timestamp: Monday at 2:30 PM
current_day = 0      # Monday
current_hour = 14.5  # 14:30

# Lookup in CSV Matrix
status = schedule_matrix[current_hour][current_day]

if status == 1:
    enable_hvac_comfort_mode()
else:
    enable_energy_savings_mode()
```

### **⚠️ Important Note**

If this file shows `1` (Occupied) at odd hours (e.g., Midnight), it was likely generated before the **"Deadband Filter"** was applied.

> **Recommendation:** Use **`final_predicted_schedule_binary.csv`** instead. It contains the same structure but uses the improved logic (filtering ghost counts and transients) to prevent equipment short-cycling.


## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.