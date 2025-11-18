# predictive-occ-modeling

Concept idea to predictive people occupancy for use in HVAC building automation industry. These are supervised binary classification models—specifically a machine-learning Random Forest and simple statistical baselines—that analyze historical time patterns to predict a "True/False" occupancy state.

1) Baseline Threshold: Simple average > threshold.
2) Probability Model: Historical frequency of "occupied" status.
3) Random Forest: A decision tree approach that learns time-of-day and day-of-week patterns.

The `all_occupancy_data.csv` file contains the full raw dataset, representing the net occupancy coming from three separate people-counter devices inside a building. Because it’s real field data, it includes some counting errors, and you’ll notice occasional negative values in the evenings when the counters lose sync. The `std_week.csv` file is a cleaned, summarized version created by taking the median value for each time slot across the entire dataset. By using medians instead of raw counts, the typical-week profile naturally filters out the negative errors and produces a stable “canonical week” that better reflects the true occupancy pattern.


## Requirements

Install Python packages
```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

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
  Sensor Noise vs. Cleaned Signal: This chart overlays the raw sensor data (gray) with our processed model input (green). It visually demonstrates how the "Deadband Filter" successfully suppresses low-level sensor noise ("ghost counts"), revealing the true occupancy profile without the false positives that cause energy waste.

  ![Hourly Average Occupancy Over Time](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/raw_vs_clean_timeseries.png)

- **Average Occupancy by Hour of Day**  
  Aggregates all days together to show a “typical day” profile, useful for building a baseline schedule or optimal start logic.

  ![Average Occupancy by Hour of Day](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/avg_occupancy_by_hour.png)

- **Weekday vs Weekend Occupancy Profile**  
  Compares average hourly occupancy for weekdays versus weekends, revealing differences in schedule, peak times, and potential savings from different control strategies.

  ![Weekday vs Weekend Occupancy Profile](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/weekday_vs_weekend_occupancy.png)

- **Occupancy Distribution**  
  Histogram of all occupancy values, showing how often the space is empty, lightly used, or heavily occupied. This is helpful for threshold selection (e.g., occupied vs unoccupied) and for spotting outliers.

  ![Occupancy Distribution](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/occupancy_histogram.png)

- **Weekday Summary Heatmap Distribution**  
  The heatmap describes the most occupied days and hours, visualizing the patterns very clearly between hours and weekedays.

  ![Occupancy Distribution](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/occupancy_probability_heatmap.png)

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

## final\_predicted\_schedule.csv Matrix Lookup Table

The `final_predicted_schedule.csv` is a static schedule generated by the Probability Model. It represents the likelihood of occupancy for every 15-minute interval of the week, converted into a binary decision using a 50% confidence threshold.

  * **Rows (Index):** Time of day in decimal hours (e.g., `0.0` is Midnight, `14.5` is 2:30 PM).
  * **Columns (Header):** Day of the week, where `0` = Monday and `6` = Sunday.
  * **Values:** `1` for Occupied, `0` for Unoccupied.

### Example Data

```csv
hour,0,1,2,3,4,5,6
0.0,0,1,1,1,1,0,1
0.25,1,1,1,1,1,0,0
0.5,0,1,1,1,1,0,0
0.75,1,1,1,1,1,0,1
...
```

### BAS Integration Concept

This matrix functions as a lightweight lookup table for a Building Automation System (BAS). The controller rounds the current time to the nearest 15-minute increment and checks the corresponding value in the table to determine the HVAC mode.

**Pseudocode:**

```lua
// 1. INPUTS
// Round current time to nearest quarter-hour (e.g., 8:05 -> 8.0)
Current_Day_Index = Get_Day_Of_Week()   // 0=Mon, 1=Tue ... 6=Sun
Current_Decimal_Hour = Round_To_Quarter_Hour(Get_Time_Now()) 

// 2. LOOKUP
// Find value at intersection of Hour (Row) and Day (Col)
Modeled_State = CSV_Table.GetValue(Row=Current_Decimal_Hour, Column=Current_Day_Index)

// 3. CONTROL LOGIC
// If the model says 1, run comfort mode. If 0, run setback.
IF Modeled_State == 1 THEN
    HVAC_Command = OCCUPIED
ELSE
    HVAC_Command = UNOCCUPIED
END IF
```

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.