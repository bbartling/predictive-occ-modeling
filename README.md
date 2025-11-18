# predictive-occ-modeling

A proof-of-concept for predictive occupancy modeling using historical people-counting data, designed for the HVAC and Building Automation industry. Demonstrated below is supervised binary classification models—specifically a machine-learning Random Forest and simple statistical baselines—that analyze historical time patterns to predict a "True/False" occupancy state.

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

1) The baseline approach assumes a stationary weekly periodicity in building usage. It constructs a predictive model by projecting historical time-series data onto a discretized weekly domain, where the binary occupancy state for any given interval $t$ is determined by thresholding the ensemble average of all historical observations at $t$. Essentially in simple terms, we are creating a generic weekly calendar and stacking all past data onto it, hour for hour, day for day. The model then looks at each resulting 'Time Bucket' to determine the average historical people counts for that exact time slot.

2) The probability model refines the baseline approach by converting raw counts into binary states before aggregation, shifting the analytical focus from people count magnitude to occupancy frequency. Instead of averaging the number of people, it calculates the empirical likelihood that a specific 'Time Bucket' (e.g., Mondays at 8:00 AM) will be active based on historical consistency. This provides a robust enhancement by mitigating the skewing effect of sporadic high-volume events, ensuring the generated schedule reflects the reliability of presence rather than the density of the crowd or people count.

3) The Random Forest classifier advances the modeling strategy by employing an ensemble of decision trees to capture non-linear interactions between temporal features, such as the specific conditional logic that distinguishes a 'Tuesday Morning' from a 'Saturday Night.' Unlike the baseline approach which relies on global averaging, this method recursively partitions the data into granular segments based on Day-of-Week and Minute-of-Day, allowing it to learn complex, localized boundaries for occupancy. This results in a robust binary schedule that mitigates overfitting by aggregating the 'votes' of hundreds of independent trees to determine the most probable state for any given timestamp.


Python run command:  
```bash
python occupancy_model_binary.py
```

**Output Log:**

```text
Loading data...

--- 1. LIVE DASHBOARD DEMO (Baseline) ---
Reference time: 2024-11-21 17:45:00+00:00 (UTC)
Current State: OCCUPIED
Next Transition: END in 975 mins at 2024-11-22 10:00:00+00:00 (UTC)
RMSE vs std_week: 9.755

--- 2. MODEL TRAINING & COMPARISON ---
Training (Rows: 5955)...
   > Tuning Random Forest (this may take a moment)...
   > Best RF Parameters: {'class_weight': 'balanced', 'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 200}
                model  accuracy
0  baseline_threshold  0.741481
1   probability_model  0.752840
2       random_forest  0.788876

WINNER: random_forest (0.789)

--- 3. EXPORTING MASTER SCHEDULE ---
Saved to: C:\Users\ben\Documents\predictive-occ-modeling\final_predicted_schedule.csv
```

---


## Model Comparison Plot

![Model Plots](https://github.com/bbartling/predictive-occ-modeling/raw/develop/charts/model_comparison_accuracy.png)

---

## Finalized Matrix Lookup Table for Building Automation

**Building Automation Systems (BAS) do not natively model data** or support complex analysis libraries like Python or machine learning frameworks. Historically, these data modeling processes have been handled by specialized Smart Building IoT platforms.

However, many modern BAS platforms are capable of **parsing and ingesting CSV files**. Our `final_predicted_schedule.csv` leverages this capability; it is a static **Matrix Lookup Table** that a BAS platform can read. By comparing the current time to the table's structure, the system can instantly retrieve a **modeled predictive occupancy value** (0 or 1), which can then be used to control HVAC scheduling and optimal start algorithms.

The "Winning model" determines the final schedule, but the output file itself is based on the **Probability Model**. This model converts the historical likelihood of occupancy for every 15-minute interval into a binary decision. This conversion uses a **50% confidence threshold** (set by the variable `prob_threshold=0.5`). This threshold can be easily adjusted within the `occupancy_model_binary.py` script (e.g., to `0.85`) if a more conservative (stricter) control strategy is required.

  * **Rows (Index):** Time of day in decimal hours (e.g., `0.0` is Midnight, `14.5` is 2:30 PM).
  * **Columns (Header):** Day of the week, where `0` = Monday and `6` = Sunday.
  * **Values:** `1` for Occupied, `0` for Unoccupied.

### Example data in CSV file

```csv
hour,0,1,2,3,4,5,6
0.0,0,1,1,1,1,0,1
0.25,1,1,1,1,1,0,0
0.5,0,1,1,1,1,0,0
0.75,1,1,1,1,1,0,1
...
```

### BAS Integration Concept

This matrix functions as a lightweight lookup table for a live Building Automation System (BAS) or IoT if it can read and parse the CSV file. The **BAS controller then has to check rounded current time to the nearest 15-minute increment**. The value at that intersection (the 0 or 1) is the ***Predicted Occupancy State*** for that specific 15-minute slot. We can then focus purely on the Future Look-Ahead logic for optimal start algorithms. (This is also open for debate or other use cases too!)

**Pseudocode:**

What is the probablitiy that the building will be occupied in 3 hours from now? Do we need to warm up or cool down the building??!

```lua
// GOAL: Check if people are coming soon so we can start warming up the building.

// 1. LOOK INTO THE FUTURE
// Pick a time ahead of now (e.g., 3 hours ahead) to see if we need to get ready.
Future_Check_Time = Current_Time + 3_Hours

// 2. ASK THE MODEL
// Look at the generated schedule and ask: "Will the building be occupied at that future time?"
Is_People_Coming = Get_Schedule_Status(Future_Check_Time)

// 3. FEED THE OPTIMAL START
// If the model says "Yes, people are coming," tell the Optimal Start logic to begin calculations.
IF Is_People_Coming == TRUE THEN
    // The building needs to be ready! Hand off control to your warmup algorithm.
    Run_Optimal_Start_Routine()
ELSE
    // No one is coming soon. Keep the system in sleep mode.
    Maintain_Night_Setback()
END IF
```

In Python we do this in the `occupancy_model_binary.py` is concept is ran once for demo purposes. 

```python
# --- A. Live Dashboard Demo (Using Baseline) ---
print("\n--- 1. LIVE DASHBOARD DEMO (Baseline) ---")

# 1. Build the schedule from history
pivot, step_minutes = build_baseline_model(df)
schedule = predict_schedule(pivot, threshold=occ_threshold)

# 2. Pick "Now" (The last row in your data)
now = df["time"].iloc[-1] 

# 3. Check status at "Now"
state_str = "OCCUPIED" if get_predicted_state_at(now, schedule, step_minutes) == 1 else "UNOCCUPIED"
print(f"Reference time: {now} (UTC)")
print(f"Current State: {state_str}")

# 4. Look into the future (Loop ahead until state changes)
res = minutes_to_next_transition(now, schedule, step_minutes)
if res:
    mins, trans, ts = res
    print(f"Next Transition: {trans.upper()} in {mins:.0f} mins at {ts} (UTC)")
```

The following logs demonstrate the live dashboard functionality, which could be continuously checked on a live system using a Python `while` loop:

```text
--- 1. LIVE DASHBOARD DEMO (Baseline) ---
Reference time: 2024-11-21 17:45:00+00:00 (UTC)
Current State: OCCUPIED
Next Transition: END in 975 mins at 2024-11-22 10:00:00+00:00 (UTC)
```

---

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.