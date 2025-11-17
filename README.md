# predictive-occ-modeling

Concept idea to predictive people occupancy for use in HVAC building automation industry.

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

Open the PNG files with any image viewer to inspect the occupancy patterns.

---

## Occupancy Predictor using Pandas, NumPy, and RMSE-based accuracy checks.

Run Script
```bash
python occupancy_model.py
```

The script will:
- Learn a baseline occupied/unoccupied schedule from historical data (day-of-week × time-of-day).
- Predict the current occupancy state (OCC/UNOCC) for a given reference time.
- Compute how many minutes until the next predicted occupancy **start** or **end** and log the transition time.
- Optionally compare the learned profile to `std_week.csv` and print an RMSE “distance” to your standard week.


## Future?
TODO: Optional ML upgrade for smarter occupancy forecasting.

---

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.