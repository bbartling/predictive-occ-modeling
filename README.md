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

See Proof-Of-Concept [README_IOT](https://github.com/bbartling/predictive-occ-modeling/blob/develop/README_iot.md)

---

## 📜 License

Everything here is **MIT Licensed** — free, open source, and made for the BAS community.  
Use it, remix it, or improve it — just share it forward so others can benefit too. 🥰🌍


【MIT License】

Copyright 2025 Ben Bartling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.