"""
FastAPI application for Predictive Occupancy Modeling.

This module defines a small FastAPI application that wraps the core
functionality provided by the ``predictive_occ`` library.  It
exposes the same REST interface as the original ``src/api.py``:

* **POST /train** – Train a model from a CSV file on disk.
* **GET /state** – Query the current occupancy state and next transition.
* **GET /schedule** – Retrieve the full weekly schedule matrix.

To run the API locally without Docker, install the project in editable
mode and execute:

    uvicorn api.app:app --host 0.0.0.0 --port 8000

The API is designed to run on lightweight edge devices or inside a
Docker container.  The core modelling code does not depend on FastAPI
and can be imported independently from notebooks or other scripts.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd

from predictive_occ.io import load_occupancy_data, clean_occupancy_counts
from predictive_occ.model import infer_step_minutes, build_probability_table, export_probability_schedule
from predictive_occ.stationarity import adf_stationary, kpss_stationary


class TrainRequest(BaseModel):
    """Request body for the /train endpoint."""
    data_path: str = Field(..., description="Path to the CSV file to train on")
    time_col: str = Field("time", description="Name of the timestamp column")
    occ_col: str = Field("occ", description="Name of the occupancy count column")
    tz: str = Field("UTC", description="Timezone for localising timestamps")
    deadband: float = Field(1.0, description="Deadband threshold for cleaning")
    prob_threshold: float = Field(0.5, description="Probability threshold for schedule generation")


class TransitionInfo(BaseModel):
    """Information about the next occupancy transition."""

    type: str = Field(..., description="START if transitioning to occupied, END if transitioning to unoccupied")
    in_minutes: int = Field(..., description="Number of minutes until the transition")
    at: str = Field(..., description="ISO 8601 timestamp when the transition occurs (UTC)")


class StateResponse(BaseModel):
    """Response body for the /state endpoint."""

    reference_time: str = Field(..., description="The time the state was evaluated (UTC)")
    current_state: str = Field(..., description="OCCUPIED or UNOCCUPIED")
    probability: float = Field(..., description="Historical probability of occupancy for the current time slot")
    next_transition: TransitionInfo


class ScheduleResponse(BaseModel):
    """Response body for the /schedule endpoint."""

    step_minutes: int = Field(..., description="The slot length in minutes")
    hours: List[float] = Field(..., description="Row index representing hours past midnight (fractional)")
    schedule: List[List[int]] = Field(..., description="Nested list of occupancy predictions for each day")


class ModelState:
    """Holds the trained model components in memory."""

    def __init__(
        self,
        step_minutes: int,
        prob_threshold: float,
        tz: str,
        prob_table: pd.DataFrame,
        schedule_matrix: pd.DataFrame,
        hour_index: List[float],
    ) -> None:
        self.step_minutes = step_minutes
        self.prob_threshold = prob_threshold
        self.tz = tz
        self.prob_table = prob_table
        self.schedule_matrix = schedule_matrix
        self.hour_index = hour_index

    def get_probability(self, dow: int, slot: int) -> float:
        rec = self.prob_table[(self.prob_table["dow"] == dow) & (self.prob_table["slot"] == slot)]
        if not rec.empty:
            return float(rec.iloc[0]["p_occ"])
        return 0.0

    def get_schedule_value(self, dow: int, slot: int) -> int:
        hour = slot / 60.0
        try:
            val = self.schedule_matrix.loc[hour, dow]
            return int(val)
        except Exception:
            return 0


app = FastAPI(
    title="Predictive Occupancy Modeling API",
    description=(
        "Train and query a probability‑based occupancy model from time‑series data. "
        "Designed for deployment on IoT or edge devices running a BAS."
    ),
    version="0.3.0",
)

_model_state: Optional[ModelState] = None


@app.post("/train")
def train_model(request: TrainRequest) -> Dict[str, Any]:
    """Train a probability‑based occupancy model."""
    global _model_state
    try:
        df = load_occupancy_data(
            request.data_path,
            time_col=request.time_col,
            occ_col=request.occ_col,
            parse_dates=True,
            tz=request.tz,
            rename_columns=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {exc}")
    df_clean = clean_occupancy_counts(df, occ_col="occ", deadband=request.deadband, make_binary=True)
    step_minutes = infer_step_minutes(df_clean, time_col="time")
    prob_table = build_probability_table(
        df_clean,
        occ_threshold=0.0,
        step_minutes=step_minutes,
        occ_col="clean_count",
    )
    matrix = export_probability_schedule(
        prob_table,
        output_path=None,
        prob_threshold=request.prob_threshold,
        step_minutes=step_minutes,
    )
    hour_index = list(matrix.index.values.tolist())
    _model_state = ModelState(
        step_minutes=step_minutes,
        prob_threshold=request.prob_threshold,
        tz=request.tz,
        prob_table=prob_table,
        schedule_matrix=matrix,
        hour_index=hour_index,
    )
    adf_stat, adf_p, _ = adf_stationary(df["occ"].values)
    kpss_stat, kpss_p, _ = kpss_stationary(df["occ"].values)
    return {
        "message": "Model trained successfully",
        "records": len(df),
        "step_minutes": step_minutes,
        "adf_p": adf_p,
        "adf_stationary": bool(adf_stat),
        "kpss_p": kpss_p,
        "kpss_stationary": bool(kpss_stat),
    }


@app.get("/state", response_model=StateResponse)
def get_state(at: Optional[str] = Query(None, description="ISO 8601 timestamp (UTC) to evaluate. Defaults to now.")) -> Any:
    """Return the current occupancy state and next transition."""
    global _model_state
    if _model_state is None:
        raise HTTPException(status_code=400, detail="Model has not been trained yet")
    if at is not None:
        try:
            ref_dt = _dt.datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid ISO 8601 timestamp")
    else:
        ref_dt = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo  # type: ignore
        tzinfo = ZoneInfo(_model_state.tz)
    except Exception:
        tzinfo = None
    if tzinfo is not None:
        ref_local = ref_dt.astimezone(tzinfo)
    else:
        ref_local = ref_dt
    dow = ref_local.weekday()
    minute_of_day = ref_local.hour * 60 + ref_local.minute
    step = _model_state.step_minutes
    slot = (minute_of_day // step) * step
    prob = _model_state.get_probability(dow, slot)
    curr_state = "OCCUPIED" if prob >= _model_state.prob_threshold else "UNOCCUPIED"
    next_state = None
    next_minutes = 0
    next_dt_local = ref_local
    curr_schedule_value = _model_state.get_schedule_value(dow, slot)
    for i in range(1, (7 * 24 * 60) // step + 1):
        minutes_ahead = i * step
        future_minute = minute_of_day + minutes_ahead
        future_dow = (dow + (future_minute // (24 * 60))) % 7
        future_slot = (future_minute % (24 * 60) // step) * step
        future_value = _model_state.get_schedule_value(future_dow, future_slot)
        if future_value != curr_schedule_value:
            next_minutes = minutes_ahead
            next_dt_local = ref_local + _dt.timedelta(minutes=minutes_ahead)
            next_state = "START" if future_value == 1 else "END"
            break
    if next_state is None:
        next_minutes = 7 * 24 * 60
        next_dt_local = ref_local + _dt.timedelta(minutes=next_minutes)
        next_state = "END" if curr_state == "OCCUPIED" else "START"
    next_dt_utc = next_dt_local.astimezone(_dt.timezone.utc)
    return {
        "reference_time": ref_dt.astimezone(_dt.timezone.utc).isoformat(),
        "current_state": curr_state,
        "probability": prob,
        "next_transition": {
            "type": next_state,
            "in_minutes": int(next_minutes),
            "at": next_dt_utc.isoformat(),
        },
    }


@app.get("/schedule", response_model=ScheduleResponse)
def get_schedule() -> Any:
    """Return the full schedule matrix."""
    global _model_state
    if _model_state is None:
        raise HTTPException(status_code=400, detail="Model has not been trained yet")
    matrix = _model_state.schedule_matrix
    cols = list(range(7))
    matrix = matrix.reindex(columns=cols, fill_value=0).fillna(0).astype(int)
    schedule_rows: List[List[int]] = matrix.values.tolist()
    return {
        "step_minutes": _model_state.step_minutes,
        "hours": _model_state.hour_index,
        "schedule": schedule_rows,
    }