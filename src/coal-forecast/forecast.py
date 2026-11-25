"""Produce the real future forecast with the selected model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def approx_interval(series: pd.Series, point: np.ndarray, z: float = 1.96, window: int = 250):
    """Approximate 95% band that widens with the horizon.

    Uses the standard deviation of recent daily price changes scaled by
    sqrt(h) (random-walk assumption). It is a rough guide, not a model-based
    interval; the README documents this.
    """
    sigma = float(series.diff().dropna().iloc[-window:].std())
    half = z * sigma * np.sqrt(np.arange(1, len(point) + 1))
    return point - half, point + half


def future_forecast(series: pd.Series, model_fn, horizon: int, seed: int = 42, model_name: str = ""):
    """Refit ``model_fn`` on ALL observed data and forecast ``horizon`` business days
    after the last observation. Output is in USD/t (never scaled)."""
    point = np.asarray(model_fn(series.values, horizon, seed), float)
    dates = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    lower, upper = approx_interval(series, point)
    return pd.DataFrame({
        "Date": dates,
        "Forecast": point,
        "Lower (approx 95%)": lower,
        "Upper (approx 95%)": upper,
        "Model": model_name,
    })
