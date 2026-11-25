import numpy as np
import pandas as pd
import pytest

from coal_forecast import models
from coal_forecast.data import clean_series, load_raw
from coal_forecast.evaluate import backtest, mape, rolling_origins
from coal_forecast.forecast import future_forecast


@pytest.fixture
def series():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=500)
    return pd.Series(80 + np.cumsum(rng.normal(0, 1, 500)), index=idx)


def test_loader_handles_header_row_and_blank_header(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_text(",A,B\nDate,Price,Price\n2020-01-02,1,2\n2020-01-03,3,x\n2020-01-03,4,5\n")
    df = load_raw(f)
    assert list(df.columns) == ["A", "B"]
    assert len(df) == 2                      # header row and duplicate date removed
    assert df.index.is_monotonic_increasing
    assert df["B"].dtype.kind == "f"          # "x" coerced to NaN


def test_clean_series_uses_only_past_values():
    s = pd.Series([1.0, np.nan, np.nan, 4.0], index=pd.bdate_range("2020-01-01", periods=4))
    assert clean_series(s).tolist() == [1.0, 1.0, 1.0, 4.0]


def test_rolling_origins_end_at_series_end_and_respect_min_train():
    o = rolling_origins(n=500, horizon=60, n_origins=4, step=30, min_train=250)
    assert o == [350, 380, 410, 440] and o[-1] + 60 == 500
    assert rolling_origins(500, 60, 6, 50, 250) == [290, 340, 390, 440]   # 240 and 190 dropped
    with pytest.raises(ValueError):
        rolling_origins(200, 60, 4, 30, 250)


@pytest.mark.parametrize("name", ["Naive", "Drift", "Linear Regression (lags)",
                                  "Random Forest (lags)", "SVR (lags)"])
def test_models_return_finite_forecast_of_right_length(series, name):
    out = models.MODELS[name](series.values, 30, 42)
    assert out.shape == (30,) and np.all(np.isfinite(out))


def test_forecasts_are_in_original_units(series):
    out = models.random_forest(series.values, 30)
    assert out.min() > series.min() - 50 and out.max() < series.max() + 50  # not 0-1 scaled


KW = dict(horizon=30, n_origins=3, step=15, min_train=100)


def test_backtest_reports_cv_and_test_and_skips_failures(series):
    def broken(history, horizon, seed=42):
        raise RuntimeError("boom")

    m, preds, skipped, sel = backtest(series, {"Drift": models.drift, "Broken": broken}, **KW)
    assert "Broken" in skipped and set(m["Model"]) == {"Naive", "Drift"}   # Naive is always included
    assert m["CV_MAPE"].is_monotonic_increasing
    assert all(len(p) == 30 for p in preds.values())
    assert sel["n_origins"] == 3


def test_selection_never_uses_the_test_window(series):
    """CV scores must be identical if the final test window is corrupted."""
    corrupted = series.copy()
    corrupted.iloc[-30:] = 1e6
    a, *_ = backtest(series, {"Drift": models.drift}, **KW)
    b, *_ = backtest(corrupted, {"Drift": models.drift}, **KW)
    np.testing.assert_allclose(a.set_index("Model")["CV_MAPE"], b.set_index("Model")["CV_MAPE"])


def test_falls_back_to_naive_when_no_model_beats_it(series):
    def worse(history, horizon, seed=42):          # naive plus a constant error
        return np.full(horizon, history[-1] + 25.0)

    _, _, _, sel = backtest(series, {"Worse": worse}, **KW)
    assert sel["selected"] == "Naive" and not sel["has_skill_over_naive"]


def test_uses_better_model_when_it_clearly_beats_naive():
    idx = pd.bdate_range("2020-01-01", periods=500)
    trend = pd.Series(50 + 0.5 * np.arange(500.0), index=idx)   # perfectly linear
    _, _, _, sel = backtest(trend, {"Drift": models.drift}, **KW)
    assert sel["selected"] == "Drift" and sel["has_skill_over_naive"]


def test_future_forecast_dates_follow_last_observation(series):
    fc = future_forecast(series, models.naive, 10, model_name="Naive")
    assert fc["Date"].min() > series.index[-1]
    assert len(fc) == 10 and (fc["Lower (approx 95%)"] <= fc["Upper (approx 95%)"]).all()


def test_mape_basic():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
