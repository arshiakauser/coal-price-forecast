"""Metrics and honest model comparison.

Protocol (per price series), all forecasts are multi-step, ``horizon``
business days ahead, in USD/t:

1. The last ``horizon`` observations are a TEST window that is never used to
   choose a model.
2. On the remaining data, models are scored over several rolling origins:
   fit on everything before the origin, forecast the next ``horizon`` days,
   compare with what happened. The mean error over origins (CV) selects the model.
3. The selected model must clearly beat the Naive baseline (lower CV error by
   ``min_improvement`` and better in at least ``min_win_rate`` of the origins).
   Otherwise Naive is used and the series is flagged "no skill".
4. Every model is then refit on all non-test data and scored once on the test window.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .models import naive

log = logging.getLogger(__name__)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)


def rolling_origins(n: int, horizon: int, n_origins: int, step: int, min_train: int) -> list[int]:
    """Indices (into a series of length ``n``) where each fold's forecast starts.

    The last fold's window ends at the end of the series; earlier folds step
    back by ``step``. Origins with fewer than ``min_train`` observations are dropped.
    """
    origins = [n - horizon - i * step for i in range(n_origins)]
    origins = sorted(o for o in origins if o >= min_train)
    if not origins:
        raise ValueError(
            f"Series too short ({n} rows) for horizon={horizon}, min_train={min_train}. "
            "Use a shorter --horizon or --min-train."
        )
    return origins


def backtest(series: pd.Series, models: dict, horizon: int, n_origins: int = 4,
             step: int | None = None, min_train: int = 250, seed: int = 42,
             min_improvement: float = 0.05, min_win_rate: float = 0.6):
    """Compare ``models`` on one series.

    Returns ``(metrics_df, test_predictions, skipped, selection)``. ``metrics_df``
    is sorted by CV MAPE; ``selection`` says which model to use and whether it
    has skill over Naive.
    """
    step = step or max(horizon // 2, 5)
    models = {"Naive": naive, **{k: v for k, v in models.items() if k != "Naive"}}

    dev, test = series.iloc[:-horizon], series.iloc[-horizon:]
    origins = rolling_origins(len(dev), horizon, n_origins, step, min_train)
    x = dev.values

    fold_mape, rows, test_preds, skipped = {}, [], {}, {}
    for name, fn in models.items():
        try:
            m, a, r = [], [], []
            for o in origins:
                pred = np.asarray(fn(x[:o], horizon, seed), float)
                actual = x[o:o + horizon]
                m.append(mape(actual, pred)); a.append(mae(actual, pred)); r.append(rmse(actual, pred))
            p_test = np.asarray(fn(x, horizon, seed), float)
            if not (np.all(np.isfinite(m)) and np.all(np.isfinite(p_test))):
                raise ValueError("model returned non-finite values")
        except Exception as exc:  # noqa: BLE001 - report and carry on
            skipped[name] = f"{type(exc).__name__}: {exc}"
            log.warning("Skipping %s: %s", name, skipped[name])
            continue
        fold_mape[name] = np.array(m)
        rows.append({
            "Model": name,
            "CV_MAE": float(np.mean(a)), "CV_RMSE": float(np.mean(r)),
            "CV_MAPE": float(np.mean(m)), "CV_MAPE_std": float(np.std(m)),
            "Test_MAE": mae(test, p_test), "Test_RMSE": rmse(test, p_test),
            "Test_MAPE": mape(test, p_test),
        })
        test_preds[name] = pd.Series(p_test, index=test.index)

    metrics = pd.DataFrame(rows).sort_values("CV_MAPE").reset_index(drop=True)
    naive_cv = metrics.loc[metrics["Model"] == "Naive", "CV_MAPE"].iloc[0]
    metrics["CV_vs_Naive"] = metrics["CV_MAPE"] / naive_cv
    metrics["Folds_beating_Naive"] = [
        np.nan if m == "Naive" else float(np.mean(fold_mape[m] < fold_mape["Naive"]))
        for m in metrics["Model"]
    ]
    metrics.attrs["origins"] = len(origins)

    best = metrics.iloc[0]
    has_skill = bool(
        best["Model"] != "Naive"
        and best["CV_vs_Naive"] <= 1 - min_improvement
        and best["Folds_beating_Naive"] >= min_win_rate
    )
    selection = {
        "best_by_cv": best["Model"],
        "selected": best["Model"] if has_skill else "Naive",
        "has_skill_over_naive": has_skill,
        "n_origins": len(origins),
    }
    return metrics, test_preds, skipped, selection
