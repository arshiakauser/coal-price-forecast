"""Static plots written by the training script."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_forecast(series, test_pred, future, title, path, history_days: int = 400):
    """History, hold-out predictions (vs actual) and the future forecast."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index[-history_days:], series.values[-history_days:], color="black", label="Actual")
    ax.plot(test_pred.index, test_pred.values, color="orange", label="Test-window prediction")
    ax.plot(future["Date"], future["Forecast"], color="tab:blue", label="Future forecast")
    ax.fill_between(future["Date"], future["Lower (approx 95%)"], future["Upper (approx 95%)"],
                    color="tab:blue", alpha=0.2, label="Approx. 95% band")
    ax.set(title=title, xlabel="Date", ylabel="USD/t")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_model_comparison(metrics, title, path):
    """Cross-validated vs test MAPE per model."""
    m = metrics.set_index("Model")[["CV_MAPE", "Test_MAPE"]]
    ax = m.plot.bar(figsize=(12, 6))
    ax.set(title=title, ylabel="MAPE (%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)
