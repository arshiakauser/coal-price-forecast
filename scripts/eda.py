"""Exploratory data analysis. Saves figures to outputs/eda/ (no scaling, no modelling).

    python scripts/eda.py --data data/coal_prices.xlsx
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from coal_forecast.config import DEFAULT_OUTPUT_DIR, safe_name  # noqa: E402
from coal_forecast.data import clean_series, ensure_dir, load_raw, select_columns  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "eda"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("eda")

    out = ensure_dir(args.output_dir)
    df = load_raw(args.data)
    cols = select_columns(df)

    log.info("Shape: %s | %s -> %s", df.shape, df.index.min().date(), df.index.max().date())
    log.info("Missing values:\n%s", df[cols].isna().sum())
    log.info("Duplicated dates: %d", df.index.duplicated().sum())
    log.info("Summary:\n%s", df[cols].describe().T)

    # distribution / outliers (visual only - nothing is capped)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[cols], ax=ax)
    ax.set_xticklabels([c[:30] for c in cols], rotation=60, ha="right")
    ax.set_title("Price distribution")
    fig.tight_layout()
    fig.savefig(out / "boxplot.png", dpi=120)
    plt.close(fig)

    # trends
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in cols:
        ax.plot(df.index, df[c], label=c[:40])
    ax.set(title="Coal price trends", xlabel="Date", ylabel="USD/t")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "trends.png", dpi=120)
    plt.close(fig)

    # correlation
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation between price series")
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=120)
    plt.close(fig)

    # per-series: moving averages, decomposition, distribution
    for c in cols:
        s = clean_series(df[c])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(s, alpha=0.5, label="Price")
        ax.plot(s.rolling(7).mean(), "--", label="7-day MA")
        ax.plot(s.rolling(30).mean(), ":", color="red", label="30-day MA")
        ax.set(title=c, ylabel="USD/t")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{safe_name(c)}_moving_avg.png", dpi=120)
        plt.close(fig)

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            res = seasonal_decompose(s, model="additive", period=21)  # ~1 trading month
            fig = res.plot()
            fig.set_size_inches(12, 8)
            fig.savefig(out / f"{safe_name(c)}_decomposition.png", dpi=120)
            plt.close(fig)
        except ImportError:
            log.warning("statsmodels not installed: skipping decomposition")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(s, bins=30, kde=True, ax=ax)
        ax.set(title=f"Distribution: {c[:50]}", xlabel="USD/t")
        fig.tight_layout()
        fig.savefig(out / f"{safe_name(c)}_hist.png", dpi=120)
        plt.close(fig)

    log.info("EDA figures saved to %s", out)


if __name__ == "__main__":
    main()
