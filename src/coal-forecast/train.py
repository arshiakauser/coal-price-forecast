"""Command-line entry point: compare models, pick the best, forecast the future.

    python -m coal_forecast.train --data data/coal_prices.xlsx --horizon 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import config
from .data import clean_series, ensure_dir, load_raw, select_columns
from .evaluate import backtest
from .forecast import future_forecast
from .models import DEEP, MODELS
from .plotting import plot_forecast, plot_model_comparison

log = logging.getLogger("coal_forecast")
NO_SKILL_NOTE = "No model clearly beat the Naive baseline; Naive (last price) is used."


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Raw CSV/XLSX (first column = date).")
    p.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR))
    p.add_argument("--columns", nargs="*", help="Price columns to model (default: all configured).")
    p.add_argument("--models", nargs="*", help=f"Subset of: {', '.join(MODELS)}")
    p.add_argument("--skip-deep", action="store_true", help="Skip LSTM/BiLSTM/GRU/CNN-LSTM (much faster).")
    p.add_argument("--horizon", type=int, default=config.HORIZON,
                   help="Business days to forecast; also the length of every evaluation window.")
    p.add_argument("--n-origins", type=int, default=config.N_ORIGINS, help="Rolling-origin folds.")
    p.add_argument("--origin-step", type=int, default=None, help="Days between folds (default horizon/2).")
    p.add_argument("--min-train", type=int, default=config.MIN_TRAIN)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = ensure_dir(args.output_dir)
    fc_dir, plot_dir = ensure_dir(out / "forecasts"), ensure_dir(out / "plots")

    models = dict(MODELS)
    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            log.error("Unknown models: %s", unknown)
            return 2
        models = {m: MODELS[m] for m in args.models}
    if args.skip_deep:
        models = {k: v for k, v in models.items() if k not in DEEP}

    df = load_raw(args.data)
    columns = select_columns(df, args.columns)
    log.info("Loaded %d rows, modelling %d series, horizon %d business days",
             len(df), len(columns), args.horizon)

    all_metrics, all_forecasts, all_history, summary, skipped_all = [], [], [], {}, {}
    for col in columns:
        log.info("Processing: %s", col)
        series = clean_series(df[col])
        metrics, test_preds, skipped, sel = backtest(
            series, models, args.horizon, args.n_origins, args.origin_step, args.min_train, args.seed,
            config.MIN_IMPROVEMENT, config.MIN_WIN_RATE,
        )
        skipped_all[col] = skipped
        chosen = sel["selected"]
        row = metrics.set_index("Model").loc[chosen]
        log.info("Best by CV: %s | selected: %s (CV MAPE %.2f%%, test MAPE %.2f%%)%s",
                 sel["best_by_cv"], chosen, row["CV_MAPE"], row["Test_MAPE"],
                 "" if sel["has_skill_over_naive"] else "  [no skill over Naive]")

        # refit the selected model on ALL data and forecast beyond the last observation
        all_models = {"Naive": MODELS["Naive"], **models}
        fc = future_forecast(series, all_models[chosen], args.horizon, args.seed, chosen)
        fc["Price Column"] = col
        fc["Note"] = "" if sel["has_skill_over_naive"] else NO_SKILL_NOTE
        name = config.safe_name(col)
        fc.to_csv(fc_dir / f"{name}.csv", index=False)

        metrics.insert(0, "Price Column", col)
        all_metrics.append(metrics)
        all_forecasts.append(fc)
        all_history.append(pd.DataFrame({"Date": series.index[-500:], "Price Column": col,
                                         "Price": series.values[-500:]}))
        summary[col] = {**sel, "cv_mape": row["CV_MAPE"], "test_mape": row["Test_MAPE"],
                        "naive_cv_mape": float(metrics.loc[metrics["Model"] == "Naive", "CV_MAPE"].iloc[0]),
                        "naive_test_mape": float(metrics.loc[metrics["Model"] == "Naive", "Test_MAPE"].iloc[0])}

        title = f"{chosen}{'' if sel['has_skill_over_naive'] else ' (no skill over Naive)'}: {col}"
        plot_forecast(series, test_preds[chosen], fc, title, plot_dir / f"{name}_forecast.png")
        plot_model_comparison(metrics, f"CV vs test MAPE: {col}", plot_dir / f"{name}_model_comparison.png")

    pd.concat(all_metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(all_forecasts).to_csv(out / "forecast_all.csv", index=False)
    pd.concat(all_history).to_csv(out / "history.csv", index=False)   # lets the dashboard draw past prices
    (out / "best_models.json").write_text(json.dumps(
        {"horizon": args.horizon, "series": summary, "skipped": skipped_all}, indent=2, default=float))
    log.info("Done. Results in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
