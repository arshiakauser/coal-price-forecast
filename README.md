# Coal Price Forecast

Compare statistical, machine-learning and deep-learning models on daily coal
prices, pick the best one by **rolling-origin cross-validation** (falling back to a naive baseline unless a model clearly beats it), report its error on an
untouched **test window**, and produce a real future forecast (in USD/t) that
can be explored in a Streamlit dashboard.

> **The dataset is proprietary and is not part of this repository.** See
> [`data/README.md`](data/README.md). A randomly generated sample is included
> so everything can be run end to end.

## Repository layout

```
├── src/coal_forecast/
│   ├── config.py      # series names, horizon, split sizes
│   ├── data.py        # load / clean / chronological split (no scaling here)
│   ├── models.py      # 12 models behind one interface (baselines, stats, ML, DL)
│   ├── evaluate.py    # metrics + rolling-origin model selection
│   ├── forecast.py    # refit on all data, forecast future business days
│   ├── plotting.py
│   └── train.py       # CLI entry point
├── app/streamlit_app.py   # dashboard
├── scripts/
│   ├── eda.py                 # exploratory plots
│   └── make_synthetic_data.py # generates the sample CSV
├── tests/                 # pytest suite (leakage, units, splits, loader)
├── data/                  # put your own file here (git-ignored); sample/ = synthetic demo data
└── outputs/               # generated results (git-ignored)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # or: pip install -e ".[deep,app,dev]"
export PYTHONPATH=src                                   # Windows: set PYTHONPATH=src  (not needed after pip install -e .)

# 1. (optional) explore the data
python scripts/eda.py --data data/coal_prices.xlsx

# 2. compare models and forecast the next 180 business days
python -m coal_forecast.train --data data/coal_prices.xlsx
#    no real data? use the synthetic sample:
python -m coal_forecast.train --data data/sample/synthetic_coal_prices.csv

# 3. dashboard
streamlit run app/streamlit_app.py
```

Useful options: `--horizon 60` (default), `--n-origins 4`, `--origin-step`, `--min-train 250`,
`--columns`, `--models "Naive" "ARIMA(5,1,0)"`, `--skip-deep` (skips the four neural
models; much faster), `--seed 42`.
TensorFlow is optional: without it the neural models are skipped and listed in
`outputs/best_models.json`.

### Outputs (`outputs/`)

| File | Content |
|---|---|
| `metrics.csv` | per model: CV (rolling-origin) and Test MAE / RMSE / MAPE, `CV_vs_Naive`, `Folds_beating_Naive` |
| `best_models.json` | per series: best by CV, model actually selected, `has_skill_over_naive`, skipped models |
| `history.csv` | last 500 cleaned prices per series (the dashboard draws these behind the forecast) |
| `forecasts/<series>.csv`, `forecast_all.csv` | future forecast in **USD/t** with approximate band and a `Note` column |
| `plots/*.png` | forecast and model-comparison charts |

## Method

For each price series and a chosen `--horizon` (business days, default 60):

1. **Test window**: the last `horizon` observations are held out and never used to choose a model.
2. **Rolling-origin CV**: on the earlier data, each model is fitted up to several origins
   (default 4, spaced `horizon/2` apart, at least 250 observations of training data) and forecasts
   the next `horizon` days. The mean error across origins ranks the models.
3. **Naive safeguard**: the top model is used only if it lowers CV error by at least 5 % versus Naive
   **and** beats Naive in at least 60 % of the folds. Otherwise Naive (last price) is used and the
   series is flagged "no skill" in the outputs and the dashboard.
4. **Test report**: every model is refit on all non-test data and scored once on the test window.
5. **Forecast**: the selected model is refit on all data and predicts the next `horizon` business days.

Models (all multi-step, all in USD/t): Naive and Drift baselines, SES, damped Holt, ARIMA(5,1,0),
Linear Regression / Random Forest / SVR on lagged daily changes, and LSTM, BiLSTM, GRU, CNN-LSTM.
Scaling is fitted on a model's own training history only.

### Changing the horizon (e.g. 60 → 180)

Just pass `--horizon 180`. The horizon sets the forecast length *and* the length of every evaluation
window, so a longer horizon means: errors grow (roughly with √horizon), the band widens, each fold and
the test window need 180 observations (so fewer folds fit in short histories), and the Naive safeguard
will trigger more often because recursive multi-step models rarely beat "last price" that far out.

## Business context and fit

Goal: reduce coal purchasing cost and the impact of price volatility on steel production cost, instead
of buying only as needed.

What the results support: on this data, price history alone does **not** reliably predict the direction
of coal prices 60 days ahead (most series fall back to the naive baseline). Prices typically move about
8-12 % over 60 days (the Naive CV MAPE in `metrics.csv`). The forecast is therefore best used as a
**cost range for budgeting and risk limits**, not as a buy-now/buy-later signal.

What it does not show: the profit-margin (+10 %) and revenue (+20 %) targets cannot be verified from a
price model. That requires procurement volumes, contract/spot terms, storage and financing costs, steel
selling prices and iron-ore data, plus a back-test of a purchasing policy against buy-as-needed.

## Limitations

- The band in the forecast is **approximate** (recent daily volatility × √horizon,
  random-walk assumption), not a model-based confidence interval.
- Forecasts use price history only (no fuel demand, freight, FX, etc.).
  Recursive multi-step forecasts over 180 days become very uncertain.
- With about 4 years of daily data only a few non-overlapping evaluation windows exist, so
  rankings can still change with a different history or horizon; the Naive safeguard is there for that reason.

## Tests

```bash
pytest
```
