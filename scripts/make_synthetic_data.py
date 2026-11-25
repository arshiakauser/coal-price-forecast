"""Generate SYNTHETIC coal-like prices so the pipeline can be run without the
proprietary dataset. The numbers are random and have no market meaning.

The file mimics the layout of the original export (blank first header, then a
"Date / Price" row) so the loader's handling of it is exercised.

    python scripts/make_synthetic_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coal_forecast.config import PRICE_COLUMNS, ROOT  # noqa: E402


def main(out=ROOT / "data" / "sample" / "synthetic_coal_prices.csv", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-04-02", "2024-06-28")
    n = len(dates)

    # shared market factor + series-specific noise, with a 2022-style spike
    def ou(sd):  # mean-reverting random walk, keeps prices in a sensible range
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.995 * x[t - 1] + rng.normal(0, sd)
        return x

    market = ou(1.0)
    spike = 60 * np.exp(-0.5 * ((np.arange(n) - 0.65 * n) / 70) ** 2)
    base = {"4800": 45, "5500": 60, "5700": 70, "6000": 75}
    cols = {}
    for i, col in enumerate(PRICE_COLUMNS):
        level = next((v for k, v in base.items() if k in col), 65)
        s = level + 0.8 * market + spike * (0.8 + 0.1 * i) + ou(0.4)
        s = np.maximum(s, 10).round(2)
        s[rng.random(n) < 0.015] = np.nan          # ~1.5% missing
        cols[col] = s

    df = pd.DataFrame(cols, index=dates)
    df.index = df.index.strftime("%Y-%m-%d")
    header_row = pd.DataFrame([["Price"] * len(PRICE_COLUMNS)], index=["Date"], columns=df.columns)
    out_df = pd.concat([header_row, df])
    out_df.index.name = None

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out)   # first header cell is blank, like the original Excel export
    print(f"Wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()
