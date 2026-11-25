"""Streamlit dashboard for the forecasts produced by ``coal_forecast.train``.

    streamlit run app/streamlit_app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from coal_forecast.config import DEFAULT_OUTPUT_DIR, safe_name  # noqa: E402

st.set_page_config(page_title="Coal Price Forecast", layout="wide")
st.title("⛏️ Coal Price Forecast Dashboard")

# ------------------------------------------------------------------ sidebar
with st.sidebar.expander("Advanced settings"):
    out_dir = Path(st.text_input("Results folder",
                                 os.getenv("COAL_RESULTS_DIR", str(DEFAULT_OUTPUT_DIR))))


@st.cache_data
def load_forecasts(folder: str, mtime: float):
    return pd.read_csv(Path(folder) / "forecast_all.csv", parse_dates=["Date"])


@st.cache_data
def load_metrics(folder: str, mtime: float):
    path = Path(folder) / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


fc_path = out_dir / "forecast_all.csv"
if not fc_path.exists():
    st.error(f"No forecasts found at `{fc_path}`. Run `python -m coal_forecast.train --data <file>` first.")
    st.stop()

forecasts = load_forecasts(str(out_dir), fc_path.stat().st_mtime)
metrics = load_metrics(str(out_dir), fc_path.stat().st_mtime)
hist_path = out_dir / "history.csv"
history = pd.read_csv(hist_path, parse_dates=["Date"]) if hist_path.exists() else None

columns = sorted(forecasts["Price Column"].unique())
selected = st.sidebar.multiselect("Price series", columns, default=columns[:1])
max_h = int(forecasts.groupby("Price Column").size().min())
horizon = st.sidebar.slider("Show first N business days", 5, max_h, max_h)

# --------------------------------------------------------------------- main
if not selected:
    st.info("Select at least one price series in the sidebar.")
    st.stop()

st.caption("All values are USD/t. Bands are approximate (recent volatility × √horizon), not model-based intervals.")
for tab, col in zip(st.tabs(selected), selected):
    with tab:
        fc = forecasts[forecasts["Price Column"] == col].head(horizon)
        model = fc["Model"].iloc[0]
        note = fc["Note"].iloc[0] if "Note" in fc.columns else ""
        if isinstance(note, str) and note:
            st.warning(note)
        full = forecasts[forecasts["Price Column"] == col]
        end = fc.iloc[-1]
        cards = st.columns(4)
        if history is not None:
            last = history[history["Price Column"] == col].iloc[-1]
            cards[0].metric(f"Last price ({last['Date']:%d %b %Y})", f"{last['Price']:.2f} USD/t")
        cards[1].metric(f"Expected on {end['Date']:%d %b %Y}", f"{end['Forecast']:.2f} USD/t")
        cards[2].metric("Likely range (approx. 95%)",
                        f"{end['Lower (approx 95%)']:.0f} – {end['Upper (approx 95%)']:.0f}")
        if metrics is not None:
            nv = metrics[(metrics["Price Column"] == col) & (metrics["Model"] == "Naive")]
            if len(nv):
                cards[3].metric(f"Typical price move over {len(full)} days",
                                f"±{nv['CV_MAPE'].iloc[0]:.0f}%", help="Average absolute change "
                                "from today's price, measured on past data.")
        left, right = st.columns(2)
        with left:
            st.subheader(f"Forecast ({model})")
            fig, ax = plt.subplots(figsize=(10, 5))
            if history is not None:
                hist = history[history["Price Column"] == col].tail(250)
                ax.plot(hist["Date"], hist["Price"], color="black", label="History")
            ax.plot(fc["Date"], fc["Forecast"], color="tab:blue", label="Forecast")
            ax.fill_between(fc["Date"], fc["Lower (approx 95%)"], fc["Upper (approx 95%)"],
                            alpha=0.2, color="tab:blue", label="Approx. 95% band")
            ax.set_ylabel("USD/t")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader("Forecast table")
            # Model and Note are the same on every row (shown in the chart title / banner),
            # so the on-screen table keeps only the numbers. The CSV download has everything.
            view = fc.drop(columns=["Price Column", "Model", "Note"], errors="ignore").copy()
            view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(view.set_index("Date").round(2))
            st.download_button("📥 Download CSV", fc.to_csv(index=False).encode(),
                               f"{safe_name(col)}_forecast.csv", "text/csv", key=f"dl_{col}")
        if metrics is not None:
            with st.expander("Technical details: model comparison"):
                st.caption("CV = average over several past forecast windows (used to choose the model); "
                           "Test = final untouched window. Lower is better; CV_vs_Naive < 1 beats the baseline.")
                st.dataframe(metrics[metrics["Price Column"] == col].drop(columns=["Price Column"])
                             .style.format(precision=2), use_container_width=True)
