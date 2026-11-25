"""Project-wide defaults."""
import re
from pathlib import Path

# Repository root (…/src/coal_forecast/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

# Price series to model. If none of these exist in the input file,
# every numeric column is used instead.
PRICE_COLUMNS = [
    "Coal Richards Bay 4800kcal NAR fob, London close, USD/t",
    "Coal Richards Bay 5500kcal NAR fob, London close, USD/t",
    "Coal Richards Bay 5700kcal NAR fob, London close, USD/t",
    "Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t",
    "Coal India 5500kcal NAR cfr, London close, USD/t",
]

HORIZON = 60       # business days to forecast; also the length of each evaluation window
N_ORIGINS = 4      # rolling-origin folds used to compare models
MIN_TRAIN = 250    # minimum observations before the first fold
MIN_IMPROVEMENT = 0.05  # a model must cut CV error vs Naive by at least 5% ...
MIN_WIN_RATE = 0.6      # ... and beat Naive in at least 60% of the folds
SEED = 42


def safe_name(col: str) -> str:
    """Filesystem-safe version of a column name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_")
