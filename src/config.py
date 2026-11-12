# -*- coding: utf-8 -*-
"""
Settings for the Taiwan semiconductor pairs trading project.
Every threshold lives here, so tuning means changing one number and rerunning.
"""

from datetime import date, timedelta
from pathlib import Path

# Project root, one level above src/. All paths below hang off this, so it
# doesn't matter whether I run a script from the repo root or from src/.
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Stock universe
# Same-industry names only. Companies in one supply chain tend to follow the
# same demand cycle, which makes cointegration plausible but proves nothing,
# so every pair still has to pass the actual test.
# ---------------------------------------------------------------------------
STOCK_UNIVERSE = {
    # foundry
    "2330": "TSMC",
    "2303": "UMC",
    "5347": "Vanguard International Semiconductor",
    "6770": "Powerchip Semiconductor Manufacturing",
    # IC design and IP
    "2454": "MediaTek",
    "3034": "Novatek",
    "2379": "Realtek",
    "3661": "Alchip Technologies",
    "3035": "Faraday Technology",
    "3443": "Global Unichip",
    "6415": "Silergy",
    "4966": "Parade Technologies",
    "5269": "ASMedia Technology",
    "8299": "Phison Electronics",
    "3529": "eMemory Technology",
    # packaging and testing
    "3711": "ASE Technology Holding",
    "6239": "Powertech Technology",
    "2449": "King Yuan Electronics",
    "6257": "Sigurd Microelectronics",
    "2441": "Greatek Electronics",
    # memory
    "2408": "Nanya Technology",
    "2344": "Winbond Electronics",
    "2337": "Macronix",
    # silicon wafer
    "6488": "GlobalWafers",
    "5483": "Sino-American Silicon Products",
    "3532": "Formosa Sumco Technology",
    # IC substrate
    "3037": "Unimicron",
    "8046": "Nan Ya Printed Circuit Board",
    "3189": "Kinsus Interconnect",
    # equipment and materials
    "3131": "Grand Process Technology",
    "3680": "Gudeng Precision",
}

# ---------------------------------------------------------------------------
# 2. Data window
# 8 years of daily bars. The last year is the out-of-sample test period;
# pair selection, features and model training only ever use the first 7.
# I went from 4 to 8 years because 3 years of training data gave too little
# statistical power: only one pair survived screening and the model had too
# few events. The cost is that the window now includes 2020-2022, so I watch
# for structural breaks in the stability check.
# ---------------------------------------------------------------------------
# END_DATE is hardcoded on purpose. With date.today() the window and the
# train/test split shift every time I run it, so results can't be reproduced
# or compared between runs. I bump it by hand when I want fresh data.
END_DATE = date(2026, 11, 12)
START_DATE = END_DATE - timedelta(days=365 * 8)

TRAIN_TEST_SPLIT_DATE = END_DATE - timedelta(days=365)  # last year = test period

# ---------------------------------------------------------------------------
# 3. Engle-Granger screening
# ---------------------------------------------------------------------------
COINTEGRATION_SIGNIFICANCE = 0.05   # keep a pair only if p-value is below this
MIN_OBSERVATIONS = 500              # skip pairs with fewer trading days than this
MIN_HALF_LIFE_DAYS = 5               # shorter than this is usually noise
MAX_HALF_LIFE_DAYS = 60              # from my 3-6 month holding tolerance and the 2-3x half-life rule of thumb

# ---------------------------------------------------------------------------
# 3b. Walk-forward pair selection (walkforward.py)
# Fixing one pair list over 7 years found nothing stable, so the main flow now
# reselects pairs on a schedule. Every REBALANCE_EVERY_DAYS I rerun the
# Engle-Granger screen on the previous FORMATION_WINDOW_DAYS and trade the
# survivors until the next reselection, using only data from before that day.
# ---------------------------------------------------------------------------
FORMATION_WINDOW_DAYS = 252         # lookback used to pick pairs, about 1 year
REBALANCE_EVERY_DAYS = 63           # reselect this often, about 1 quarter
MAX_PAIRS_PER_PERIOD = 10           # keep only the most significant pairs per period. Even with no real
                                    # cointegration about 5% of pairs pass by luck, so I cap the list.
WALKFORWARD_PAIRS_PATH = str(BASE_DIR / "output/walkforward_pairs.csv")

# ---------------------------------------------------------------------------
# 4. Stability check
# Re-estimate hedge ratio and theta on rolling windows and drop pairs whose
# parameters drift too much.
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS = 120           # window length in trading days, about 6 months
ROLLING_STEP_DAYS = 20              # how far the window moves each step, about 1 month
HEDGE_RATIO_CV_THRESHOLD = 0.30     # hedge ratio CV above this = unstable
THETA_NONREVERT_RATIO_THRESHOLD = 0.20  # share of windows with theta >= 0 above this = unstable

# ---------------------------------------------------------------------------
# 5. Benchmarks
# Used only for the relative-strength features (stock return minus benchmark
# return). Both come from the same TaiwanStockPrice endpoint as the stocks.
# ---------------------------------------------------------------------------
BENCHMARK_TICKERS = {
    "0050": "Yuanta Taiwan Top 50 ETF (market proxy)",
    "00892": "Fubon Taiwan Semiconductor ETF (sector proxy)",
}

# ---------------------------------------------------------------------------
# 6. Feature windows
# ---------------------------------------------------------------------------
ZSCORE_WINDOW_DAYS = 20               # spread z-score window, about 1 month
VOLUME_AVG_WINDOW_DAYS = 20           # window for the volume ratio
ATR_WINDOW_DAYS = 14                  # standard ATR window
RELATIVE_STRENGTH_WINDOW_DAYS = 20    # lookback for stock return minus benchmark return

FEATURE_DATASET_PATH = str(BASE_DIR / "output/features_dataset.csv")

# ---------------------------------------------------------------------------
# 7. Entry / exit / stop rules
# Shared by the label definition (model.py) and the backtest, so what the
# model learns to predict is the same thing I actually trade. A label is 1 if,
# within LABEL_HORIZON_DAYS of entry, the spread gets back inside EXIT_Z
# before it blows out past STOP_Z.
# ---------------------------------------------------------------------------
ENTRY_Z = 2.0                  # enter when |z| reaches this
EXIT_Z = 0.5                   # reverted once |z| is back inside this
STOP_Z = 4.0                   # stopped out once |z| goes past this
LABEL_HORIZON_DAYS = 30        # label window and max holding period in the backtest.
                               # Kept well under the 60-day half-life cap so slow
                               # pairs don't tie up capital.

# ---------------------------------------------------------------------------
# 8. Model
# ---------------------------------------------------------------------------
MODEL_PROB_THRESHOLD = 0.55    # only trade when predicted reversion probability >= this
MODEL_RANDOM_SEED = 42
PREDICTIONS_PATH = str(BASE_DIR / "output/predictions.csv")
MODEL_REPORT_PATH = str(BASE_DIR / "output/model_report.txt")
MODEL_PATH = str(BASE_DIR / "output/model.pkl")

# ---------------------------------------------------------------------------
# 9. Risk rules and backtest
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000          # starting capital in TWD
MAX_POSITIONS = 5                    # max pairs held at once
PER_PAIR_CAPITAL_PCT = 0.15          # max gross notional per pair (both legs) as a share of equity
MAX_GROSS_EXPOSURE_PCT = 0.70        # max total gross notional as a share of equity
MAX_LOSS_PER_TRADE_PCT = 0.02        # loss I accept if a trade hits the stop, as a share of equity (sets position size)
TRANSACTION_COST_PCT = 0.004         # rough one-way cost per leg: commission + tax + slippage, charged on entry and exit
COOLDOWN_DAYS = 5                    # wait this long after closing a pair before re-entering it

LIQUIDITY_CAP_PCT = 0.05             # a leg's notional may not exceed this share of that stock's average daily traded value
ADV_WINDOW_DAYS = 20                 # window for average daily traded value (TWD), known as of the signal day

TRADES_PATH = str(BASE_DIR / "output/trades.csv")
EQUITY_CURVE_PATH = str(BASE_DIR / "output/equity_curve.csv")
BACKTEST_REPORT_PATH = str(BASE_DIR / "output/backtest_report.txt")

# data and intermediate outputs
# ---------------------------------------------------------------------------
# 10. Extra analysis (analysis.py)
# ---------------------------------------------------------------------------
COST_SENSITIVITY_PCTS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]    # one-way cost per leg to try
CAPACITY_LEVELS = [1_000_000, 10_000_000, 50_000_000, 200_000_000, 1_000_000_000]   # starting capital in TWD
RANDOM_CONTROL_RUNS = 300            # how many random signal subsets to draw
ANALYSIS_REPORT_PATH = str(BASE_DIR / "output/analysis_report.txt")
ANALYSIS_PLOT_PATH = str(BASE_DIR / "output/analysis_plots.png")

RAW_DATA_PATH = str(BASE_DIR / "data/prices_raw.parquet")
COINTEGRATION_RESULT_PATH = str(BASE_DIR / "output/cointegration_candidates.csv")
STABILITY_RESULT_PATH = str(BASE_DIR / "output/stability_report.csv")

# Stock splits. FinMind's dividend table doesn't cover them, so they are listed by hand.
# Each entry is (first trading day at the new price, price factor). A 1-for-4 split is
# 0.25: prices before that day are multiplied by it, volume before that day is divided by it.
# Found by the 30% one-day-move check in data_collector.py and confirmed against the raw
# close and volume around the date.
KNOWN_SPLITS = {
    "0050": [("2025-06-18", 0.25)],   # raw close 188.65 -> 47.57, volume about 8x
    "6415": [("2022-07-13", 0.25)],   # raw close 2485 -> 560, volume about 5x
}
