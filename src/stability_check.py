# -*- coding: utf-8 -*-
"""
Stability check on the pairs that passed cointegration.

cointegration.py estimates one hedge ratio and one theta over the whole train
period, which assumes the relationship never changes. Real pairs drift: market
share shifts, fundamentals diverge. Here I cut the train period into rolling
windows, re-estimate both numbers in each window, and drop pairs where they
move around too much.

Two things I look at:
- hedge ratio CV across windows (does the long-run ratio keep changing?)
- share of windows where theta >= 0 (does the spread stop reverting at times?)

This isn't a second cointegration test. It's a sanity check on pairs that
already passed one.

Usage:
    python src/stability_check.py

Input:  output/cointegration_candidates.csv, data/prices_raw.parquet
Output: output/stability_report.csv
"""

import os
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import (
    COINTEGRATION_RESULT_PATH,
    STABILITY_RESULT_PATH,
    ROLLING_WINDOW_DAYS,
    ROLLING_STEP_DAYS,
    HEDGE_RATIO_CV_THRESHOLD,
    THETA_NONREVERT_RATIO_THRESHOLD,
)
from cointegration import load_log_prices, estimate_theta


def parse_direction(direction: str):
    """'2330~2303' -> ('2330', '2303'), i.e. (dependent, independent)."""
    dependent, independent = direction.split("~")
    return dependent, independent


def rolling_estimate(log_price: pd.DataFrame, dependent: str, independent: str) -> pd.DataFrame:
    """Hedge ratio and theta re-estimated on each rolling window for one pair."""
    y = log_price[dependent]
    x = log_price[independent]
    n = len(y)

    records = []
    start = 0
    while start + ROLLING_WINDOW_DAYS <= n:
        end = start + ROLLING_WINDOW_DAYS
        y_win = y.iloc[start:end]
        x_win = x.iloc[start:end]

        X = add_constant(x_win)
        beta = OLS(y_win, X).fit().params.iloc[1]
        spread_win = y_win - beta * x_win
        theta = estimate_theta(spread_win)

        records.append({
            "window_start": y_win.index[0],
            "window_end": y_win.index[-1],
            "hedge_ratio": beta,
            "theta": theta,
        })
        start += ROLLING_STEP_DAYS

    return pd.DataFrame(records)


def summarize_stability(rolling_df: pd.DataFrame) -> dict:
    """Boil the rolling estimates down to two numbers and a pass/fail."""
    hedge_ratios = rolling_df["hedge_ratio"]
    thetas = rolling_df["theta"]

    mean_hr = hedge_ratios.mean()
    hedge_ratio_cv = (hedge_ratios.std() / abs(mean_hr)) if mean_hr != 0 else np.inf
    nonrevert_ratio = (thetas >= 0).mean()  # theta >= 0 = no reversion in that window

    is_stable = (
        hedge_ratio_cv <= HEDGE_RATIO_CV_THRESHOLD
        and nonrevert_ratio <= THETA_NONREVERT_RATIO_THRESHOLD
    )

    return {
        "n_windows": len(rolling_df),
        "hedge_ratio_mean": round(mean_hr, 4),
        "hedge_ratio_cv": round(hedge_ratio_cv, 4),
        "theta_nonrevert_ratio": round(nonrevert_ratio, 4),
        "is_stable": is_stable,
    }


def run_all() -> pd.DataFrame:
    candidates = pd.read_csv(COINTEGRATION_RESULT_PATH)
    log_price = load_log_prices()

    if len(log_price) < ROLLING_WINDOW_DAYS:
        raise ValueError(
            f"Train period has only {len(log_price)} trading days, "
            f"fewer than the rolling window of {ROLLING_WINDOW_DAYS}, so the stability check can't run"
        )

    results = []
    for _, row in candidates.iterrows():
        dependent, independent = parse_direction(row["direction"])
        rolling_df = rolling_estimate(log_price, dependent, independent)

        if rolling_df.empty:
            continue

        summary = summarize_stability(rolling_df)
        summary.update({
            "stock_a": row["stock_a"],
            "stock_b": row["stock_b"],
            "direction": row["direction"],
            "p_value": row["p_value"],
            "half_life_days": row["half_life_days"],
        })
        results.append(summary)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        cols = [
            "stock_a", "stock_b", "direction", "p_value", "half_life_days",
            "n_windows", "hedge_ratio_mean", "hedge_ratio_cv",
            "theta_nonrevert_ratio", "is_stable",
        ]
        result_df = result_df[cols].sort_values(
            ["is_stable", "hedge_ratio_cv"], ascending=[False, True]
        ).reset_index(drop=True)
    return result_df


def main():
    result_df = run_all()
    os.makedirs(os.path.dirname(STABILITY_RESULT_PATH) or ".", exist_ok=True)
    result_df.to_csv(STABILITY_RESULT_PATH, index=False, encoding="utf-8-sig")

    n_stable = int(result_df["is_stable"].sum()) if not result_df.empty else 0
    print(f"Checked {len(result_df)} candidate pairs, {n_stable} passed the stability check\n")
    if not result_df.empty:
        print(result_df.head(10).to_string(index=False))
    print(f"\nSaved to {STABILITY_RESULT_PATH}")


if __name__ == "__main__":
    main()
