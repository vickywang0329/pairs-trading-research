# -*- coding: utf-8 -*-
"""
Engle-Granger cointegration screen. Tests every pair in the stock universe
and keeps the ones that pass both the p-value cutoff and the half-life range.

Usage:
    python src/cointegration.py

Output: output/cointegration_candidates.csv
Columns: stock_a, stock_b, direction, p_value, hedge_ratio, half_life_days
"""

import os
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import (
    RAW_DATA_PATH,
    COINTEGRATION_RESULT_PATH,
    TRAIN_TEST_SPLIT_DATE,
    COINTEGRATION_SIGNIFICANCE,
    MIN_OBSERVATIONS,
    MIN_HALF_LIFE_DAYS,
    MAX_HALF_LIFE_DAYS,
    STOCK_UNIVERSE,
)


def load_log_prices() -> pd.DataFrame:
    """Log closing prices, one column per stock, train period only.
    I cut at the split date so pair selection never sees the test year.
    Benchmark ETFs are dropped here since they're only used in features.py."""
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df[df["close"] > 0]
    df = df[df["date"] < pd.Timestamp(TRAIN_TEST_SPLIT_DATE)]
    df = df[df["stock_id"].isin(STOCK_UNIVERSE.keys())]

    wide = df.pivot(index="date", columns="stock_id", values="close")
    wide = wide.sort_index()

    # A stock that listed late would force dropna() below to cut the whole
    # window down to its listing date. I drop such stocks instead and say so.
    first_valid = wide.apply(lambda col: col.first_valid_index())
    late = first_valid[first_valid > wide.index[0] + pd.Timedelta(days=30)].index
    if len(late) > 0:
        print(f"Dropped for short history: {', '.join(late)}")
        wide = wide.drop(columns=late)

    wide = wide.ffill().dropna(how="any")  # align trading days, drop gaps
    log_price = np.log(wide)
    return log_price


def estimate_theta(spread: pd.Series) -> float:
    """Mean-reversion speed from the AR(1) form of an O-U process: regress the
    spread's daily change on its previous level. theta < 0 means it reverts
    (more negative = stronger pull); theta >= 0 means no reversion."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_diff.index]

    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()
    return model.params.iloc[1]


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """Half-life in trading days, from theta. Shorter = reverts faster."""
    theta = estimate_theta(spread)
    if theta >= 0:
        return np.inf  # no mean reversion at all
    return -np.log(2) / theta


def test_pair(log_price: pd.DataFrame, stock_a: str, stock_b: str,
              min_obs: int = MIN_OBSERVATIONS) -> dict | None:
    """Run Engle-Granger in both regression directions and keep the more
    significant one. The test isn't symmetric (A on B gives different residuals
    than B on A), so one direction can pass while the other fails."""
    y = log_price[stock_a]
    x = log_price[stock_b]

    if len(y) < min_obs:
        return None

    # direction 1: A ~ B
    score_ab, pvalue_ab, _ = coint(y, x)
    # direction 2: B ~ A
    score_ba, pvalue_ba, _ = coint(x, y)

    if pvalue_ab <= pvalue_ba:
        direction = f"{stock_a}~{stock_b}"
        pvalue = pvalue_ab
        dependent, independent = y, x
    else:
        direction = f"{stock_b}~{stock_a}"
        pvalue = pvalue_ba
        dependent, independent = x, y

    if pvalue > COINTEGRATION_SIGNIFICANCE:
        return None

    # hedge ratio and spread come from the regression in the better direction
    X = add_constant(independent)
    hedge_model = OLS(dependent, X).fit()
    hedge_ratio = hedge_model.params.iloc[1]
    spread = dependent - hedge_ratio * independent

    hl = half_life_of_mean_reversion(spread)

    # A significant p-value isn't enough. Too short is probably noise, too long
    # is slower than I'm willing to hold.
    if not (MIN_HALF_LIFE_DAYS <= hl <= MAX_HALF_LIFE_DAYS):
        return None

    return {
        "stock_a": stock_a,
        "stock_b": stock_b,
        "direction": direction,
        "p_value": round(pvalue, 5),
        "hedge_ratio": round(hedge_ratio, 4),
        "half_life_days": round(hl, 1) if np.isfinite(hl) else None,
    }


def run_all_pairs() -> pd.DataFrame:
    log_price = load_log_prices()
    stocks = list(log_price.columns)
    print(f"Train period: {log_price.index.min().date()} ~ {log_price.index.max().date()}, "
          f"{len(log_price)} trading days, {len(stocks)} stocks")
    print(f"Testing {len(list(itertools.combinations(stocks, 2)))} pairs\n")

    results = []
    for stock_a, stock_b in itertools.combinations(stocks, 2):
        r = test_pair(log_price, stock_a, stock_b)
        if r is not None:
            results.append(r)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # smaller p-value and shorter half-life first
        result_df = result_df.sort_values(
            ["p_value", "half_life_days"], na_position="last"
        ).reset_index(drop=True)
    return result_df


def main():
    result_df = run_all_pairs()
    os.makedirs(os.path.dirname(COINTEGRATION_RESULT_PATH) or ".", exist_ok=True)
    result_df.to_csv(COINTEGRATION_RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"{len(result_df)} candidate pairs kept "
          f"(p < {COINTEGRATION_SIGNIFICANCE}, "
          f"half-life between {MIN_HALF_LIFE_DAYS} and {MAX_HALF_LIFE_DAYS} days)")
    if not result_df.empty:
        print(result_df.head(10).to_string(index=False))
    print(f"\nSaved to {COINTEGRATION_RESULT_PATH}")


if __name__ == "__main__":
    main()
