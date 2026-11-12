# -*- coding: utf-8 -*-
"""
Daily features for every pair that passed the stability check. This is the
training data for the model in model.py.

Pair-level features:
- spread: log(A) - hedge_ratio * log(B), using the fixed hedge ratio from the
  stability check. I don't refit it daily, otherwise the spread itself would
  keep changing meaning.
- zscore: spread vs its rolling mean and std
- spread_vol: rolling std of the spread's daily change. Different from the
  z-score denominator, which is the std of the spread level.

Per-stock features (computed for each leg, suffixed _a / _b):
- volume_ratio: today's volume over its N-day average
- atr_pct: ATR divided by close, so stocks at different price levels compare
- rel_strength_market / rel_strength_sector: N-day return minus the return of
  0050 / 00892. Helps tell a spread move caused by the two stocks themselves
  from one caused by the whole market or sector moving.

Usage:
    python src/features.py

Input:  output/stability_report.csv (is_stable == True only), data/prices_raw.parquet
Output: output/features_dataset.csv, one row per pair per day
"""

import os
import numpy as np
import pandas as pd

from config import (
    RAW_DATA_PATH,
    STABILITY_RESULT_PATH,
    FEATURE_DATASET_PATH,
    STOCK_UNIVERSE,
    BENCHMARK_TICKERS,
    ZSCORE_WINDOW_DAYS,
    VOLUME_AVG_WINDOW_DAYS,
    ATR_WINDOW_DAYS,
    RELATIVE_STRENGTH_WINDOW_DAYS,
)


def load_all_wide_prices():
    """Close/high/low/volume tables, one column per ticker, stocks and benchmarks.
    This reads every date, not just the train period. That's fine because each
    day's feature only uses data up to that day. The look-ahead risk is in
    choosing pairs and hedge ratios, and cointegration.py already restricts
    that to the train period."""
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df.sort_values(["stock_id", "date"])

    def to_wide(col):
        wide = df.pivot(index="date", columns="stock_id", values=col)
        return wide.sort_index().ffill()

    return to_wide("close"), to_wide("high"), to_wide("low"), to_wide("volume")


def compute_atr_pct(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """ATR as a fraction of the close."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr / close


def compute_relative_strength(stock_close: pd.Series, benchmark_close: pd.Series, window: int) -> pd.Series:
    """N-day log return of the stock minus N-day log return of the benchmark."""
    stock_return = np.log(stock_close) - np.log(stock_close.shift(window))
    benchmark_return = np.log(benchmark_close) - np.log(benchmark_close.shift(window))
    return stock_return - benchmark_return


def compute_stock_features(stock_id: str, close_wide, high_wide, low_wide, volume_wide) -> pd.DataFrame:
    """Per-stock features for one ticker."""
    close = close_wide[stock_id]
    high = high_wide[stock_id]
    low = low_wide[stock_id]
    volume = volume_wide[stock_id]

    volume_ratio = volume / volume.rolling(
        window=VOLUME_AVG_WINDOW_DAYS, min_periods=VOLUME_AVG_WINDOW_DAYS
    ).mean()

    atr_pct = compute_atr_pct(close, high, low, ATR_WINDOW_DAYS)

    rel_strength_market = compute_relative_strength(
        close, close_wide[list(BENCHMARK_TICKERS.keys())[0]], RELATIVE_STRENGTH_WINDOW_DAYS
    )
    rel_strength_sector = compute_relative_strength(
        close, close_wide[list(BENCHMARK_TICKERS.keys())[1]], RELATIVE_STRENGTH_WINDOW_DAYS
    )

    return pd.DataFrame({
        "volume_ratio": volume_ratio,
        "atr_pct": atr_pct,
        "rel_strength_market": rel_strength_market,
        "rel_strength_sector": rel_strength_sector,
    })


def compute_pair_features(dependent_id: str, independent_id: str, hedge_ratio: float, close_wide) -> pd.DataFrame:
    """spread / zscore / spread_vol for one pair, plus the rolling mean and std
    behind the z-score."""
    log_dep = np.log(close_wide[dependent_id])
    log_indep = np.log(close_wide[independent_id])
    spread = log_dep - hedge_ratio * log_indep

    rolling_mean = spread.rolling(window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).mean()
    rolling_std = spread.rolling(window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).std()
    zscore = (spread - rolling_mean) / rolling_std

    spread_vol = spread.diff().rolling(
        window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS
    ).std()

    # spread_mean and spread_std aren't model features. model.py and backtest.py
    # use them as frozen anchors at entry. The rolling z-score has a problem: if
    # the spread jumps to a new level and stays there, the rolling mean catches
    # up within days and z drifts back to zero, which looks like a reversion
    # even though the spread never went back. I ran into this on synthetic data,
    # where almost every trade looked like a win. Measuring reversion against
    # the mean and std from the entry day fixes it.
    return pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "spread_vol": spread_vol,
        "spread_mean": rolling_mean,
        "spread_std": rolling_std,
    })


def build_feature_dataset() -> pd.DataFrame:
    stability = pd.read_csv(STABILITY_RESULT_PATH)
    stable_pairs = stability[stability["is_stable"]].reset_index(drop=True)

    if stable_pairs.empty:
        raise ValueError("No pair in stability_report.csv has is_stable=True, "
                          "so there is nothing to build features for. Check the previous two steps.")

    close_wide, high_wide, low_wide, volume_wide = load_all_wide_prices()

    # Several pairs can share a stock, so cache per-stock features instead of
    # recomputing them for every pair.
    stock_feature_cache = {}

    def get_stock_features(stock_id: str) -> pd.DataFrame:
        if stock_id not in stock_feature_cache:
            stock_feature_cache[stock_id] = compute_stock_features(
                stock_id, close_wide, high_wide, low_wide, volume_wide
            )
        return stock_feature_cache[stock_id]

    all_frames = []
    for _, row in stable_pairs.iterrows():
        dependent_id, independent_id = row["direction"].split("~")
        hedge_ratio = row["hedge_ratio_mean"]  # mean of the rolling-window estimates from the stability check

        pair_feat = compute_pair_features(dependent_id, independent_id, hedge_ratio, close_wide)
        feat_a = get_stock_features(dependent_id).add_suffix("_a")
        feat_b = get_stock_features(independent_id).add_suffix("_b")

        merged = pd.concat([pair_feat, feat_a, feat_b], axis=1)
        merged = merged.dropna(how="any")  # drop the warm-up days where rolling windows aren't full yet

        merged.insert(0, "pair_id", f"{dependent_id}_{independent_id}")
        merged.insert(1, "stock_a", dependent_id)
        merged.insert(2, "stock_b", independent_id)
        merged.insert(3, "hedge_ratio", hedge_ratio)  # the backtest needs it to size the second leg
        merged = merged.reset_index().rename(columns={"date": "date"})

        all_frames.append(merged)

    return pd.concat(all_frames, ignore_index=True)


def main():
    feature_df = build_feature_dataset()
    os.makedirs(os.path.dirname(FEATURE_DATASET_PATH) or ".", exist_ok=True)
    feature_df.to_csv(FEATURE_DATASET_PATH, index=False, encoding="utf-8-sig")

    n_pairs = feature_df["pair_id"].nunique()
    print(f"{n_pairs} stable pairs, {len(feature_df)} daily feature rows")
    print(feature_df.head(5).to_string(index=False))
    print(f"\nSaved to {FEATURE_DATASET_PATH}")


if __name__ == "__main__":
    main()
