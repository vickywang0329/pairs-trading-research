# -*- coding: utf-8 -*-
"""
Walk-forward pair selection and feature building. This replaces the fixed pair
list from cointegration.py + stability_check.py + features.py in the main flow.

Why: on the full 7-year train period the static screen found almost nothing
stable (see README). Relationships between Taiwanese chip stocks drift, so
asking one hedge ratio to hold for years is too much. Instead:

1. Every REBALANCE_EVERY_DAYS (about a quarter), take the previous
   FORMATION_WINDOW_DAYS (about a year) and run the same Engle-Granger screen
   on every pair: both directions, p-value and half-life limits as before.
2. Keep pairs with a positive hedge ratio (a negative one means the "spread"
   is really a sum of two stocks moving opposite ways, which isn't a pairs
   trade), then only the MAX_PAIRS_PER_PERIOD most significant.
3. Trade those pairs until the next reselection, with the hedge ratio fixed
   from the formation window.

Nothing here looks past the day a period starts: formation uses
dates[t - FORMATION_WINDOW_DAYS : t], trading starts on dates[t].

Each (pair, period) gets its own pair_id like 2408_2344_P12, so model.py and
backtest.py treat it as a separate pair without changes to how they group.

Extra columns in the output:
- tradable: True inside the pair's selection period. The rows after it (LABEL_HORIZON_DAYS + 2 of them)
  exist only so labels and exits can see what happened to a trade entered late
  in the period. They are never entry days.
- form_p_value, form_half_life: how strongly the pair passed screening. The
  model can use them as features.

Usage:
    python src/walkforward.py

Input:  data/prices_raw.parquet
Output: output/features_dataset.csv, output/walkforward_pairs.csv
"""

import os
import itertools
import numpy as np
import pandas as pd

from config import (
    STOCK_UNIVERSE,
    FEATURE_DATASET_PATH,
    WALKFORWARD_PAIRS_PATH,
    FORMATION_WINDOW_DAYS,
    REBALANCE_EVERY_DAYS,
    MAX_PAIRS_PER_PERIOD,
    LABEL_HORIZON_DAYS,
)
from cointegration import test_pair
from features import load_all_wide_prices, compute_stock_features, compute_pair_features

LEAD_ROWS = 5                                # a few rows before the period so 5-day features aren't NaN on day one
TAIL_ROWS = LABEL_HORIZON_DAYS + 2           # rows after the period: label horizon, plus the next-day fill, plus one spare


def select_pairs(formation_log: pd.DataFrame) -> pd.DataFrame:
    """Engle-Granger screen on one formation window. Only stocks with a full
    window of prices take part, so a stock that listed mid-window is skipped
    until it has enough history."""
    stocks = [c for c in formation_log.columns if formation_log[c].notna().all()]
    rows = []
    for a, b in itertools.combinations(stocks, 2):
        r = test_pair(formation_log, a, b, min_obs=FORMATION_WINDOW_DAYS - 5)
        if r is None or r["hedge_ratio"] <= 0:
            continue
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["p_value", "half_life_days"]).reset_index(drop=True)
    return out.head(MAX_PAIRS_PER_PERIOD)


def build_walkforward_dataset():
    close_wide, high_wide, low_wide, volume_wide = load_all_wide_prices()
    stocks = [s for s in STOCK_UNIVERSE if s in close_wide.columns]
    log_close = np.log(close_wide[stocks])
    n = len(close_wide)
    F, R = FORMATION_WINDOW_DAYS, REBALANCE_EVERY_DAYS
    positions = np.arange(n)

    stock_feature_cache = {}

    def get_stock_features(stock_id: str) -> pd.DataFrame:
        if stock_id not in stock_feature_cache:
            stock_feature_cache[stock_id] = compute_stock_features(
                stock_id, close_wide, high_wide, low_wide, volume_wide
            )
        return stock_feature_cache[stock_id]

    frames, selections = [], []
    period = 0
    for t in range(F, n, R):
        period += 1
        formation = log_close.iloc[t - F:t]          # strictly before day t
        pairs = select_pairs(formation)
        start_date = close_wide.index[t].date()
        print(f"Period {period:02d} (trading from {start_date}): "
              f"{len(pairs)} pairs selected", flush=True)
        if pairs.empty:
            continue

        tradable = pd.Series((positions >= t) & (positions < t + R), index=close_wide.index)
        lo, hi = max(t - LEAD_ROWS, 0), min(t + R + TAIL_ROWS, n)

        for _, row in pairs.iterrows():
            dep, indep = row["direction"].split("~")
            pair_feat = compute_pair_features(dep, indep, row["hedge_ratio"], close_wide)
            feat_a = get_stock_features(dep).add_suffix("_a")
            feat_b = get_stock_features(indep).add_suffix("_b")

            merged = pd.concat([pair_feat, feat_a, feat_b], axis=1).iloc[lo:hi]
            merged = merged.dropna(how="any")
            if merged.empty:
                continue

            merged.insert(0, "pair_id", f"{dep}_{indep}_P{period:02d}")
            merged.insert(1, "stock_a", dep)
            merged.insert(2, "stock_b", indep)
            merged.insert(3, "hedge_ratio", row["hedge_ratio"])  # fixed from the formation window
            merged["period"] = period
            merged["form_p_value"] = row["p_value"]
            merged["form_half_life"] = row["half_life_days"]
            merged["tradable"] = tradable.loc[merged.index].to_numpy()
            frames.append(merged.reset_index())

            selections.append({
                "period": period, "trade_start": start_date, "stock_a": dep, "stock_b": indep,
                "p_value": row["p_value"], "hedge_ratio": row["hedge_ratio"],
                "half_life_days": row["half_life_days"],
            })

    if not frames:
        raise ValueError("No pair was selected in any period. Check the data range and the screening limits in config.py.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(selections)


def persistence(selections: pd.DataFrame) -> float:
    """Share of the pairs selected in one period that are selected again in the
    next one. High means relationships last at least a quarter or two. Near
    zero means the screen is mostly picking noise."""
    by_period = {
        p: {frozenset((a, b)) for a, b in zip(g["stock_a"], g["stock_b"])}
        for p, g in selections.groupby("period")
    }
    shares = []
    for p, pairs in by_period.items():
        nxt = by_period.get(p + 1)
        if nxt is not None and pairs:
            shares.append(len(pairs & nxt) / len(pairs))
    return float(np.mean(shares)) if shares else float("nan")


def main():
    feature_df, selections = build_walkforward_dataset()

    os.makedirs(os.path.dirname(FEATURE_DATASET_PATH) or ".", exist_ok=True)
    feature_df.to_csv(FEATURE_DATASET_PATH, index=False, encoding="utf-8-sig")
    selections.to_csv(WALKFORWARD_PAIRS_PATH, index=False, encoding="utf-8-sig")

    n_periods = selections["period"].nunique()
    print(f"\n{len(selections)} pair selections across {n_periods} periods "
          f"({len(selections) / n_periods:.1f} per period), {len(feature_df)} daily feature rows")
    print(f"Persistence (selected again next period): {persistence(selections):.1%}")
    print("\nMost frequently selected pairs:")
    pair_name = [" / ".join(sorted((a, b))) for a, b in zip(selections["stock_a"], selections["stock_b"])]
    freq = selections.groupby(pair_name).size().sort_values(ascending=False).head(8)
    print(freq.to_string())
    print(f"\nSaved to {FEATURE_DATASET_PATH} and {WALKFORWARD_PAIRS_PATH}")


if __name__ == "__main__":
    main()
