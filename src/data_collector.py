# -*- coding: utf-8 -*-
"""
Downloads daily price and volume from FinMind for the stock universe plus the
two benchmark ETFs, and saves everything to one long-format parquet file.

Usage:
    export FINMIND_API_TOKEN="your token"
    python src/data_collector.py

Prices are adjusted for ex-dividend / ex-rights events. FinMind's TaiwanStockPrice
is raw (an ex-dividend drop shows up as a price fall), and the ready-made adjusted
dataset needs a paid FinMind level. So I build the adjustment from the free
TaiwanStockDividendResult table: on each ex-date the factor is
reference_price / before_price, and every price before that date is multiplied by it.

Output: data/prices_raw.parquet
Columns: date, stock_id, stock_name, close, high, low (adjusted), volume (split-adjusted),
         close_raw and volume_raw (as traded, used for traded value in the liquidity cap)
"""

import os
import time
import numpy as np
import pandas as pd
import requests

from config import STOCK_UNIVERSE, BENCHMARK_TICKERS, START_DATE, END_DATE, RAW_DATA_PATH, KNOWN_SPLITS

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_stock_price(stock_id: str, token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily bars for one ticker (stock or ETF) from the TaiwanStockPrice dataset."""
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != 200 or not payload.get("data"):
        print(f"  [warning] no data for {stock_id}, status: {payload.get('status')}, "
              f"message: {payload.get('msg')}")
        return pd.DataFrame()

    df = pd.DataFrame(payload["data"])
    # FinMind returns: date, stock_id, Trading_Volume, Trading_money, open, max,
    # min, close, spread, Trading_turnover. I only need close, volume, and the
    # daily high/low (max/min), which feed the ATR feature.
    df = df[["date", "stock_id", "close", "max", "min", "Trading_Volume"]].rename(
        columns={"max": "high", "min": "low", "Trading_Volume": "volume"}
    )
    return df


def fetch_adjustment_factors(stock_id: str, token: str, start_date: str, end_date: str) -> pd.Series:
    """Ex-date -> adjustment factor (reference_price / before_price) for one ticker."""
    params = {
        "dataset": "TaiwanStockDividendResult",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": token,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200 or not payload.get("data"):
        return pd.Series(dtype=float)
    d = pd.DataFrame(payload["data"])
    d["date"] = pd.to_datetime(d["date"])
    d["factor"] = d["reference_price"] / d["before_price"]
    d = d[(d["factor"] > 0) & (d["factor"] <= 1.5)].dropna(subset=["factor"])
    return d.groupby("date")["factor"].last().sort_index()


def apply_adjustment(df: pd.DataFrame, factors: pd.Series, splits: list | None = None) -> pd.DataFrame:
    """Backward-adjust close/high/low. A row dated before an ex-date is multiplied
    by that ex-date's factor; several events multiply together. Rows on or after
    the latest ex-date stay as traded."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["close_raw"] = df["close"]
    df["volume_raw"] = df["volume"]
    dates = df["date"].to_numpy()
    vol_mult = np.ones(len(df))
    for day, f in (splits or []):          # splits: prices x f, volume / f before the split day
        before = dates < np.datetime64(pd.Timestamp(day))
        vol_mult[before] /= f
        factors = pd.concat([factors, pd.Series([f], index=[pd.Timestamp(day)])]).groupby(level=0).prod()
    df["volume"] = df["volume"] * vol_mult
    if factors.empty:
        return df
    mult = np.ones(len(df))
    for ex_date, f in factors.items():
        mult[dates < np.datetime64(ex_date)] *= f
    for col in ("close", "high", "low"):
        df[col] = df[col] * mult
    return df


def collect_all(token: str) -> pd.DataFrame:
    """Loop over the universe plus benchmarks and stack into one long table.
    Benchmarks are only for the relative-strength features, never used as pair candidates."""
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    all_tickers = {**STOCK_UNIVERSE, **BENCHMARK_TICKERS}

    all_frames = []
    for stock_id, stock_name in all_tickers.items():
        print(f"Fetching {stock_id} {stock_name} ...")
        df = fetch_stock_price(stock_id, token, start_str, end_str)
        if not df.empty:
            factors = fetch_adjustment_factors(stock_id, token, start_str, end_str)
            time.sleep(0.6)
            if factors.empty:
                print(f"  [warning] no ex-dividend/ex-rights records for {stock_id}; left unadjusted")
            df = apply_adjustment(df, factors, KNOWN_SPLITS.get(stock_id))
            print(f"  {len(factors)} ex-date adjustments")
            df["stock_name"] = stock_name
            all_frames.append(df)
        # the free tier is rate limited, so pause between requests
        time.sleep(0.6)

    if not all_frames:
        raise RuntimeError("No data came back for any ticker. Check the API token and the network connection.")

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Sanity check: a one-day move above 30% after adjusting usually means a split
    # or capital reduction that the dividend table doesn't cover.
    ret = result[result["close"] > 0].groupby("stock_id")["close"].pct_change().abs()
    jumps = result.loc[ret[ret > 0.30].index, ["stock_id", "date", "close"]]
    if not jumps.empty:
        print("\n[warning] one-day moves above 30% after adjustment (possible uncovered split):")
        print(jumps.to_string(index=False))
    return result


def main():
    token = os.environ.get("FINMIND_API_TOKEN", "")
    if not token:
        print("[note] FINMIND_API_TOKEN is not set. "
              "The free tier works without a token, but request count and history length are limited.")

    df = collect_all(token)
    os.makedirs(os.path.dirname(RAW_DATA_PATH) or ".", exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, index=False)
    print(f"\nDone: {df['stock_id'].nunique()} tickers, "
          f"{len(df)} rows saved to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
