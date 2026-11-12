# -*- coding: utf-8 -*-
"""
Downloads daily price and volume from FinMind for the stock universe plus the
two benchmark ETFs, and saves everything to one long-format parquet file.

Usage:
    export FINMIND_API_TOKEN="your token"
    python src/data_collector.py

Output: data/prices_raw.parquet
Columns: date, stock_id, stock_name, close, high, low, volume
"""

import os
import time
import pandas as pd
import requests

from config import STOCK_UNIVERSE, BENCHMARK_TICKERS, START_DATE, END_DATE, RAW_DATA_PATH

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
            df["stock_name"] = stock_name
            all_frames.append(df)
        # the free tier is rate limited, so pause between requests
        time.sleep(0.6)

    if not all_frames:
        raise RuntimeError("No data came back for any ticker. Check the API token and the network connection.")

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)
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
