# -*- coding: utf-8 -*-
"""
資料蒐集模組：透過 FinMind API 抓取台股半導體股票日線資料

使用方式：
    export FINMIND_API_TOKEN="你的token"
    python src/data_collector.py

輸出：
    data/prices_raw.parquet
    欄位：date, stock_id, stock_name, close, high, low, volume
    （包含股票池14檔 + 大盤/產業基準指數2檔，共16個data_id）
"""

import os
import time
import pandas as pd
import requests

from config import STOCK_UNIVERSE, BENCHMARK_TICKERS, START_DATE, END_DATE, RAW_DATA_PATH

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_stock_price(stock_id: str, token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """呼叫 FinMind TaiwanStockPrice API，抓取單一股票（或ETF）的日線價量資料"""
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
        print(f"  [警告] {stock_id} 沒有取得資料，回應狀態：{payload.get('status')}, "
              f"訊息：{payload.get('msg')}")
        return pd.DataFrame()

    df = pd.DataFrame(payload["data"])
    # FinMind 欄位：date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover
    # max/min 是當天最高/最低價，拿來算ATR用
    df = df[["date", "stock_id", "close", "max", "min", "Trading_Volume"]].rename(
        columns={"max": "high", "min": "low", "Trading_Volume": "volume"}
    )
    return df


def collect_all(token: str) -> pd.DataFrame:
    """迴圈抓取股票池 + 基準指數，並合併成一張長格式(long-format)表
    （基準指數只用來算相對強弱特徵，不會被拿去做協整配對）"""
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    all_tickers = {**STOCK_UNIVERSE, **BENCHMARK_TICKERS}

    all_frames = []
    for stock_id, stock_name in all_tickers.items():
        print(f"抓取 {stock_id} {stock_name} ...")
        df = fetch_stock_price(stock_id, token, start_str, end_str)
        if not df.empty:
            df["stock_name"] = stock_name
            all_frames.append(df)
        # FinMind 免費額度有速率限制，禮貌性間隔避免被擋
        time.sleep(0.6)

    if not all_frames:
        raise RuntimeError("所有股票都沒有抓到資料，請檢查 API token 或網路連線")

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return result


def main():
    token = os.environ.get("FINMIND_API_TOKEN", "")
    if not token:
        print("[提醒] 未偵測到環境變數 FINMIND_API_TOKEN，"
              "免費額度沒有 token 也能跑，但請求次數/歷史長度會受限。")

    df = collect_all(token)
    os.makedirs(os.path.dirname(RAW_DATA_PATH) or ".", exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, index=False)
    print(f"\n完成，共 {df['stock_id'].nunique()} 檔股票、"
          f"{len(df)} 筆資料，已存到 {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
