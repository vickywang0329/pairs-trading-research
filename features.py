# -*- coding: utf-8 -*-
"""
特徵工程（階段一步驟3）

對每一組通過協整檢定 + 穩定性複查的配對，計算逐日的特徵時間序列，
供下一步（階段一步驟4）的梯度提升樹模型使用。特徵分兩類：

1. 配對層級特徵（跟兩檔股票的組合有關）：
   - spread：對數價差，用穩定性複查時估計出的避險比率固定計算
     （不是每天重新迴歸，因為模型要學的正是「這個固定關係下，
       價差現在的狀態」，若每天重新估避險比率會讓spread本身失去意義）
   - zscore：價差用滾動視窗算出的z-score，衡量目前偏離正常水準多少個標準差
   - spread_vol：價差「日變動量」的滾動標準差，衡量價差本身最近的躁動程度
     （跟zscore的分母不同——zscore的分母是價差『水準值』的標準差，
       這裡是價差『變動量』的標準差，兩者捕捉的是不同層次的波動資訊）

2. 個股層級特徵（每檔股票各自獨立，跟配對哪一組無關）：
   - volume_ratio：當天成交量 / 過去N天平均成交量，抓量能異常放大/縮小
   - atr_pct：ATR（真實波動幅度均值）除以收盤價，做成不受股價水準影響的
     百分比波動度指標，才能跨股票比較
   - rel_strength_market：個股N天累積報酬 - 大盤(0050)N天累積報酬
   - rel_strength_sector：個股N天累積報酬 - 半導體產業ETF(00892)N天累積報酬
     （相對強弱是為了讓模型能分辨「價差擴大是因為這兩檔股票各自的
       特異走勢，還是單純被大盤/產業系統性風險牽動」，這兩種情況
       對價差是否會回歸，理論上有不同含義）

用法：
    python src/features.py

輸入：
    output/stability_report.csv （只取 is_stable == True 的配對）
    data/prices_raw.parquet

輸出：
    output/features_dataset.csv
    每一列是「某配對、某天」的一筆特徵紀錄，欄位包含 pair_id、
    spread/zscore/spread_vol，以及兩個leg分別的volume_ratio/atr_pct/
    rel_strength_market/rel_strength_sector（用 _a / _b 後綴區分）
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
    """讀取原始長格式資料（股票池 + 基準指數），轉成寬格式，
    回傳 close/high/low/volume 四張表，欄位是 stock_id，索引是日期。
    這裡刻意讀取『全部』日期（不限訓練期），因為特徵計算本身是
    因果的(causal)——每一天的特徵只用當天以前的資料算出來，
    本身不會造成look-ahead bias；look-ahead的風險是出在『用哪些資料
    去選配對、估避險比率』，那一步已經在cointegration.py限定只用
    訓練期處理過了，這裡沿用那個結果即可"""
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df.sort_values(["stock_id", "date"])

    def to_wide(col):
        wide = df.pivot(index="date", columns="stock_id", values=col)
        return wide.sort_index().ffill()

    return to_wide("close"), to_wide("high"), to_wide("low"), to_wide("volume")


def compute_atr_pct(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """計算ATR（真實波動幅度均值）並除以收盤價，做成百分比波動度，
    不同股價水準的股票才能直接比較"""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr / close


def compute_relative_strength(stock_close: pd.Series, benchmark_close: pd.Series, window: int) -> pd.Series:
    """個股N天累積對數報酬 - 基準指數N天累積對數報酬"""
    stock_return = np.log(stock_close) - np.log(stock_close.shift(window))
    benchmark_return = np.log(benchmark_close) - np.log(benchmark_close.shift(window))
    return stock_return - benchmark_return


def compute_stock_features(stock_id: str, close_wide, high_wide, low_wide, volume_wide) -> pd.DataFrame:
    """計算單一股票的個股層級特徵時間序列"""
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
    """計算單一配對的配對層級特徵時間序列（spread / zscore / spread_vol）"""
    log_dep = np.log(close_wide[dependent_id])
    log_indep = np.log(close_wide[independent_id])
    spread = log_dep - hedge_ratio * log_indep

    rolling_mean = spread.rolling(window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).mean()
    rolling_std = spread.rolling(window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).std()
    zscore = (spread - rolling_mean) / rolling_std

    spread_vol = spread.diff().rolling(
        window=ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS
    ).std()

    return pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "spread_vol": spread_vol,
    })


def build_feature_dataset() -> pd.DataFrame:
    stability = pd.read_csv(STABILITY_RESULT_PATH)
    stable_pairs = stability[stability["is_stable"]].reset_index(drop=True)

    if stable_pairs.empty:
        raise ValueError("stability_report.csv 裡沒有任何 is_stable=True 的配對，"
                          "沒有東西可以做特徵工程，請先確認前面兩步的輸出")

    close_wide, high_wide, low_wide, volume_wide = load_all_wide_prices()

    # 個股層級特徵只要算一次，不同配對如果用到同一檔股票可以重複使用，
    # 不用每組配對都重算
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
        hedge_ratio = row["hedge_ratio_mean"]  # 用穩定性複查裡滾動視窗估計的平均避險比率

        pair_feat = compute_pair_features(dependent_id, independent_id, hedge_ratio, close_wide)
        feat_a = get_stock_features(dependent_id).add_suffix("_a")
        feat_b = get_stock_features(independent_id).add_suffix("_b")

        merged = pd.concat([pair_feat, feat_a, feat_b], axis=1)
        merged = merged.dropna(how="any")  # 捨棄還在滾動視窗暖機期、算不出特徵的日子

        merged.insert(0, "pair_id", f"{dependent_id}_{independent_id}")
        merged.insert(1, "stock_a", dependent_id)
        merged.insert(2, "stock_b", independent_id)
        merged = merged.reset_index().rename(columns={"date": "date"})

        all_frames.append(merged)

    return pd.concat(all_frames, ignore_index=True)


def main():
    feature_df = build_feature_dataset()
    os.makedirs(os.path.dirname(FEATURE_DATASET_PATH) or ".", exist_ok=True)
    feature_df.to_csv(FEATURE_DATASET_PATH, index=False, encoding="utf-8-sig")

    n_pairs = feature_df["pair_id"].nunique()
    print(f"共 {n_pairs} 組穩定配對、{len(feature_df)} 筆逐日特徵紀錄")
    print(feature_df.head(5).to_string(index=False))
    print(f"\n已存到 {FEATURE_DATASET_PATH}")


if __name__ == "__main__":
    main()
