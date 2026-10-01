# -*- coding: utf-8 -*-
"""
Engle-Granger 協整檢定：對股票池做兩兩配對檢定，篩出候選配對

用法：
    python src/cointegration.py

輸出：
    output/cointegration_candidates.csv
    欄位：stock_a, stock_b, direction, p_value, hedge_ratio, half_life
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
    """讀取原始長格式價格資料，轉成寬格式(每檔股票一欄)的對數收盤價，
    並且只取訓練期（切分測試期之前）的資料，避免協整篩選偷看未來。
    只保留股票池內的標的，基準指數(0050/00892)不參與配對協整檢定，
    它們是留給 features.py 算相對強弱用的"""
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df[df["date"] < pd.Timestamp(TRAIN_TEST_SPLIT_DATE)]
    df = df[df["stock_id"].isin(STOCK_UNIVERSE.keys())]

    wide = df.pivot(index="date", columns="stock_id", values="close")
    wide = wide.sort_index().ffill().dropna(how="any")  # 對齊交易日、去除缺值
    log_price = np.log(wide)
    return log_price


def estimate_theta(spread: pd.Series) -> float:
    """估計均值回歸速度係數 theta（O-U過程離散化後的AR(1)迴歸係數）
    theta < 0 代表存在均值回歸現象，越負代表拉回的力道越強；
    theta >= 0 代表沒有觀察到均值回歸現象"""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_diff.index]

    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()
    return model.params.iloc[1]


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """用 Ornstein-Uhlenbeck 近似估計價差回歸半衰期（單位：交易日）
    半衰期越短，代表價差偏離均值後回歸的速度越快，越適合做交易策略"""
    theta = estimate_theta(spread)
    if theta >= 0:
        return np.inf  # 沒有均值回歸現象
    return -np.log(2) / theta


def test_pair(log_price: pd.DataFrame, stock_a: str, stock_b: str) -> dict | None:
    """對單一配對做雙向 Engle-Granger 檢定
    （EG 檢定的迴歸方向會影響結果，因此兩個方向都做，取 p-value 較小者，
     這是文獻上常見的作法，也是這個 MVP 相對嚴謹的地方）"""
    y = log_price[stock_a]
    x = log_price[stock_b]

    if len(y) < MIN_OBSERVATIONS:
        return None

    # 方向一：A ~ B
    score_ab, pvalue_ab, _ = coint(y, x)
    # 方向二：B ~ A
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

    # 用較顯著方向的迴歸，估計避險比率(hedge ratio)與價差
    X = add_constant(independent)
    hedge_model = OLS(dependent, X).fit()
    hedge_ratio = hedge_model.params.iloc[1]
    spread = dependent - hedge_ratio * independent

    hl = half_life_of_mean_reversion(spread)

    # 半衰期太短（可能是雜訊、假訊號）或太長（超出可接受持有期）都排除，
    # 不能只看p-value顯著就視為可交易的配對
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
    print(f"訓練期資料：{log_price.index.min().date()} ~ {log_price.index.max().date()}，"
          f"共 {len(log_price)} 個交易日、{len(stocks)} 檔股票")
    print(f"總共要檢定 {len(list(itertools.combinations(stocks, 2)))} 組配對\n")

    results = []
    for stock_a, stock_b in itertools.combinations(stocks, 2):
        r = test_pair(log_price, stock_a, stock_b)
        if r is not None:
            results.append(r)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # 半衰期越短、p-value 越小，代表配對品質越好，排在前面
        result_df = result_df.sort_values(
            ["p_value", "half_life_days"], na_position="last"
        ).reset_index(drop=True)
    return result_df


def main():
    result_df = run_all_pairs()
    os.makedirs(os.path.dirname(COINTEGRATION_RESULT_PATH) or ".", exist_ok=True)
    result_df.to_csv(COINTEGRATION_RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"篩選出 {len(result_df)} 組候選配對"
          f"（p < {COINTEGRATION_SIGNIFICANCE}，"
          f"半衰期介於 {MIN_HALF_LIFE_DAYS}~{MAX_HALF_LIFE_DAYS} 天）")
    if not result_df.empty:
        print(result_df.head(10).to_string(index=False))
    print(f"\n已存到 {COINTEGRATION_RESULT_PATH}")


if __name__ == "__main__":
    main()
