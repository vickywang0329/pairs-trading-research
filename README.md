# 台股半導體配對交易 / 統計套利專案

## 專案定位

本專案從 **Engle-Granger 協整檢定**的雙變量配對交易出發，鎖定台股半導體產業，
驗證核心方法論後，計畫進一步擴充為 **Johansen 檢定**的多變量統計套利，
比較兩種方法在實際績效上的差異。

## 目前進度：階段一 / 步驟 1-3（資料蒐集 + 協整檢定 + 特徵工程）

- 股票池：14 檔半導體股票，涵蓋晶圓代工、IC設計、封測、記憶體、矽晶圓材料五個子產業
  （見 `src/config.py` 的 `STOCK_UNIVERSE`）
- 基準指數：大盤代理（0050）+ 半導體產業代理（00892），只用來算個股相對強弱特徵，
  不參與協整配對
- 資料期間：近 4 年日線，前 3 年作為訓練期（協整篩選 + 之後的模型訓練），
  最後 1 年保留做樣本外回測，避免用到未來資訊
- 協整檢定：對股票池做兩兩配對的 Engle-Granger 檢定，**雙方向都測**
  （分別以 A 對 B、B 對 A 迴歸，取 p-value 較小者），並額外估計
  避險比率(hedge ratio)與價差的均值回歸半衰期(half-life)，篩選條件是
  p-value < 0.05 **且**半衰期落在5-60天區間
- 穩定性複查：用滾動視窗（120天一段、每次移動20天）重新估計避險比率與
  均值回歸速度(theta)，篩掉統計上顯著協整、但參數本身劇烈漂移的配對
- 特徵工程：對通過穩定性複查的配對計算逐日特徵，包含價差/z-score/
  價差波動度，以及兩個leg各自的成交量相對強度/ATR波動度/相對大盤與
  產業的相對強弱

## 使用方式

```bash
pip install -r requirements.txt

# 1. 蒐集資料（需要 FinMind API token，免費註冊即可取得）
export FINMIND_API_TOKEN="你的token"
cd src
python data_collector.py

# 2. 跑協整檢定，篩出候選配對
python cointegration.py

# 3. 對候選配對做穩定性複查（滾動視窗重估避險比率與theta）
python stability_check.py

# 4. 對通過穩定性複查的配對做特徵工程
python features.py
```

`cointegration.py` 輸出結果在 `output/cointegration_candidates.csv`，
每一列是一組候選配對，包含 p-value、避險比率、半衰期，已依 p-value 由小到大排序。

`stability_check.py` 輸出結果在 `output/stability_report.csv`，對每一組
通過協整檢定的候選配對，重新估計避險比率與theta，計算兩個穩定性指標：

- `hedge_ratio_cv`：避險比率在不同視窗間的變異係數，超過0.30視為不穩定
- `theta_nonrevert_ratio`：視窗中theta≥0（喪失均值回歸現象）的比例，超過0.20視為不穩定

`is_stable` 欄位是兩個指標都通過才標記為True。這一步是為了處理
「協整檢定跟半衰期都是用整段訓練期估一次、假設參數不變」這個簡化假設
的已知限制——如果一組配對統計上顯著協整，但避險比率或theta在不同期間
劇烈漂移，代表這組協整關係可能正在弱化、或已經因為基本面變化而失效，
不該只看p-value跟半衰期就直接拿去交易。

`features.py` 只對 `is_stable == True` 的配對計算特徵，輸出結果在
`output/features_dataset.csv`，每一列是「某配對、某天」的一筆特徵紀錄，
包含配對層級特徵（`spread`、`zscore`、`spread_vol`）跟兩個leg各自的
個股層級特徵（`volume_ratio`、`atr_pct`、`rel_strength_market`、
`rel_strength_sector`，以 `_a`/`_b` 後綴區分）。這個檔案就是下一步
（梯度提升樹模型）的訓練資料來源。

## 下一步（階段一步驟 4-6，尚未開始）

4. 用梯度提升樹（LightGBM/XGBoost）預測價差回歸機率
5. 設定風控規則：停損、單一配對部位上限、總曝險上限
6. 模擬下單邏輯與績效追蹤（Sharpe ratio、最大回撤、勝率等）

## 階段二（進階延伸，尚未開始）

7. 把股票池擴大成 3-5 檔一組的籃子
8. 用 Johansen 檢定找出多股票的協整組合與權重
9. 調整特徵與風控邏輯以適應多股票部位管理
10. 比較 Engle-Granger 配對版 vs Johansen 籃子版的績效差異

## 檔案結構

```
taiwan-pairs-trading/
├── src/
│   ├── config.py          # 股票池、資料期間、檢定門檻、特徵參數
│   ├── data_collector.py  # 透過 FinMind API 抓取日線價量（含高低價、基準指數）
│   ├── cointegration.py   # 兩兩配對 Engle-Granger 檢定
│   ├── stability_check.py # 滾動視窗複查避險比率/theta穩定性
│   └── features.py        # 特徵工程：價差/z-score/波動度/相對強弱
├── data/                  # 存放原始價格資料（.parquet）
├── output/                # 存放協整/穩定性/特徵工程結果（.csv）
└── requirements.txt
```

## 技術筆記

- Engle-Granger 檢定的迴歸方向會影響結果（A 對 B 迴歸出的殘差，
  跟 B 對 A 迴歸出的殘差，ADF 檢定的 p-value 通常不同），
  這裡採用文獻上常見的作法：兩個方向都做，取較顯著的一組
- 半衰期是用 Ornstein-Uhlenbeck 過程的離散近似估計：
  對價差的一階差分對其落後一期的水準值做迴歸，
  係數的負倒數乘以 ln(2) 即為半衰期。半衰期會用來做篩選（見
  `MIN_HALF_LIFE_DAYS`、`MAX_HALF_LIFE_DAYS`），不只是計算後排序顯示：
  太短（預設5天）常是雜訊造成的假訊號，太長（預設60天）代表即使統計上
  顯著協整，實務上也可能超出可接受的持有期，兩者都會被排除
- 本專案採用 O-U 過程作為價差動態的簡化假設，用於估計均值回歸速度；
  該假設之限制（參數隨時間變化、可能存在結構性斷點）以
  `stability_check.py` 的滾動視窗重估來部分緩解，而非假設協整關係
  一經檢定通過就永遠成立
- `features.py` 計算價差時，用的是穩定性複查階段估計出的**固定**避險比率，
  不是每天重新迴歸——因為z-score要衡量的是「在這個既定關係下，價差目前
  偏離多少」，如果避險比率每天都重新估，z-score量到的就會變成「估計誤差
  的波動」而不是「價差本身的波動」，兩者意義不同
- ATR除以收盤價做成百分比（`atr_pct`），是為了讓不同股價水準的股票可以
  直接比較波動度——原始ATR是價格單位，台積電股價跟晶豪科股價量級差很多，
  直接比原始ATR沒有意義
