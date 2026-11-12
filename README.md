# Taiwan Semiconductor Pairs Trading / Statistical Arbitrage

I use the Engle-Granger cointegration test to pick pairs among Taiwan-listed semiconductor stocks,
train a gradient-boosted-tree model to predict whether a spread that has diverged will revert,
and backtest the result out of sample under human-set risk rules. The backtest is compared against
a baseline that uses the same rules without the model, to check whether the model adds anything.

## Scope

Phase 1 is a two-asset pairs strategy built on the **Engle-Granger** test. Once the core method
has been validated, Phase 2 extends it to multi-asset statistical arbitrage with the **Johansen**
test and compares the two approaches on realized performance.

## Status

Phase 1 is finished and has been run end to end on real data (FinMind, 2018-11 to 2026-11). The short
version of what it found: **a simple statistical-arbitrage strategy on Taiwan semiconductor stocks
does not show a reliable edge at retail trading costs.** Details and numbers are in [Results](#results).

Prices are adjusted for dividends and splits. FinMind's price series is raw, and its ready-made adjusted
series needs a paid level, so I built the adjustment from the free dividend table (see Known limitations).
I first ran everything on unadjusted prices and kept those numbers: adjusting changed the details but not
the conclusion (Results, finding 8). Not done: Phase 2. This is research code, not investment advice.

## Pipeline

| Step | Module | What it does | Output |
|---|---|---|---|
| 1 | `data_collector.py` | Pulls daily price and volume for 31 semiconductor-chain stocks and 2 benchmark ETFs from the FinMind API, adjusted for dividends and splits | `data/prices_raw.parquet` |
| 2 | `walkforward.py` | Every quarter, runs Engle-Granger on the previous year for every pair, keeps the best few with a positive hedge ratio, and builds daily features for the period they are traded: spread, z-score, spread volatility, per-leg volume ratio, ATR, strength relative to the market and the sector | `output/features_dataset.csv`, `walkforward_pairs.csv` |
| 3 | `model.py` | Defines the reversion label, trains the boosted trees with a time-based split, reports out-of-sample metrics and predicted probabilities | `output/predictions.csv`, `model_report.txt` |
| 4 | `backtest.py` | Risk rules plus a day-by-day simulation, with-model vs no-model baseline, with a liquidity cap on each leg | `output/trades.csv`, `equity_curve.csv`, `backtest_report.txt`, `equity_curve.png` |
| 5 | `analysis.py` | Random-subset control for the model, cost sensitivity, capacity under the liquidity cap | `output/analysis_report.txt`, `analysis_plots.png` |

The static-pair experiment that motivated `walkforward.py` is kept as its own path:
`cointegration.py` (one Engle-Granger screen over the whole train period), `stability_check.py`
(rolling re-estimation of hedge ratio and theta) and `features.py` (features for the pairs that
survive). `walkforward.py` reuses their functions.

## Usage

```bash
pip install -r requirements.txt
export FINMIND_API_TOKEN="your_token"      # Windows PowerShell: $env:FINMIND_API_TOKEN="your_token"

python src/data_collector.py
python src/walkforward.py
python src/model.py
python src/backtest.py
python src/analysis.py

# the static-pair experiment, optional:
python src/cointegration.py
python src/stability_check.py
```

All paths are relative to the project root, so running from the root or from `src/` gives the same result.
Run in GitHub Codespaces on Python 3.14 (pandas 3.1, statsmodels 0.15, LightGBM 4.7). A free FinMind
token is enough. Raw data and generated files under `data/` and `output/` are not committed (the
scripts recreate them); the reports and charts from my run are kept in `results/`. `END_DATE` is fixed
in `config.py`, so the same FinMind data gives the same results.

## Universe and data window

- 31 stocks across the Taiwan semiconductor chain: foundry (TSMC, UMC, Vanguard International
  Semiconductor, Powerchip), IC design and IP (MediaTek, Novatek, Realtek, Alchip, Faraday, Global
  Unichip, Silergy, Parade, ASMedia, Phison, eMemory), packaging and testing (ASE, Powertech, King
  Yuan, Sigurd, Greatek), memory (Nanya, Winbond, Macronix), silicon wafer (GlobalWafers,
  Sino-American Silicon, Formosa Sumco), IC substrate (Unimicron, Nan Ya PCB, Kinsus), equipment and
  materials (Grand Process Technology, Gudeng Precision). Full list in `config.py`.
- The first run on 14 of these stocks found only 1 pair that passed screening, too few to train a
  model, so I widened the universe. With 31 stocks there are 465 possible pairs. Each pair is tested in
  both regression directions and the better p-value is kept, so between 5% and 10% of pairs pass at
  p < 0.05 by chance alone, which is roughly 23 to 46 pairs. That's why the stability check and the
  out-of-sample test matter more than the p-value.
- Each screening run only uses stocks with a full price history in its window. Powerchip has a shorter
  history, so the first 8 walk-forward periods test 30 stocks (435 pairs) and the other 19 test 31
  stocks (465 pairs). The static 7-year screen also tests 30 stocks (435 pairs).
- Benchmarks: 0050 (market proxy) and 00892 (semiconductor sector proxy). They are only used for
  relative-strength features and never enter a pair.
- The window is fixed by `END_DATE` in `config.py` (not `date.today()`, so results are reproducible):
  8 years of daily bars, the first 7 for training and the last 1 held out as the out-of-sample test period.
  I started with 4 years, but 3 years of training data only produced one stable pair, so I lengthened it.

## Results

All numbers below are from the real-data run. The out-of-sample test period is the last year of the
data (2025-11-13 to 2026-11-12), which no step used for selecting pairs, features or training. The
raw outputs are in [`results/`](results/).

**1. A fixed pair list does not hold up.** On 3 years of data and 14 stocks, 1 pair passed screening.
On 7 years, 4 of the 435 pairs (30 stocks, Powerchip dropped for short history) passed both the
Engle-Granger p-value and the half-life screen. Chance alone would pass roughly 22 to 44 pairs on the
p-value, so the screen found nothing beyond noise, and none passed the stability check (hedge-ratio coefficient of variation
0.6 to 0.8 against a 0.30 limit). The one pair that looked fine on 3 years (Winbond / Nanya) did not
survive the longer window.

**2. Reselecting pairs every quarter finds pairs that do not last.** `walkforward.py` selected 10 pairs
in nearly every period (the cap was almost always binding). Only 6.7% of the pairs selected in one quarter
were selected again in the next (about 2% would be expected from random picks, since 10 pairs are chosen out of 435 to 465). Five early periods were
lost because the sector ETF used for features only has data from 2021-06.

**3. The model has no predictive skill.** 1,025 training events and 228 out-of-sample events.
Out-of-sample AUC 0.513, walk-forward AUC inside the training set 0.466 (folds 0.42, 0.52, 0.46). Brier
score 0.2550 against 0.2380 for always predicting the mean, so it is worse than the baseline. The
calibration table is not monotonic.

**4. Backtest on the test year** (start capital 1,000,000 TWD, one-way cost 0.4%):

| | With model | No model (every signal) |
|---|---|---|
| Trades | 30 | 83 |
| Total return | -1.8% | -10.2% |
| Sharpe (rf = 0) | -0.29 | -0.95 |
| Max drawdown | -9.2% | -15.8% |
| P&L before costs | +1.6% | -1.3% |
| Costs / capital | 3.4% | 9.0% |
| Annual turnover | 9.3x | 24.8x |

**5. Random control.** The model trades far fewer signals, so it pays far fewer costs. To separate that
from skill I drew 300 random subsets of the baseline's signals at the same selection rate (22%). They
averaged -8.2% (5th to 95th percentile: -16.8% to +1.0%). The model's -1.8% beats 86% of them (Sharpe:
83%). That is not in the top few percent, so most of the gap between the model and the baseline is the
effect of trading less, not of picking better trades.

**6. Cost sensitivity.** The baseline loses money even at zero cost (-1.3%), so the raw rule has no edge
here before costs. The model is positive only below roughly 0.19% one-way cost, which is below what a
retail trader pays in Taiwan (about 0.4%, mostly the 0.3% sell-side tax plus commission and slippage).

| One-way cost | 0.0% | 0.1% | 0.2% | 0.3% | 0.4% | 0.5% |
|---|---|---|---|---|---|---|
| With model | +1.7% | +0.8% | -0.1% | -0.9% | -1.8% | -2.7% |
| No model | -1.3% | -3.6% | -5.8% | -8.1% | -10.2% | -12.3% |

**7. Liquidity is not the constraint at retail size.** With each leg capped at 5% of average daily
traded value, nothing is constrained at 1M or 10M TWD. For the baseline, 4.8% of trades are cut down at
50M TWD, 24% at 200M TWD and 70% at 1B TWD. Market impact is not modelled, so large sizes look better
than they would be.

**8. Effect of adjusting prices.** I ran the whole pipeline twice, on raw prices first and then on
dividend- and split-adjusted prices.

| | Raw prices | Adjusted prices |
|---|---|---|
| Pairs selected again next quarter | 9.5% | 6.7% |
| Out-of-sample AUC | 0.552 | 0.513 |
| Brier score (baseline for always predicting the mean) | 0.2413 (0.2411) | 0.2550 (0.2380) |
| Model trades / total return | 14 / +0.6% | 30 / -1.8% |
| No-model trades / total return | 84 / -10.9% | 83 / -10.2% |
| Model beats this share of random subsets | 94% | 86% |

The conclusion does not change. What disappears is the model's small positive return, which I had
already flagged as resting on 14 trades without support from AUC or Brier. I can't say how much of the
change comes from removing ex-dividend jumps and how much is the ordinary sensitivity of a result built
on a few trades to small changes in the data.

**What I take from it.** The method is sound but the opportunity, as I measured it, is not there for a
retail-cost trader, with raw or adjusted prices: relationships between these stocks drift faster than the strategy can use them,
and transaction costs take most of what is left. I would not call the model an
edge: on adjusted prices it loses money at realistic costs. Institutions do something more elaborate (factor-neutral residuals across hundreds of stocks,
very low costs), which is the natural next step.

## Methodology notes

**Walk-forward pair selection**
Cointegration between two stocks is a statement about a stretch of history, and for these stocks it
doesn't last. So every `REBALANCE_EVERY_DAYS` (63) I screen all pairs on the previous
`FORMATION_WINDOW_DAYS` (252), keep pairs with a positive hedge ratio (a negative one means the
"spread" is two stocks moving opposite ways, which isn't a pairs trade), take the
`MAX_PAIRS_PER_PERIOD` (10) most significant, and trade them with the hedge ratio fixed from that
window until the next reselection. Formation uses only dates before the first trading day.
The p-value cap matters: with 435 to 465 pairs, 5% to 10% pass by luck in every period (both directions are tested), so the screen alone
proves nothing. What I check is whether the selected pairs then revert out of sample, and
`walkforward.py` prints how often a pair selected in one quarter is selected again in the next.

**Avoiding look-ahead bias (where pairs trading plus machine learning most often goes wrong)**
- Pair selection and hedge ratios only use data from before each trading period, and the last year is locked away as out-of-sample for the model. On synthetic data I checked this by changing prices after a date and confirming every earlier feature row stayed identical.
- A label looks 30 days past the entry, so any training sample whose label window crosses the split date is dropped (purging).
- Inside the train set I also run walk-forward cross-validation. Hyperparameters are fixed and conservative; I never tune them on the test period.
- The backtest fills at the next day's close after the signal, not the close of the signal day.

**Frozen z-score (shared by labels and backtest)**
A rolling z-score's mean chases the spread. If the spread jumps to a new level and stays there (for
example a structural break from a change in fundamentals), the rolling mean catches up within days
and z drifts to zero on its own. That looks like a reversion even though the spread never went back
to its old equilibrium. So reversion and stop-outs are always measured against the mean and std
frozen on the entry-signal day. I found this while testing on synthetic data: the first version of
the labels came out about 100% successful, and only after the fix did they look sensible.

**Cointegration test and half-life**
- The Engle-Granger regression is not symmetric in direction, so I test both and keep the more significant one. I use statsmodels `coint` (with MacKinnon critical values).
- Half-life comes from a discrete approximation of the Ornstein-Uhlenbeck process. I keep 5-60 days: shorter is usually noise, longer is beyond the holding period I can accept.
- O-U is a simplification (fixed parameters, independent normal residuals). The rolling re-estimation in `stability_check.py` mitigates this only partly.

**Feature design**
- The per-leg stock features (volume ratio, ATR%, relative strength) are multiplied by the spread direction `side` so they line up. "Stock A is strong relative to the sector" is with-trend for a short-spread trade and against-trend for a long-spread trade, and trees have a hard time learning that from raw values.
- ATR is divided by the close so stocks at different price levels are comparable.

**Risk rules** (all parameters in `config.py`)
- Per-trade risk: position size is backed out so that a move to the stop line loses no more than 2% of equity.
- Gross notional per pair is capped at 15%, at most 5 pairs are held at once, and total gross exposure is capped at 70%.
- Exits: frozen z back inside 0.5 (reversion), out past 4.0 (stop), or 30 trading days held (timeout).
- 5-trading-day cooldown after a pair is closed; 0.4% one-way cost, charged on each leg at entry and at exit.
- In my synthetic tests the 15% per-pair cap usually bound before the 2% risk limit, so the loss at a stop-out was smaller than 2%.

**Checks that don't tune anything** (`analysis.py`)
Three views of the same backtest: a random-subset control (does the model beat random subsets taken at
the same selection rate?), cost sensitivity (how much of the result is the cost assumption?), and
capacity (how performance changes as the account grows, under a liquidity cap on each leg). The rules,
thresholds and model are not changed in response to what these show; the test year has been looked at,
so tuning on it would make it a training set.

**Does the model add value? Look at the control group.**
`backtest.py` runs a with-model version and a no-model baseline (same z-score rules and risk
limits, every signal is traded). If the model version isn't clearly better than the baseline, the
honest conclusion is that the model adds nothing.

## Known limitations

- **The price adjustment is my own.** FinMind's `TaiwanStockPrice` is raw: on 2330 the daily close change
  matched the API's own `spread` on every day except 17, and those 17 were the ex-dividend dates. The
  ready-made adjusted series needs a paid level, so `data_collector.py` multiplies each price before an
  ex-date by `reference_price / before_price` from `TaiwanStockDividendResult`. That table doesn't cover
  splits. A check for one-day moves above 30% found two (0050 on 2025-06-18 and 6415 on 2022-07-13, both
  1-for-4), which I added by hand in `KNOWN_SPLITS` after checking the raw price and volume. Smaller
  jumps from splits or capital reductions would not be caught by that check.
- The universe is the set of stocks listed today, so delisted names are missing (survivorship bias).
- Short-borrow availability and fees are not modelled. Shorting individual Taiwan stocks has quota and cost constraints.
- Gap risk through the stop is not modelled, so real stop-out losses can exceed the configured amount.
- Fills are at the close, with no intraday liquidity or market impact. The cost rate is a rough estimate.
- The test period is about a year with a limited number of trades, so the confidence intervals on Sharpe and win rate are wide. Only the broad direction and robustness mean anything.
- When a pair signals on many consecutive days, neighbouring labels are highly correlated, so the effective sample size is far smaller than the row count.
- Reselecting pairs each quarter means the same two stocks can come back in consecutive periods. The backtest treats them as one pair, so it never holds the same trade twice, but it doesn't model how correlated the open positions are with each other.

## Next steps (not started)

- Phase 2 as originally planned: baskets of 3-5 stocks with the Johansen test, compared against pairs.
  Given how unstable the pairs were, I expect baskets of stocks to have the same problem.
- A closer match to what institutions do: remove market and sector factors from a large universe
  (PCA or sector regressions) and trade the mean reversion of each stock's residual, with
  portfolio-level risk control. Needs a much larger universe than 31 stocks.

## File structure

```
pairs-trading-research/
├── README.md
├── requirements.txt
├── .gitignore
├── results/                # reports and charts from the real-data run
└── src/
    ├── config.py           # universe, dates, test thresholds, feature/label/model/risk parameters
    ├── data_collector.py   # FinMind daily price and volume (incl. high/low and benchmark indices)
    ├── walkforward.py      # quarterly pair reselection + feature engineering (main flow)
    ├── cointegration.py    # pairwise Engle-Granger test + half-life filter (static-pair experiment, reused by walkforward)
    ├── stability_check.py  # rolling-window check of hedge ratio / theta stability (static-pair experiment)
    ├── features.py         # feature functions (reused by walkforward) and the static-pair feature build
    ├── model.py            # labels, gradient-boosted trees, out-of-sample evaluation
    ├── backtest.py         # risk rules, liquidity cap and simulated-trade backtest
    └── analysis.py         # random control, cost sensitivity, capacity
```
