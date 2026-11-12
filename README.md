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

Phase 1 is written end to end. Here is where it stands on real data:

- **Static pair list (`cointegration.py` + `stability_check.py`): run on real data, negative result.**
  On 3 years of training data and 14 stocks, 1 pair passed screening. On 31 stocks and 7 years, 4 of 435
  pairs passed the Engle-Granger test (about 22 would be expected by chance alone) and none passed the
  stability check. The pair that looked fine on 3 years (Winbond / Nanya) did not survive the longer
  window. So one fixed pair list held for years is not supported by the data.
- **Walk-forward selection (`walkforward.py`): written in response, not yet run on real data.**
  It reselects pairs every quarter on the trailing year. The pipeline from there (model, backtest) was
  checked on synthetic data only: no same-day fills, entries only inside a pair's selection period, no
  duplicate trades on one stock pair, exposure caps, and a leak test (changing prices after a date
  leaves every earlier feature row byte-identical).

There are no model or backtest results on real data yet, so this README quotes no performance numbers.

## Pipeline

| Step | Module | What it does | Output |
|---|---|---|---|
| 1 | `data_collector.py` | Pulls daily price and volume for 31 semiconductor-chain stocks and 2 benchmark ETFs from the FinMind API | `data/prices_raw.parquet` |
| 2 | `walkforward.py` | Every quarter, runs Engle-Granger on the previous year for every pair, keeps the best few with a positive hedge ratio, and builds daily features for the period they are traded: spread, z-score, spread volatility, per-leg volume ratio, ATR, strength relative to the market and the sector | `output/features_dataset.csv`, `walkforward_pairs.csv` |
| 3 | `model.py` | Defines the reversion label, trains the boosted trees with a time-based split, reports out-of-sample metrics and predicted probabilities | `output/predictions.csv`, `model_report.txt` |
| 4 | `backtest.py` | Risk rules plus a day-by-day simulation, with-model vs no-model baseline | `output/trades.csv`, `equity_curve.csv`, `backtest_report.txt`, `equity_curve.png` |

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

# the static-pair experiment, optional:
python src/cointegration.py
python src/stability_check.py
```

All paths are relative to the project root, so running from the root or from `src/` gives the same result.

## Universe and data window

- 31 stocks across the Taiwan semiconductor chain: foundry (TSMC, UMC, Vanguard International
  Semiconductor, Powerchip), IC design and IP (MediaTek, Novatek, Realtek, Alchip, Faraday, Global
  Unichip, Silergy, Parade, ASMedia, Phison, eMemory), packaging and testing (ASE, Powertech, King
  Yuan, Sigurd, Greatek), memory (Nanya, Winbond, Macronix), silicon wafer (GlobalWafers,
  Sino-American Silicon, Formosa Sumco), IC substrate (Unimicron, Nan Ya PCB, Kinsus), equipment and
  materials (Grand Process Technology, Gudeng Precision). Full list in `config.py`.
- The first run on 14 of these stocks found only 1 pair that passed screening, too few to train a
  model, so I widened the universe. With 31 stocks there are 465 pairs to test, so about 23 would
  pass at p < 0.05 by chance alone. That's why the stability check and the out-of-sample test matter more than the p-value.
- Benchmarks: 0050 (market proxy) and 00892 (semiconductor sector proxy). They are only used for
  relative-strength features and never enter a pair.
- The window is fixed by `END_DATE` in `config.py` (not `date.today()`, so results are reproducible):
  8 years of daily bars, the first 7 for training and the last 1 held out as the out-of-sample test period.
  I started with 4 years, but 3 years of training data only produced one stable pair, so I lengthened it.

## Methodology notes

**Walk-forward pair selection**
Cointegration between two stocks is a statement about a stretch of history, and for these stocks it
doesn't last. So every `REBALANCE_EVERY_DAYS` (63) I screen all pairs on the previous
`FORMATION_WINDOW_DAYS` (252), keep pairs with a positive hedge ratio (a negative one means the
"spread" is two stocks moving opposite ways, which isn't a pairs trade), take the
`MAX_PAIRS_PER_PERIOD` (10) most significant, and trade them with the hedge ratio fixed from that
window until the next reselection. Formation uses only dates before the first trading day.
The p-value cap matters: with 435 pairs, about 5% pass by luck in every period, so the screen alone
proves nothing. What I check is whether the selected pairs then revert out of sample, and
`walkforward.py` prints how often a pair selected in one quarter is selected again in the next.

**Avoiding look-ahead bias (where pairs trading plus machine learning most often goes wrong)**
- Pair selection and hedge ratios only use data from before each trading period, and the last year is locked away as out-of-sample for the model. I tested this by changing prices after a date and checking that every earlier feature row stays identical.
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
- In practice the 15% per-pair cap usually binds before the 2% risk limit, so the actual loss at a stop-out is smaller than 2%.

**Does the model add value? Look at the control group.**
`backtest.py` runs a with-model version and a no-model baseline (same z-score rules and risk
limits, every signal is traded). If the model version isn't clearly better than the baseline, the
honest conclusion is that the model adds nothing.

## Known limitations

- Short-borrow availability and fees are not modelled. Shorting individual Taiwan stocks has quota and cost constraints.
- Gap risk through the stop is not modelled, so real stop-out losses can exceed the configured amount.
- Fills are at the close, with no intraday liquidity or market impact. The cost rate is a rough estimate.
- The test period is about a year with a limited number of trades, so the confidence intervals on Sharpe and win rate are wide. Only the broad direction and robustness mean anything.
- When a pair signals on many consecutive days, neighbouring labels are highly correlated, so the effective sample size is far smaller than the row count.
- Reselecting pairs each quarter means the same two stocks can come back in consecutive periods. The backtest treats them as one pair, so it never holds the same trade twice, but it doesn't model how correlated the open positions are with each other.

## Phase 2 (not started)

7. Extend to baskets of 3-5 stocks
8. Use the Johansen test to find multi-stock cointegrating combinations and their weights
9. Adapt features and risk rules to multi-asset position management
10. Compare Engle-Granger pairs against Johansen baskets on performance

## File structure

```
taiwan-pairs-trading/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── config.py           # universe, dates, test thresholds, feature/label/model/risk parameters
    ├── data_collector.py   # FinMind daily price and volume (incl. high/low and benchmark indices)
    ├── walkforward.py      # quarterly pair reselection + feature engineering (main flow)
    ├── cointegration.py    # pairwise Engle-Granger test + half-life filter (static-pair experiment, reused by walkforward)
    ├── stability_check.py  # rolling-window check of hedge ratio / theta stability (static-pair experiment)
    ├── features.py         # feature functions (reused by walkforward) and the static-pair feature build
    ├── model.py            # labels, gradient-boosted trees, out-of-sample evaluation
    └── backtest.py         # risk rules and simulated-trade backtest
```
