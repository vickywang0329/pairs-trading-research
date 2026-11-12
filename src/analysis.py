# -*- coding: utf-8 -*-
"""
Three checks on the backtest in backtest.py. None of them tunes anything: the
rules, thresholds and model stay as they are, and I only look at the result
from different angles.

1. Random control. The model version trades far fewer signals than the
   baseline, so it also pays far fewer costs. That alone can make it look
   better even if the model has no skill. To separate the two, I draw random
   subsets of the baseline's signals at the same selection rate as the model
   (RANDOM_CONTROL_RUNS times) and see where the model's result falls among
   them.

2. Cost sensitivity. Rerun both versions with different one-way costs. This
   shows how much of the result is the strategy and how much is the cost
   assumption, and roughly where it breaks even.

3. Capacity. Rerun with larger starting capital under the liquidity cap
   (each leg limited to LIQUIDITY_CAP_PCT of average daily traded value). With
   1M TWD the cap hardly matters, so this shows how performance and the share
   of constrained trades change as the account grows. It limits participation
   but does not model market impact, so it flatters large sizes.

Usage:
    python src/analysis.py

Input:  output/predictions.csv, output/features_dataset.csv, data/prices_raw.parquet
Output: output/analysis_report.txt, output/analysis_plots.png
"""

import os
import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    MODEL_PROB_THRESHOLD,
    MODEL_RANDOM_SEED,
    LIQUIDITY_CAP_PCT,
    COST_SENSITIVITY_PCTS,
    CAPACITY_LEVELS,
    RANDOM_CONTROL_RUNS,
    ANALYSIS_REPORT_PATH,
    ANALYSIS_PLOT_PATH,
)
from backtest import load_inputs, run_backtest, compute_metrics


def _run(pair_tables, preds, dates, adv, use_model, capital=INITIAL_CAPITAL, cost_pct=None, liq=True):
    kwargs = {} if cost_pct is None else {"cost_pct": cost_pct}
    trades, curve = run_backtest(
        pair_tables, preds, dates, use_model=use_model, capital=capital,
        adv=adv if liq else None, liq_cap_pct=LIQUIDITY_CAP_PCT if liq else None, **kwargs
    )
    return compute_metrics(trades, curve, capital)


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.1%}"


def _num(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def random_control(pair_tables, preds, dates, adv, model_metrics):
    rate = float((preds["prob_revert"] >= MODEL_PROB_THRESHOLD).mean())
    rets, sharpes, ntrades = [], [], []
    for i in range(RANDOM_CONTROL_RUNS):
        sub = preds.sample(frac=rate, random_state=MODEL_RANDOM_SEED + i)
        m = _run(pair_tables, sub, dates, adv, use_model=False)
        rets.append(m.get("total_return", np.nan))
        sharpes.append(m.get("sharpe", np.nan))
        ntrades.append(m.get("n_trades", 0))
    return rate, np.array(rets, float), np.array(sharpes, float), np.array(ntrades, float)


def breakeven_cost(costs, rets):
    """Cost at which total return crosses zero, by linear interpolation between
    the tested costs. None if it never crosses in the tested range."""
    for (c0, r0), (c1, r1) in zip(zip(costs[:-1], rets[:-1]), zip(costs[1:], rets[1:])):
        if np.isfinite(r0) and np.isfinite(r1) and r0 > 0 >= r1:
            return c0 + (c1 - c0) * r0 / (r0 - r1)
    return None


def build_analysis():
    pair_tables, preds, dates, adv = load_inputs()
    if not dates:
        raise ValueError("No trading days in the test period. Check TRAIN_TEST_SPLIT_DATE and the data range.")
    lines = []

    # headline
    m_model = _run(pair_tables, preds, dates, adv, use_model=True)
    m_base = _run(pair_tables, preds, dates, adv, use_model=False)
    lines.append("=== Headline (same as backtest_report.txt) ===")
    lines.append(f"{'':<24}{'With model':>12}{'No model':>12}")
    for label, key, fmt in [("Trades", "n_trades", "int"), ("Total return", "total_return", "pct"),
                            ("P&L before costs", "gross_pnl_pct", "pct"), ("Costs / capital", "cost_pct_of_capital", "pct"),
                            ("Sharpe (rf = 0)", "sharpe", "num")]:
        f = (lambda v: str(v) if v is not None else "n/a") if fmt == "int" else (_pct if fmt == "pct" else _num)
        lines.append(f"{label:<24}{f(m_model.get(key)):>12}{f(m_base.get(key)):>12}")

    # 1. random control
    rate, rets, sharpes, ntr = random_control(pair_tables, preds, dates, adv, m_model)
    mr = m_model.get("total_return", np.nan)
    ms = m_model.get("sharpe", np.nan)
    lines.append("")
    lines.append(f"=== 1. Random control ({RANDOM_CONTROL_RUNS} random subsets of the baseline's signals) ===")
    lines.append(f"The model keeps {rate:.1%} of signals (probability >= {MODEL_PROB_THRESHOLD}); each random run keeps the same share.")
    lines.append(f"Random subsets: {np.nanmean(ntr):.1f} trades on average, total return mean {_pct(np.nanmean(rets))}, "
                 f"std {_pct(np.nanstd(rets))}, 5th-95th percentile {_pct(np.nanpercentile(rets, 5))} to {_pct(np.nanpercentile(rets, 95))}")
    pr = float(np.mean(rets < mr)) if np.isfinite(mr) else np.nan
    lines.append(f"Model total return {_pct(mr)}: better than {_pct(pr)} of the random subsets")
    if np.isfinite(ms) and np.isfinite(sharpes).any():
        ps = float(np.mean(sharpes[np.isfinite(sharpes)] < ms))
        lines.append(f"Model Sharpe {_num(ms)}: better than {_pct(ps)} of the random subsets")
    lines.append("If the model's rank is not clearly in the top few percent, its edge over the baseline is mostly "
                 "the effect of trading less, not of picking better trades.")

    # 2. cost sensitivity
    lines.append("")
    lines.append("=== 2. Cost sensitivity (one-way cost per leg, charged at entry and exit) ===")
    lines.append(f"{'Cost':<8}{'Model ret':>11}{'Model Sharpe':>14}{'Base ret':>11}{'Base Sharpe':>13}")
    cm, cb = [], []
    for c in COST_SENSITIVITY_PCTS:
        a = _run(pair_tables, preds, dates, adv, True, cost_pct=c)
        b = _run(pair_tables, preds, dates, adv, False, cost_pct=c)
        cm.append(a.get("total_return", np.nan)); cb.append(b.get("total_return", np.nan))
        lines.append(f"{c:<8.2%}{_pct(cm[-1]):>11}{_num(a.get('sharpe')):>14}{_pct(cb[-1]):>11}{_num(b.get('sharpe')):>13}")
    for name, series in (("model", cm), ("baseline", cb)):
        be = breakeven_cost(COST_SENSITIVITY_PCTS, series)
        if be is not None:
            lines.append(f"Break-even one-way cost, {name}: about {be:.2%}")
        elif series and np.isfinite(series[0]) and series[0] <= 0:
            lines.append(f"{name.capitalize()} loses money even at zero cost")
        else:
            lines.append(f"{name.capitalize()} stays positive across the tested costs")
    lines.append("Retail all-in cost in Taiwan is roughly 0.4% one-way (commission, 0.3% sell-side tax, slippage); "
                 "institutions pay far less.")

    # 3. capacity
    lines.append("")
    lines.append(f"=== 3. Capacity (each leg capped at {LIQUIDITY_CAP_PCT:.0%} of average daily traded value) ===")
    lines.append(f"{'Capital (TWD)':<16}{'Ret model':>11}{'Ret base':>10}{'Trades b':>10}{'Constrained':>13}{'Avg size kept':>15}")
    for cap in CAPACITY_LEVELS:
        a = _run(pair_tables, preds, dates, adv, True, capital=cap)
        b = _run(pair_tables, preds, dates, adv, False, capital=cap)
        lines.append(f"{cap:<16,}{_pct(a.get('total_return')):>11}{_pct(b.get('total_return')):>10}"
                     f"{b.get('n_trades', 0):>10}{_pct(b.get('share_liq_constrained')):>13}{_pct(b.get('avg_liq_scale')):>15}")
    lines.append("A level with 0 trades means the liquidity cap shrank every position below the 0.5%-of-equity minimum "
                 "size, i.e. the strategy can't be run at that size.")
    lines.append("'Constrained' is the share of baseline trades cut down by the liquidity cap; 'Avg size kept' is the "
                 "average fraction of the intended size that was allowed. Market impact is not modelled.")

    return "\n".join(lines), {"rets": rets, "model_ret": mr, "costs": COST_SENSITIVITY_PCTS, "cm": cm, "cb": cb}


def plot(data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(data["rets"][np.isfinite(data["rets"])] * 100, bins=25, color="lightgray", edgecolor="gray")
    if np.isfinite(data["model_ret"]):
        ax.axvline(data["model_ret"] * 100, color="tab:blue", linewidth=2, label="Model")
    ax.set_title("Random control: total return (%)")
    ax.set_xlabel("Total return, %")
    ax.legend()
    ax = axes[1]
    c = np.array(data["costs"]) * 100
    ax.plot(c, np.array(data["cm"]) * 100, marker="o", label="With model")
    ax.plot(c, np.array(data["cb"]) * 100, marker="o", linestyle="--", label="No model")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Total return vs one-way cost")
    ax.set_xlabel("One-way cost per leg, %")
    ax.set_ylabel("Total return, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    report, data = build_analysis()
    os.makedirs(os.path.dirname(ANALYSIS_REPORT_PATH) or ".", exist_ok=True)
    with open(ANALYSIS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    plotted = plot(data, ANALYSIS_PLOT_PATH)
    print(report)
    print(f"\nReport: {ANALYSIS_REPORT_PATH}" + (f"; plots: {ANALYSIS_PLOT_PATH}" if plotted else " (matplotlib not installed, plots skipped)"))


if __name__ == "__main__":
    main()

