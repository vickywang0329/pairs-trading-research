# -*- coding: utf-8 -*-
"""
Backtest with risk rules, run only on the out-of-sample test period.

It steps through the test days one at a time and each day does three things:
1. Check open positions and close any that should be closed.
2. Enter new trades from yesterday's signals, at today's close. The signal is
   computed from the signal day's close, so filling at that same close would
   quietly use information I wouldn't have had.
3. Mark everything to market at today's close for the equity curve.

Exit rules (same as the labels in model.py, so the model predicts what I trade):
revert (frozen z back inside EXIT_Z), stop (frozen z past STOP_Z), or timeout
after LABEL_HORIZON_DAYS. Frozen z is measured against the mean and std from the
signal day.

Risk rules:
- Position size is backed out of the stop: if the spread runs to the stop line I
  lose at most MAX_LOSS_PER_TRADE_PCT of equity.
- Gross notional per pair is capped at PER_PAIR_CAPITAL_PCT, total gross at
  MAX_GROSS_EXPOSURE_PCT, and at most MAX_POSITIONS pairs at once.
- After closing a pair it sits out for COOLDOWN_DAYS.
- Costs: TRANSACTION_COST_PCT per leg on entry and again on exit.

P&L: spread = ln(A) - hedge_ratio * ln(B). Leg A has notional N, leg B has
hedge_ratio * N on the other side. z > 0 means the spread is high, so short A and
long B. P&L = -side * N * (change in spread), a log-return approximation.

I run two versions side by side: one that only trades when the model's
probability clears the threshold, and one baseline with the same z-score rules
and risk limits that takes every signal. The gap between them is the only honest
measure of what the model adds. If it doesn't beat the baseline, it doesn't help.

What this does NOT model:
- short selling availability and borrow fees, which matter for Taiwan stocks
- overnight gaps through the stop, so a real stop can lose more than planned
- intraday liquidity or market impact (the cost rate is a rough guess)
- With about a year of test data and a modest number of trades, the performance
  numbers are weak evidence. I look at direction and robustness, not decimals.

Usage:
    python src/backtest.py

Input:  output/predictions.csv, output/features_dataset.csv
Output: output/trades.csv, output/equity_curve.csv, output/backtest_report.txt,
        output/equity_curve.png (skipped if matplotlib isn't installed)
"""

import os
import numpy as np
import pandas as pd

from config import (
    FEATURE_DATASET_PATH,
    PREDICTIONS_PATH,
    TRADES_PATH,
    EQUITY_CURVE_PATH,
    BACKTEST_REPORT_PATH,
    TRAIN_TEST_SPLIT_DATE,
    ENTRY_Z,
    EXIT_Z,
    STOP_Z,
    LABEL_HORIZON_DAYS,
    MODEL_PROB_THRESHOLD,
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    PER_PAIR_CAPITAL_PCT,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_LOSS_PER_TRADE_PCT,
    TRANSACTION_COST_PCT,
    COOLDOWN_DAYS,
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_inputs():
    feats = pd.read_csv(FEATURE_DATASET_PATH, parse_dates=["date"])
    preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

    needed = ["pair_id", "date", "spread", "spread_mean", "spread_std", "hedge_ratio"]
    missing = [c for c in needed if c not in feats.columns]
    if missing:
        raise ValueError(
            f"{FEATURE_DATASET_PATH} is missing columns {missing}. "
            "Regenerate it with the latest features.py, which outputs the spread anchors and the hedge ratio."
        )

    # one date-indexed table per pair, so looking up a day's spread is cheap
    pair_tables = {
        pid: g.set_index("date").sort_index()[["spread", "spread_mean", "spread_std", "hedge_ratio"]]
        for pid, g in feats.groupby("pair_id")
    }
    split = pd.Timestamp(TRAIN_TEST_SPLIT_DATE)
    trading_dates = sorted(d for d in feats["date"].unique() if d >= split)
    return pair_tables, preds, [pd.Timestamp(d) for d in trading_dates]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def run_backtest(pair_tables: dict, preds: pd.DataFrame, trading_dates: list, use_model: bool):
    """Day-by-day simulation. With use_model=False the model probability is
    ignored and every signal is taken (the no-model baseline)."""
    sig = preds.copy()
    if use_model:
        sig = sig[sig["prob_revert"] >= MODEL_PROB_THRESHOLD]
    # group by signal date so "yesterday's signal, enter today" is a lookup
    sig_by_date = {d: g.sort_values("prob_revert", ascending=False) for d, g in sig.groupby("date")}

    realized_equity = float(INITIAL_CAPITAL)   # cash after realised P&L
    positions = {}                              # pair_id -> open position
    cooldown_until = {}                         # pair_id -> day index it can trade again
    trades, curve = [], []

    for i, d in enumerate(trading_dates):
        # ---- 1. check open positions and close ----
        for pid in list(positions.keys()):
            pos = positions[pid]
            tbl = pair_tables[pid]
            if d not in tbl.index:
                continue  # no data for this pair today (e.g. halted), leave it alone
            s_now = tbl.at[d, "spread"]
            fz = pos["side"] * (s_now - pos["mu0"]) / pos["sd0"]
            held = i - pos["entry_idx"]

            reason = None
            if fz <= EXIT_Z:
                reason = "revert"
            elif fz >= STOP_Z:
                reason = "stop"
            elif held >= LABEL_HORIZON_DAYS:
                reason = "timeout"

            if reason is not None:
                gross_pnl = -pos["side"] * pos["notional_a"] * (s_now - pos["entry_spread"])
                cost = pos["gross"] * TRANSACTION_COST_PCT * 2  # entry + exit
                net = gross_pnl - cost
                realized_equity += net
                trades.append({
                    "pair_id": pid, "side": int(pos["side"]),
                    "signal_date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": d,
                    "holding_days": held, "exit_reason": reason,
                    "prob_revert": pos["prob"], "notional_gross": round(pos["gross"], 0),
                    "gross_pnl": round(gross_pnl, 0), "cost": round(cost, 0), "net_pnl": round(net, 0),
                    "return_on_gross": round(net / pos["gross"], 5),
                })
                cooldown_until[pid] = i + COOLDOWN_DAYS
                del positions[pid]

        # ---- 2. entries: yesterday's signals, filled today ----
        if i > 0:
            prev_d = trading_dates[i - 1]
            candidates = sig_by_date.get(prev_d)
            if candidates is not None:
                for _, row in candidates.iterrows():
                    pid = row["pair_id"]
                    if pid in positions or i < cooldown_until.get(pid, -1):
                        continue
                    if len(positions) >= MAX_POSITIONS:
                        break
                    tbl = pair_tables[pid]
                    if prev_d not in tbl.index or d not in tbl.index:
                        continue

                    # frozen anchors come from the signal day's rolling mean and std
                    mu0 = tbl.at[prev_d, "spread_mean"]
                    sd0 = tbl.at[prev_d, "spread_std"]
                    h = abs(tbl.at[prev_d, "hedge_ratio"])
                    if not np.isfinite(sd0) or sd0 <= 0:
                        continue

                    side = np.sign(row["zscore"])
                    s_entry = tbl.at[d, "spread"]
                    z_entry = side * (s_entry - mu0) / sd0  # frozen z at the actual fill
                    if z_entry <= EXIT_Z or z_entry >= STOP_Z:
                        continue  # already reverted overnight, or already near the stop: skip

                    # current equity (realised + open P&L) sets how much I can risk
                    equity_now = realized_equity + _unrealized(positions, pair_tables, d)

                    # Size from the stop: distance from entry to the stop line,
                    # then solve for the leg A notional that loses at most the allowed amount
                    stop_dist = (STOP_Z - z_entry) * sd0
                    notional_a = MAX_LOSS_PER_TRADE_PCT * equity_now / stop_dist
                    gross = notional_a * (1.0 + h)

                    # per-pair and total exposure caps, scale down if over
                    gross_cap = PER_PAIR_CAPITAL_PCT * equity_now
                    open_gross = sum(p["gross"] for p in positions.values())
                    gross_cap = min(gross_cap, MAX_GROSS_EXPOSURE_PCT * equity_now - open_gross)
                    if gross_cap <= 0:
                        continue
                    if gross > gross_cap:
                        scale = gross_cap / gross
                        notional_a *= scale
                        gross = gross_cap
                    if gross < 0.005 * equity_now:
                        continue  # too small to bother with (and fees would eat it)

                    positions[pid] = {
                        "side": side, "entry_idx": i, "entry_date": d, "signal_date": prev_d,
                        "entry_spread": s_entry, "mu0": mu0, "sd0": sd0,
                        "notional_a": notional_a, "gross": gross, "prob": float(row["prob_revert"]),
                    }

        # ---- 3. mark to market ----
        unreal = _unrealized(positions, pair_tables, d)
        # Costs on open positions are only booked at exit. I deduct them up front
        # here so the drawdown isn't understated.
        accrued_cost = sum(p["gross"] * TRANSACTION_COST_PCT * 2 for p in positions.values())
        curve.append({
            "date": d,
            "equity": realized_equity + unreal - accrued_cost,
            "open_positions": len(positions),
            "gross_exposure": sum(p["gross"] for p in positions.values()),
        })

    # anything still open when the test period ends gets closed at the last price
    if positions:
        last_d = trading_dates[-1]
        for pid, pos in list(positions.items()):
            s_now = pair_tables[pid].at[last_d, "spread"] if last_d in pair_tables[pid].index else pos["entry_spread"]
            gross_pnl = -pos["side"] * pos["notional_a"] * (s_now - pos["entry_spread"])
            cost = pos["gross"] * TRANSACTION_COST_PCT * 2
            net = gross_pnl - cost
            realized_equity += net
            trades.append({
                "pair_id": pid, "side": int(pos["side"]),
                "signal_date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": last_d,
                "holding_days": len(trading_dates) - 1 - pos["entry_idx"], "exit_reason": "end_of_test",
                "prob_revert": pos["prob"], "notional_gross": round(pos["gross"], 0),
                "gross_pnl": round(gross_pnl, 0), "cost": round(cost, 0), "net_pnl": round(net, 0),
                "return_on_gross": round(net / pos["gross"], 5),
            })

    return pd.DataFrame(trades), pd.DataFrame(curve)


def _unrealized(positions: dict, pair_tables: dict, d) -> float:
    """Open P&L of all positions on day d, before costs."""
    total = 0.0
    for pid, pos in positions.items():
        tbl = pair_tables[pid]
        if d in tbl.index:
            total += -pos["side"] * pos["notional_a"] * (tbl.at[d, "spread"] - pos["entry_spread"])
    return total


# ---------------------------------------------------------------------------
# performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    m = {"n_trades": len(trades)}
    if curve.empty:
        return m

    eq = curve["equity"].to_numpy()
    rets = np.diff(eq) / eq[:-1]
    n_days = len(curve)
    m["total_return"] = eq[-1] / INITIAL_CAPITAL - 1
    m["annualized_return"] = (eq[-1] / INITIAL_CAPITAL) ** (252 / max(n_days, 1)) - 1
    m["annualized_vol"] = rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 else np.nan
    m["sharpe"] = (rets.mean() / rets.std(ddof=1) * np.sqrt(252)
                   if len(rets) > 1 and rets.std(ddof=1) > 0 else np.nan)
    peak = np.maximum.accumulate(eq)
    m["max_drawdown"] = ((eq - peak) / peak).min()
    m["avg_gross_exposure_pct"] = (curve["gross_exposure"] / curve["equity"]).mean()

    if len(trades) > 0:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        m["win_rate"] = len(wins) / len(trades)
        m["avg_holding_days"] = trades["holding_days"].mean()
        m["avg_net_pnl"] = trades["net_pnl"].mean()
        gp, gl = wins["net_pnl"].sum(), -losses["net_pnl"].sum()
        m["profit_factor"] = gp / gl if gl > 0 else np.inf
        m["total_cost"] = trades["cost"].sum()
        m["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
    return m


def _fmt(v, kind="pct"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.1%}" if kind == "pct" else f"{v:,.2f}"


def build_report(m_model: dict, m_base: dict) -> str:
    rows = [
        ("Trades", "n_trades", "int"),
        ("Total return", "total_return", "pct"),
        ("Annualized return", "annualized_return", "pct"),
        ("Annualized vol", "annualized_vol", "pct"),
        ("Sharpe (rf = 0)", "sharpe", "num"),
        ("Max drawdown", "max_drawdown", "pct"),
        ("Win rate", "win_rate", "pct"),
        ("Profit factor", "profit_factor", "num"),
        ("Avg holding days", "avg_holding_days", "num"),
        ("Avg exposure / equity", "avg_gross_exposure_pct", "pct"),
        ("Total costs (TWD)", "total_cost", "num"),
    ]
    lines = []
    lines.append("=== Backtest report (out-of-sample test period) ===")
    lines.append(f"Initial capital {INITIAL_CAPITAL:,}; entry at |z| >= {ENTRY_Z}, revert at {EXIT_Z}, stop at {STOP_Z}, "
                 f"max holding {LABEL_HORIZON_DAYS} days")
    lines.append(f"Risk: max loss per trade {MAX_LOSS_PER_TRADE_PCT:.0%}, per-pair cap {PER_PAIR_CAPITAL_PCT:.0%}, "
                 f"max {MAX_POSITIONS} positions, total exposure {MAX_GROSS_EXPOSURE_PCT:.0%}, one-way cost {TRANSACTION_COST_PCT:.2%}")
    lines.append(f"Model entry threshold: predicted reversion probability >= {MODEL_PROB_THRESHOLD}")
    lines.append("")
    lines.append(f"{'Metric':<22}{'With model':>14}{'No model':>14}")
    for label, key, kind in rows:
        a, b = m_model.get(key), m_base.get(key)
        if kind == "int":
            lines.append(f"{label:<22}{(a if a is not None else 'n/a'):>14}{(b if b is not None else 'n/a'):>14}")
        else:
            lines.append(f"{label:<22}{_fmt(a, kind):>14}{_fmt(b, kind):>14}")
    lines.append("")
    lines.append(f"Exit reasons (with model): {m_model.get('exit_reasons', {})}")
    lines.append(f"Exit reasons (no model): {m_base.get('exit_reasons', {})}")
    lines.append("")
    lines.append("How to read this:")
    lines.append("- The model only adds value if it is clearly better than the no-model baseline. A small gap, or very few trades, is just noise.")
    lines.append("- The test period is about a year with a limited number of trades, so the confidence intervals on Sharpe and win rate are wide. Don't read them as precise.")
    lines.append("- Not modelled: short-selling availability and fees, gaps through the stop, intraday liquidity impact. Live results will usually be worse than this.")
    return "\n".join(lines)


def plot_equity(curve_model: pd.DataFrame, curve_base: pd.DataFrame, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve_model["date"], curve_model["equity"], label="With model", linewidth=1.8)
    ax.plot(curve_base["date"], curve_base["equity"], label="No-model baseline",
            linewidth=1.4, linestyle="--")
    ax.axhline(INITIAL_CAPITAL, color="gray", linewidth=0.8)
    ax.set_title("Out-of-sample equity curve")
    ax.set_ylabel("Equity (TWD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    pair_tables, preds, trading_dates = load_inputs()
    if not trading_dates:
        raise ValueError("No trading days in the test period. Check TRAIN_TEST_SPLIT_DATE and the data range.")

    trades_m, curve_m = run_backtest(pair_tables, preds, trading_dates, use_model=True)
    trades_b, curve_b = run_backtest(pair_tables, preds, trading_dates, use_model=False)

    report = build_report(compute_metrics(trades_m, curve_m), compute_metrics(trades_b, curve_b))

    os.makedirs(os.path.dirname(TRADES_PATH) or ".", exist_ok=True)
    trades_m.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
    curve_out = curve_m.rename(columns={"equity": "equity_model"}).merge(
        curve_b[["date", "equity"]].rename(columns={"equity": "equity_baseline"}), on="date", how="left"
    )
    curve_out.to_csv(EQUITY_CURVE_PATH, index=False, encoding="utf-8-sig")
    with open(BACKTEST_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    png_path = os.path.join(os.path.dirname(EQUITY_CURVE_PATH) or ".", "equity_curve.png")
    plotted = plot_equity(curve_m, curve_b, png_path)

    print(report)
    print(f"\nTrades: {TRADES_PATH}; equity curve: {EQUITY_CURVE_PATH}; report: {BACKTEST_REPORT_PATH}"
          + (f"; plot: {png_path}" if plotted else " (matplotlib not installed, plot skipped)"))


if __name__ == "__main__":
    main()
