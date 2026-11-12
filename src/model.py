# -*- coding: utf-8 -*-
"""
Gradient boosted trees that predict whether a spread will revert after it
gets stretched.

What happens here:
1. Load the daily features from features.py.
2. Pick out the "events": days where ENTRY_Z <= |z| < STOP_Z. Days already
   past the stop line aren't tradable, so they're left out of training.
3. Label each event. 1 if, within LABEL_HORIZON_DAYS, the spread gets back
   inside EXIT_Z before reaching STOP_Z. 0 if it hits the stop first or
   just doesn't revert in time. The z used here is frozen at the entry day's
   mean and std (see the note in features.py).
4. Split by time, train on the early part, and write out predicted reversion
   probabilities for the test period.

Leakage is the main way this kind of model goes wrong, so three precautions:
- A label looks N days ahead. A training event whose label window crosses the
  split date has seen the test period, so I drop those (purging).
- Events in the last N days of the data have no label yet. They aren't scored,
  but they still get a prediction because the backtest needs it, which is how
  it would work live.
- Hyperparameters are fixed and conservative. I never tune them on the test set.

One caveat: when a pair stays above the threshold for several days in a row,
neighbouring events share almost the same label, so the real sample size is much
smaller than the row count. I read the metrics for the rough picture only.

Usage:
    python src/model.py

Input:  output/features_dataset.csv
Output: output/predictions.csv, output/model_report.txt, output/model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

from config import (
    FEATURE_DATASET_PATH,
    PREDICTIONS_PATH,
    MODEL_REPORT_PATH,
    MODEL_PATH,
    TRAIN_TEST_SPLIT_DATE,
    ENTRY_Z,
    EXIT_Z,
    STOP_Z,
    LABEL_HORIZON_DAYS,
    MODEL_PROB_THRESHOLD,
    MODEL_RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# loading and extra features
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_DATASET_PATH)
    if "date" not in df.columns:
        raise ValueError(
            f"{FEATURE_DATASET_PATH} has no date column. Check the output format of features.py"
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["pair_id", "date"]).reset_index(drop=True)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few direction-aligned features on top of features.py.

    Long-spread (z < 0) and short-spread (z > 0) trades mean opposite things.
    "A is strong vs the sector" goes with a short spread but against a long one.
    A tree can learn that flip on its own, but not with this little data, so I
    multiply by side (sign of z) up front.
    """
    df = df.copy()
    g = df.groupby("pair_id", sort=False)

    df["side"] = np.sign(df["zscore"])
    df["abs_zscore"] = df["zscore"].abs()
    # is z moving toward the middle or further out over the last 5 days (negative = converging after aligning by side)
    df["zscore_chg_5"] = df["side"] * (df["zscore"] - g["zscore"].shift(5))

    # relative strength gap between the two legs, aligned by side
    df["rs_diff_market"] = df["side"] * (df["rel_strength_market_a"] - df["rel_strength_market_b"])
    df["rs_diff_sector"] = df["side"] * (df["rel_strength_sector_a"] - df["rel_strength_sector_b"])
    # volume and volatility gap between the two legs
    df["volume_ratio_diff"] = df["volume_ratio_a"] - df["volume_ratio_b"]
    df["atr_pct_diff"] = df["atr_pct_a"] - df["atr_pct_b"]
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Columns that go into the model: pair-level, both legs, and the aligned extras."""
    pair_level = ["abs_zscore", "side", "zscore_chg_5", "spread_vol"]
    # how strongly the pair passed screening when it was selected (walkforward.py only)
    pair_level += [c for c in ("form_p_value", "form_half_life") if c in df.columns]
    leg_level = [c for c in df.columns if c.endswith("_a") or c.endswith("_b")]
    # stock_a / stock_b are ticker strings, not features
    leg_level = [c for c in leg_level if c not in ("stock_a", "stock_b")]
    aligned = ["rs_diff_market", "rs_diff_sector", "volume_ratio_diff", "atr_pct_diff"]
    return pair_level + leg_level + aligned


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def label_events(df: pd.DataFrame) -> pd.DataFrame:
    """Find the trading events and label them.

    Each returned row is one event (a pair on a day with |z| in [ENTRY_Z, STOP_Z)).
    Added columns:
    - label: 1 = reverted inside EXIT_Z first, 0 = hit the stop first or timed out,
             NaN = label window not finished yet (events near the end of the data)
    - days_to_outcome: trading days until the outcome
    - label_end_date: last day of the label window, used to purge train rows
    """
    H = LABEL_HORIZON_DAYS
    frames = []
    for _, g in df.groupby("pair_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        z = g["zscore"].to_numpy()
        spread = g["spread"].to_numpy()
        mu = g["spread_mean"].to_numpy()
        sd = g["spread_std"].to_numpy()
        dates = g["date"].to_numpy()
        n = len(g)
        # Rows past the end of a pair's selection period exist only so labels can
        # see the outcome. I can't enter on them, so they're never events.
        tradable = g["tradable"].to_numpy() if "tradable" in g.columns else np.ones(n, dtype=bool)

        label = np.full(n, np.nan)
        days = np.full(n, np.nan)
        end_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

        for i in range(n):
            if not tradable[i] or not (ENTRY_Z <= abs(z[i]) < STOP_Z):
                continue
            if i + H > n - 1:
                continue  # label window not finished, leave NaN
            if not np.isfinite(sd[i]) or sd[i] <= 0:
                continue

            side = np.sign(z[i])
            outcome, d = 0, H
            for j in range(1, H + 1):
                # Frozen z: measure every later day against the entry day's mean
                # and std, not that day's rolling z-score, whose mean chases the
                # spread and pulls z back to zero by itself. A lower s means the
                # trade is going my way. If the spread jumps and stays there, s
                # stays high and the event is correctly labelled a failure.
                s = side * (spread[i + j] - mu[i]) / sd[i]
                if s <= EXIT_Z:
                    outcome, d = 1, j
                    break
                if s >= STOP_Z:
                    outcome, d = 0, j
                    break
            label[i] = outcome
            days[i] = d
            end_date[i] = dates[i + H]

        g["label"] = label
        g["days_to_outcome"] = days
        g["label_end_date"] = end_date
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    is_event = (out["zscore"].abs() >= ENTRY_Z) & (out["zscore"].abs() < STOP_Z)
    if "tradable" in out.columns:
        is_event &= out["tradable"].astype(bool)
    return out[is_event].reset_index(drop=True)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model():
    """LightGBM if it's installed, otherwise sklearn's gradient boosting, which
    behaves about the same. Settings are deliberately conservative (shallow trees,
    heavy regularisation, big leaves) because there isn't much data."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=8,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=MODEL_RANDOM_SEED,
            verbose=-1,
        )
        return model, "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=8,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=MODEL_RANDOM_SEED,
        )
        return model, "sklearn HistGradientBoosting (LightGBM not installed, using the fallback)"


def split_train_test(events: pd.DataFrame):
    """Split by time and purge train rows whose label window crosses the split date.

    - train: labelled, and the label window ends before the split date
    - test_eval: labelled events on or after the split date (out of sample)
    - test_all: every event on or after the split date, including unlabelled
      ones, because the backtest needs those
    """
    split = pd.Timestamp(TRAIN_TEST_SPLIT_DATE)
    labeled = events["label"].notna()
    train = events[labeled & (events["label_end_date"] < split)]
    test_eval = events[labeled & (events["date"] >= split)]
    test_all = events[events["date"] >= split]
    return train, test_eval, test_all


def walk_forward_cv(train: pd.DataFrame, feature_cols: list, n_folds: int = 3) -> list:
    """Expanding-window CV inside the train set: fit on earlier data, score the
    next chunk. Checks that the model can rank events it hasn't seen from the
    future. Each fold purges train rows whose label window runs into the
    validation chunk."""
    dates = np.sort(train["date"].unique())
    chunks = np.array_split(dates, n_folds + 1)
    aucs = []
    for k in range(1, n_folds + 1):
        val_start = chunks[k][0]
        val_end = chunks[k][-1]
        tr = train[train["label_end_date"] < val_start]
        va = train[(train["date"] >= val_start) & (train["date"] <= val_end)]
        if len(tr) < 100 or va["label"].nunique() < 2 or tr["label"].nunique() < 2:
            continue
        model, _ = build_model()
        model.fit(tr[feature_cols], tr["label"].astype(int))
        prob = model.predict_proba(va[feature_cols])[:, 1]
        aucs.append(roc_auc_score(va["label"].astype(int), prob))
    return aucs


def calibration_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Bucket predictions and compare the average predicted probability with the
    actual reversion rate in each bucket."""
    df = pd.DataFrame({"y": y_true, "p": prob})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    tbl = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean")
    )
    return tbl.round(3)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run() -> dict:
    df = add_model_features(load_features())
    feature_cols = get_feature_columns(df)

    events = label_events(df).dropna(subset=feature_cols).reset_index(drop=True)
    train, test_eval, test_all = split_train_test(events)

    if len(train) < 100:
        raise ValueError(f"Only {len(train)} training events, too few to train a model")
    if train["label"].nunique() < 2:
        raise ValueError("Training labels contain a single class. Check ENTRY_Z / EXIT_Z / STOP_Z.")

    cv_aucs = walk_forward_cv(train, feature_cols)

    model, model_name = build_model()
    model.fit(train[feature_cols], train["label"].astype(int))

    lines = []
    lines.append(f"Model: {model_name}")
    lines.append(f"Number of features: {len(feature_cols)}")
    lines.append(
        f"Rule: enter at |z| >= {ENTRY_Z}; within {LABEL_HORIZON_DAYS} trading days the spread must get back inside {EXIT_Z} "
        f"(success) before it goes past {STOP_Z} (failure)"
    )
    lines.append("")
    lines.append(f"Train events: {len(train)} (reversion rate {train['label'].mean():.1%}), "
                 f"across {train['pair_id'].nunique()} pairs")
    lines.append(f"Out-of-sample labelled events: {len(test_eval)}")
    if cv_aucs:
        lines.append(f"Walk-forward CV AUC inside the train set: {[round(a, 3) for a in cv_aucs]}, "
                     f"mean {np.mean(cv_aucs):.3f}")
    else:
        lines.append("Walk-forward CV: not enough samples, skipped")

    metrics = {}
    if len(test_eval) > 0:
        prob_eval = model.predict_proba(test_eval[feature_cols])[:, 1]
        y_eval = test_eval["label"].astype(int).to_numpy()
        base_rate = y_eval.mean()
        metrics["base_rate"] = base_rate
        metrics["brier"] = brier_score_loss(y_eval, prob_eval)
        # Baseline Brier: the score from always predicting the test-period base
        # rate. The model only adds something if it beats this.
        metrics["brier_baseline"] = brier_score_loss(y_eval, np.full_like(prob_eval, base_rate, dtype=float))

        lines.append("")
        lines.append("=== Out-of-sample performance ===")
        lines.append(f"Actual reversion rate (base rate): {base_rate:.1%}")
        if len(np.unique(y_eval)) == 2:
            metrics["auc"] = roc_auc_score(y_eval, prob_eval)
            lines.append(f"AUC: {metrics['auc']:.3f} (0.5 = coin flip)")
        else:
            lines.append("AUC: test labels contain a single class, can't compute")
        lines.append(f"Brier score: {metrics['brier']:.4f} (baseline from always predicting the mean: {metrics['brier_baseline']:.4f}, "
                     f"lower is better, and the model only adds value if it beats the baseline)")

        sel = prob_eval >= MODEL_PROB_THRESHOLD
        lines.append(f"Events with predicted probability >= {MODEL_PROB_THRESHOLD}: {int(sel.sum())}"
                     + (f", actual reversion rate {y_eval[sel].mean():.1%}" if sel.sum() > 0 else ""))

        cal = calibration_table(y_eval, prob_eval)
        if not cal.empty:
            lines.append("")
            lines.append("Calibration (predicted probability bucket vs actual reversion rate, the closer the better):")
            lines.append(cal.to_string())

        # Permutation importance (AUC drop when a feature is shuffled). I trust it
        # more than the tree's own split counts.
        if len(np.unique(y_eval)) == 2:
            pi = permutation_importance(
                model, test_eval[feature_cols], y_eval,
                scoring="roc_auc", n_repeats=5, random_state=MODEL_RANDOM_SEED,
            )
            imp = pd.Series(pi.importances_mean, index=feature_cols).sort_values(ascending=False)
            lines.append("")
            lines.append("Permutation importance (AUC drop after shuffling, top 8; near zero or negative = no contribution):")
            lines.append(imp.head(8).round(4).to_string())
    else:
        lines.append("")
        lines.append("No scorable out-of-sample events (test period too short, or no pair produced a signal)")

    # predictions for every test-period event, including ones without a label yet (the backtest uses them)
    pred = test_all[["pair_id", "stock_a", "stock_b", "date", "zscore", "side", "label"]].copy()
    pred["prob_revert"] = model.predict_proba(test_all[feature_cols])[:, 1]

    os.makedirs(os.path.dirname(PREDICTIONS_PATH) or ".", exist_ok=True)
    pred.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    report = "\n".join(lines)
    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    return {"report": report, "metrics": metrics, "predictions": pred, "cv_aucs": cv_aucs}


def main():
    result = run()
    print(result["report"])
    print(f"\nPredictions saved to {PREDICTIONS_PATH}, report saved to {MODEL_REPORT_PATH}")


if __name__ == "__main__":
    main()
