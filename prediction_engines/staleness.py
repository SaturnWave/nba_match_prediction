"""
How fast do the model's weights go stale?

THE QUESTION
    Features are always as-of the game: predicting a team's 75th game uses
    rolling windows built from its first 74. The model WEIGHTS are a different
    matter. Production trains once on the first 75% of the target season and
    then predicts the remaining 25%, so an April game is scored by weights
    frozen in early March. Whether that costs anything decides how often a live
    deployment has to retrain - and since training takes seconds, the answer
    is worth having rather than guessing.

DESIGN
    Hold the test games FIXED and vary the training cutoff. For each test month,
    train one model per lag - cutoff at the month start, and 30, 60, 90 and 150
    days before it - then score all of them on the same games.

    The first version of this experiment did the opposite: it fixed the cutoff
    and bucketed the following games by how far ahead they were. That confounds
    weight age with the calendar. Games 61-120 days past an autumn cutoff are
    spring games, and spring is genuinely more predictable (April lifts 0.698
    against January's 0.621), so the stalest bucket looked the BEST. Controlling
    for each bucket's naive baseline did not rescue it either, because month
    difficulty is almost unrelated to the base rate - regressing one on the
    other gives an R^2 of 0.029.

    With the test games fixed, every lag faces identical opposition, identical
    schedule and identical difficulty, so any difference is the weights.

Run:  py prediction_engines/staleness.py [--months 10] [--seeds 3]
Output: output/staleness.json
"""
import os
import sys
import json
import time
import argparse
import importlib.util

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "staleness.json")

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
LAGS_DAYS = [0, 30, 60, 90, 150]
MIN_TRAIN = 3000
MIN_TEST = 40
BURN_IN = 10


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_sibling("calibration")
wf = _load_sibling("walk_forward")


def fit_ensemble(train_df, features, seeds):
    fit_df, stop_df, _ = cal.three_way(train_df.sort_values("game_date"))
    X_fit, y_fit = cal.Xy(fit_df, features)
    X_stop, y_stop = cal.Xy(stop_df, features)
    members = []
    for seed in seeds:
        m = lgb.LGBMClassifier(random_state=seed, **PARAMS)
        m.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        members.append(m)
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--months", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    dataset, features, _ = wf.load_any_dataset(args.dataset)
    dataset = wf.add_team_game_index(dataset)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    dataset = dataset.sort_values("game_date").reset_index(drop=True)
    seeds = [42 + 7 * i for i in range(args.seeds)]

    months = sorted(dataset["game_date"].dt.to_period("M").unique())
    usable = [m for m in months
              if (dataset["game_date"] < m.start_time - pd.Timedelta(days=max(LAGS_DAYS))
                  ).sum() >= MIN_TRAIN]
    test_months = usable[-args.months:]
    print(f"dataset {len(dataset):,} mac | {len(test_months)} test ayi | "
          f"gecikmeler {LAGS_DAYS} gun | {len(seeds)} seed\n")

    rows = []
    for i, month in enumerate(test_months, 1):
        start = month.start_time
        test = dataset[dataset["game_date"].dt.to_period("M") == month]
        seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
        keep = seasoned >= BURN_IN
        if keep.sum() < MIN_TEST:
            continue
        X_test, y_test = cal.Xy(test, features)
        y = y_test.values[keep]
        if len(np.unique(y)) < 2:
            continue

        t0 = time.time()
        line = []
        for lag in LAGS_DAYS:
            cutoff = start - pd.Timedelta(days=lag)
            train = dataset[dataset["game_date"] < cutoff]
            if len(train) < MIN_TRAIN:
                continue
            members = fit_ensemble(train, features, seeds)
            p = np.mean([m.predict_proba(X_test)[:, 1] for m in members], axis=0)[keep]
            rows.append({
                "month": str(month), "lag_days": lag, "n_train": int(len(train)),
                "n_test": int(keep.sum()),
                "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
                "auc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p))})
            line.append(f"{lag}g:{rows[-1]['accuracy']:.3f}")
        print(f"[{i}/{len(test_months)}] {month} n={int(keep.sum()):3d}  "
              + "  ".join(line) + f"  ({time.time()-t0:.0f} sn)", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("olcum yapilamadi")
        return 1

    print(f"\n=== AYNI MACLAR, FARKLI EGITIM YASI ({df.month.nunique()} ay) ===")
    print(f"  {'gecikme':>8} {'ay':>4} {'accuracy':>17} {'auc':>8} {'brier':>8} "
          f"{'0g farki':>10} {'se':>8} {'sonuc':>18}")
    pivot = df.pivot_table(index="month", columns="lag_days", values="accuracy")
    summary = {}
    for lag in LAGS_DAYS:
        g = df[df.lag_days == lag]
        if g.empty:
            continue
        entry = {"n_months": int(len(g)),
                 "accuracy_mean": float(g.accuracy.mean()),
                 "accuracy_std": float(g.accuracy.std(ddof=1)) if len(g) > 1 else None,
                 "auc": float(g.auc.mean()), "brier": float(g.brier.mean())}
        if lag != 0 and 0 in pivot.columns and lag in pivot.columns:
            d = (pivot[lag] - pivot[0]).dropna()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            entry["delta_vs_fresh"] = float(d.mean())
            entry["delta_se"] = float(se)
            verdict = "ANLAMLI" if abs(d.mean()) > 2 * se else "ayirt edilemiyor"
            tail = f"{d.mean():+10.4f} {se:8.4f} {verdict:>18}"
        else:
            tail = f"{'-':>10} {'-':>8} {'-':>18}"
        summary[f"lag_{lag}"] = entry
        print(f"  {lag:>6}g {len(g):4} {g.accuracy.mean():.4f}"
              f"+/-{(g.accuracy.std(ddof=1) if len(g) > 1 else 0):.4f} "
              f"{g.auc.mean():8.4f} {g.brier.mean():8.4f} {tail}")

    with open(REPORT_PATH, "w") as f:
        json.dump({"design": "test games fixed, training cutoff varied",
                   "lags_days": LAGS_DAYS, "burn_in": BURN_IN, "seeds": seeds,
                   "rows": rows, "summary": summary}, f, indent=2)
    print(f"\nYazildi: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
