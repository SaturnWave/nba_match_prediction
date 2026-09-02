"""
Does averaging several seeds make the model more consistent, and by how much?

WHY THIS QUESTION
    Across 120 walk-forward cells the model's accuracy has a standard deviation
    of 0.0557. Decomposing it: 0.0221 comes from the random seed alone (the same
    month, the same data, a different subsample draw) and 0.0544 from genuine
    month-to-month differences - so the seed accounts for 14% of the variance.

    Averaging N models should cut the seed part by sqrt(N) and leave the month
    part untouched, which predicts only a 3-4% reduction in total spread. That
    is a prediction worth checking rather than asserting, because bagging also
    tends to buy a little accuracy, and the prediction says nothing about that.

WHAT IS MEASURED
    Per monthly fold, an ensemble of N models fitted on identical data with
    different seeds, their probabilities averaged. N=1 is the current setup, so
    the comparison is paired within the fold and the only thing that changes is
    how many models vote.

    Reported per N: mean accuracy, its spread across folds, AUC, Brier, and the
    paired gain over N=1.

Run:  py prediction_engines/seed_ensemble.py [--months 24] [--sizes 1 3 5 10]
Output: output/seed_ensemble.json
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
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "seed_ensemble.json")

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
MIN_TEST_GAMES = 40
MIN_TRAIN_GAMES = 2000
BURN_IN = 10


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_sibling("calibration")
wf = _load_sibling("walk_forward")


def fit_one(train_df, features, seed):
    fit_df, stop_df, _ = cal.three_way(train_df.sort_values("game_date"))
    X_fit, y_fit = cal.Xy(fit_df, features)
    X_stop, y_stop = cal.Xy(stop_df, features)
    model = lgb.LGBMClassifier(random_state=seed, **PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model


def scores(y, p):
    return {"accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0.0, 1.0]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--sizes", type=int, nargs="*", default=[1, 3, 5, 10])
    args = parser.parse_args()

    dataset, features, _ = wf.load_any_dataset(args.dataset)
    dataset = wf.add_team_game_index(dataset)
    folds = wf.month_folds(dataset, args.months)
    max_n = max(args.sizes)
    seeds = [42 + 7 * i for i in range(max_n)]
    print(f"dataset {len(dataset):,} mac | {len(folds)} ay | "
          f"ensemble boyutlari {args.sizes} | en fazla {max_n} model/ay\n")

    results = []
    for i, (month, train, test) in enumerate(folds, 1):
        t0 = time.time()
        X_test, y_test = cal.Xy(test, features)
        seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
        keep = seasoned >= BURN_IN
        if keep.sum() < MIN_TEST_GAMES or len(np.unique(y_test.values[keep])) < 2:
            continue

        # Fit every seed once, then read each ensemble size off the same models.
        probs = [fit_one(train, features, s).predict_proba(X_test)[:, 1] for s in seeds]
        entry = {"month": month, "n_test": int(keep.sum())}
        for n in args.sizes:
            mean_p = np.mean(probs[:n], axis=0)
            entry[f"n{n}"] = scores(y_test.values[keep], mean_p[keep])
        results.append(entry)
        print(f"[{i}/{len(folds)}] {month} n={entry['n_test']:3d}  "
              + "  ".join(f"N{n}={entry[f'n{n}']['accuracy']:.3f}" for n in args.sizes)
              + f"  ({time.time()-t0:.0f} sn)", flush=True)

    print(f"\n=== {len(results)} ay katmani, burn-in >= {BURN_IN} ===")
    print(f"  {'N':>3} {'accuracy':>17} {'auc':>9} {'brier':>9} "
          f"{'N=1 farki':>11} {'se':>8}")
    summary = {}
    base = np.array([r["n1"]["accuracy"] for r in results])
    for n in args.sizes:
        acc = np.array([r[f"n{n}"]["accuracy"] for r in results])
        auc = np.mean([r[f"n{n}"]["auc"] for r in results])
        bri = np.mean([r[f"n{n}"]["brier"] for r in results])
        delta = acc - base
        se = delta.std(ddof=1) / np.sqrt(len(delta)) if n != 1 else 0.0
        summary[f"n{n}"] = {"accuracy_mean": float(acc.mean()),
                            "accuracy_std": float(acc.std(ddof=1)),
                            "auc": float(auc), "brier": float(bri),
                            "delta_vs_n1": float(delta.mean()),
                            "delta_se": float(se)}
        print(f"  {n:>3} {acc.mean():.4f}+/-{acc.std(ddof=1):.4f} {auc:9.4f} "
              f"{bri:9.4f} {delta.mean():+11.4f} {se:8.4f}")

    std1 = summary["n1"]["accuracy_std"]
    print(f"\n  varyans: N=1 std {std1:.4f}")
    for n in args.sizes[1:]:
        s = summary[f"n{n}"]["accuracy_std"]
        print(f"           N={n:<2} std {s:.4f}  ({(s - std1) / std1:+.1%})")
    print("\n  tahmin edilen: seed payi %14 oldugu icin ~%3-4 azalma bekleniyordu")

    with open(REPORT_PATH, "w") as f:
        json.dump({"burn_in": BURN_IN, "sizes": args.sizes, "seeds": seeds,
                   "n_folds": len(results), "folds": results,
                   "summary": summary}, f, indent=2)
    print(f"\nYazildi: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
