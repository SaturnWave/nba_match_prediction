"""
Walk-forward (expanding-origin) cross-validation over monthly folds and seeds.

WHY BOTH AXES
    A single chronological split told us home_win accuracy was 0.786. Monthly
    folds put it at 0.65 - the split's test window (4 Mar - 12 Apr 2026) was the
    season's most predictable stretch, with a 61.0% home-win base rate against
    55.5% for the season.

    Folds alone were still not enough. Re-running the identical configuration
    after only changing the row sort order moved results by 1-1.5 points,
    because LightGBM's subsample=0.8 is order-sensitive. That is LARGER than the
    effect sizes we were trying to judge, so every arm is now fitted under
    several seeds and averaged over (fold x seed). A difference smaller than the
    seed spread is not a difference.

ARMS
    A             the current 190 features
    A+rating      plus 18 opponent-adjusted ratings (Elo, Massey, SOS,
                  impact-vs-expected) - the information the other 184 lack
    B             plus 24 trailing defensive-box + player-tracking features
    A+rating+B    all of it
    Each arm is also scored with its probabilities passed through that fold's
    calibrator, which is fitted on the fold's own calibration slice.

PROTOCOL PER (FOLD, SEED)
    train = every game before the first day of month M, split chronologically
            into fit 80% / early-stop 10% / calibrate 10%
    test  = the games played in month M
    Nothing from month M or later reaches the fit, the stopping, or the
    calibrator.

BURN-IN
    A team's opening games are worth training on but not worth being judged on:
    the rolling features are still cold. Each burn-in keeps a test game only
    when BOTH teams already have that many games that season. Training is never
    filtered. All thresholds are scored from the same fitted models.

Run:  py prediction_engines/walk_forward.py [--months N] [--seeds N]
Output: output/walk_forward_2025_26.json
"""
import os
import json
import argparse
import importlib.util
import itertools

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss, mean_absolute_error)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "walk_forward_2025_26.json")

BASE_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
MIN_TEST_GAMES = 40
MIN_TRAIN_GAMES = 2000
BURN_INS = (0, 10, 15)
PRIMARY_BURN_IN = "10"


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_sibling("calibration")
dtf = _load_sibling("defensive_tracking_features")
ratings = _load_sibling("team_ratings")


def add_team_game_index(dataset):
    """Add home_gp / away_gp: games that team had already played THAT season.

    A pre-game quantity, so filtering on it says nothing about the game itself.
    """
    ds = dataset.sort_values(["game_date", "game_id"]).copy()
    played = {}
    home_gp, away_gp = [], []
    for season, home, away in zip(ds["season"], ds["home_team"], ds["away_team"]):
        k_home, k_away = (season, home), (season, away)
        home_gp.append(played.get(k_home, 0))
        away_gp.append(played.get(k_away, 0))
        played[k_home] = played.get(k_home, 0) + 1
        played[k_away] = played.get(k_away, 0) + 1
    ds["home_gp"] = home_gp
    ds["away_gp"] = away_gp
    return ds


def month_folds(dataset, max_months=None):
    ds = dataset.sort_values("game_date")
    folds = []
    for month in sorted(ds["game_date"].dt.to_period("M").unique()):
        test = ds[ds["game_date"].dt.to_period("M") == month]
        train = ds[ds["game_date"] < month.start_time]
        if len(test) < MIN_TEST_GAMES or len(train) < MIN_TRAIN_GAMES:
            continue
        folds.append((str(month), train, test))
    return folds[-max_months:] if max_months else folds


def fit_classifier(train_df, features, seed):
    fit_df, stop_df, cal_df = cal.three_way(train_df.sort_values("game_date"))
    X_fit, y_fit = cal.Xy(fit_df, features)
    X_stop, y_stop = cal.Xy(stop_df, features)
    model = lgb.LGBMClassifier(random_state=seed, **BASE_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model, cal_df


def fit_regressor(train_df, features, target, seed):
    tr = train_df.sort_values("game_date")
    cut = int(len(tr) * 0.9)
    fit_df, stop_df = tr.iloc[:cut], tr.iloc[cut:]
    X_fit = fit_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_stop = stop_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    model = lgb.LGBMRegressor(random_state=seed, **BASE_PARAMS)
    model.fit(X_fit, fit_df[target].astype(float),
              eval_set=[(X_stop, stop_df[target].astype(float))], eval_metric="mae",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model


def clf_scores(y, p):
    return {"accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0.0, 1.0]))}


def run_cell(month, train, test, arms, seed):
    """One (fold, seed): fit every arm, score at every burn-in."""
    probs, margins = {}, {}
    for name, features in arms.items():
        model, cal_df = fit_classifier(train, features, seed)
        X_test, y_test = cal.Xy(test, features)
        p = model.predict_proba(X_test)[:, 1]
        probs[name] = p

        X_cal, y_cal = cal.Xy(cal_df, features)
        p_cal = model.predict_proba(X_cal)[:, 1]
        calibrators = cal.fit_calibrators(p_cal, y_cal)
        # Chosen on the calibration slice, never on the test month - picking by
        # test Brier would be selecting on the answer.
        pick = min(calibrators, key=lambda k: brier_score_loss(
            y_cal, cal.apply_calibrator(k, calibrators[k], p_cal)))
        probs[f"{name}+cal"] = cal.apply_calibrator(pick, calibrators[pick], p)

        reg = fit_regressor(train, features, "point_diff", seed)
        X = test[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        margins[name] = reg.predict(X)

    y = test["home_win"].astype(float).values
    seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
    actual_margin = test["point_diff"].astype(float).values

    cell = {"month": month, "seed": seed, "n_train": int(len(train)), "by_burn_in": {}}
    for burn in BURN_INS:
        keep = seasoned >= burn
        if keep.sum() < MIN_TEST_GAMES or len(np.unique(y[keep])) < 2:
            continue
        entry = {"n_test": int(keep.sum()),
                 "home_win_base_rate": float(y[keep].mean())}
        for name, p in probs.items():
            entry[name] = clf_scores(y[keep], p[keep])
        for name, m in margins.items():
            entry[name]["point_diff_mae"] = float(
                mean_absolute_error(actual_margin[keep], m[keep]))
        cell["by_burn_in"][str(burn)] = entry
    return cell


def collect(cells, burn, arm, metric):
    vals = [c["by_burn_in"][burn][arm][metric] for c in cells
            if burn in c["by_burn_in"] and metric in c["by_burn_in"][burn].get(arm, {})]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}


def seed_spread(cells, burn, arm, metric):
    """How much this arm moves on seed alone, holding the month fixed.

    This is the noise floor: any arm-to-arm gap smaller than it is unreadable.
    """
    by_month = {}
    for c in cells:
        if burn not in c["by_burn_in"]:
            continue
        by_month.setdefault(c["month"], []).append(c["by_burn_in"][burn][arm][metric])
    spreads = [max(v) - min(v) for v in by_month.values() if len(v) > 1]
    return float(np.mean(spreads)) if spreads else float("nan")


def paired_delta(cells, burn, arm, base, metric):
    """Arm minus base within the same (month, seed) cell."""
    deltas = [c["by_burn_in"][burn][arm][metric] - c["by_burn_in"][burn][base][metric]
              for c in cells if burn in c["by_burn_in"]]
    if not deltas:
        return None
    mean, std = float(np.mean(deltas)), float(np.std(deltas, ddof=1))
    return {"mean": mean, "std": std, "n": len(deltas),
            "standard_error": std / np.sqrt(len(deltas)),
            "cells_ahead": int(sum(1 for d in deltas if d > 0))}


def naive_baseline(cells, burn):
    rates = {c["month"]: c["by_burn_in"][burn]["home_win_base_rate"]
             for c in cells if burn in c["by_burn_in"]}
    return float(np.mean([max(r, 1 - r) for r in rates.values()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    dataset, features_a = cal.load_dataset()
    print(f"dataset: {len(dataset)} mac, ARM A {len(features_a)} feature")

    dataset = dtf.add_defensive_tracking_features(
        dataset, data_dir=os.path.join(PROJECT_ROOT, "nba_data"))
    dt_feats = [c for c in dtf.DT_FEATURE_COLS if c in dataset.columns]

    dataset = ratings.add_rating_features(dataset)
    rating_feats = [c for c in ratings.RATING_FEATURE_COLS if c in dataset.columns]

    arms = {
        "A": features_a,
        "A+rating": features_a + rating_feats,
        "B": features_a + dt_feats,
        "A+rating+B": features_a + rating_feats + dt_feats,
    }
    for name, feats in arms.items():
        print(f"  {name:12} {len(feats)} feature")

    dataset = add_team_game_index(dataset)
    folds = month_folds(dataset, args.months)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"\n{len(folds)} ay x {len(seeds)} seed = {len(folds) * len(seeds)} hucre, "
          f"hucre basina {len(arms)} kol\n")

    cells = []
    for i, ((month, train, test), seed) in enumerate(itertools.product(folds, seeds), 1):
        cell = run_cell(month, train, test, arms, seed)
        cells.append(cell)
        e = cell["by_burn_in"].get(PRIMARY_BURN_IN) or next(iter(cell["by_burn_in"].values()), None)
        if e:
            print(f"[{i}/{len(folds)*len(seeds)}] {month} seed={seed} n={e['n_test']:3d}  "
                  + "  ".join(f"{k}={e[k]['accuracy']:.3f}" for k in arms), flush=True)

    summary = {}
    for burn in (str(b) for b in BURN_INS):
        if not any(burn in c["by_burn_in"] for c in cells):
            continue
        base = naive_baseline(cells, burn)
        block = {"naive_baseline": base,
                 "n_cells": sum(1 for c in cells if burn in c["by_burn_in"])}
        for arm in list(arms) + [f"{a}+cal" for a in arms]:
            block[arm] = {m: collect(cells, burn, arm, m)
                          for m in ("accuracy", "auc", "brier", "log_loss")}
            if arm in arms:
                block[arm]["point_diff_mae"] = collect(cells, burn, arm, "point_diff_mae")
        block["seed_noise_floor_accuracy"] = {
            arm: seed_spread(cells, burn, arm, "accuracy") for arm in arms}
        block["paired_vs_A"] = {
            arm: paired_delta(cells, burn, arm, "A", "accuracy")
            for arm in list(arms)[1:] + [f"{a}+cal" for a in arms]}
        summary[burn] = block

    for burn, block in summary.items():
        base = block["naive_baseline"]
        print(f"\n=== BURN-IN >= {burn}  ({block['n_cells']} hucre, "
              f"naif taban {base:.4f}) ===")
        print(f"  {'kol':16} {'accuracy':>17} {'kazanc':>8} {'auc':>17} {'brier':>9}")
        for arm in list(arms) + [f"{a}+cal" for a in arms]:
            s = block[arm]
            print(f"  {arm:16} {s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}  "
                  f"{s['accuracy']['mean'] - base:+8.4f}  "
                  f"{s['auc']['mean']:.4f}+/-{s['auc']['std']:.4f}  "
                  f"{s['brier']['mean']:9.4f}")
        print(f"\n  seed gurultu tabani (ayni ayda seed'den gelen yayilim):")
        for arm, v in block["seed_noise_floor_accuracy"].items():
            print(f"    {arm:16} {v:.4f}")
        print(f"\n  A'ya gore eslesmis fark (ayni ay + ayni seed):")
        for arm, d in block["paired_vs_A"].items():
            if d is None:
                continue
            sig = "ANLAMLI" if abs(d["mean"]) > 2 * d["standard_error"] else "ayirt edilemiyor"
            print(f"    {arm:16} {d['mean']:+.4f} +/- {d['std']:.4f}  "
                  f"(se {d['standard_error']:.4f}, {d['cells_ahead']}/{d['n']})  {sig}")

    with open(REPORT_PATH, "w") as f:
        json.dump({"protocol": "expanding-origin monthly folds x seeds; per cell "
                               "fit 80% / early-stop 10% / calibrate 10% of prior games; "
                               "burn-in filters the TEST set only",
                   "burn_ins": list(BURN_INS), "seeds": seeds,
                   "arm_sizes": {k: len(v) for k, v in arms.items()},
                   "n_cells": len(cells), "cells": cells,
                   "summary_by_burn_in": summary}, f, indent=2)
    print(f"\nYazildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
