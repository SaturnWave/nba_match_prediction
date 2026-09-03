"""Does changing HOW the model is trained buy what more features could not?

WHY THIS EXPERIMENT
    Six feature families have now been screened at high power - ratings,
    calibration, defensive/tracking, matchups, clutch, availability - and every
    one landed inside its own error bars. Walk-forward accuracy has sat at
    0.663 (single model) to 0.679 (ten-seed ensemble) against a 0.555 naive
    baseline throughout. If the information is not in a new column, the
    remaining place it can be is in the training protocol.

    That is also the question the project owner raised directly: training on a
    team's first 40 games and then predicting its 75th looks like a long shot,
    and a model fitted mostly on early-season rows may simply be answering a
    different question than the one the test month asks.

THE ARMS
    Every arm sees the SAME 190 features. Only the fitting changes.

    base            all prior games, unweighted - what production does today
    recency_hl*     exponential sample weights by game age, half-life in days.
                    730 / 365 / 180 spans "barely discount" to "last winter
                    matters twice as much as the one before"
    recent_2s       only the two most recent seasons, hard cut. The blunt
                    version of the same idea, kept because a hard cut and a
                    soft decay fail differently
    phase_match     weights training games by how close their team-games-played
                    is to the test month's. This is the owner's hypothesis made
                    measurable: if a 75th-game prediction wants 75th-game
                    training rows, this arm should win
    margin_prob     ignore the classifier; convert the point-diff regressor's
                    predicted margin into a win probability with a logistic map
                    fitted on the calibration slice. Sports models often do
                    better predicting the margin and reading the sign off it
    blend           average of base and margin_prob, the cheap ensemble of two
                    different views of the same game

DISCIPLINE
    Same folds, same seeds, same burn-in filtering, same paired comparison as
    walk_forward.py - an arm is scored against base within each (month, seed)
    cell, never as a difference of averages. Nothing here touches the test
    month: weights are a function of training-row age or of the FOLD's date,
    both known before the games are played.

    This screens at five seeds. An arm that survives gets rerun at fifteen
    before anyone believes it - the availability family looked real at 105
    cells and was gone at 315.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "training_protocol.json")

SEEDS = [42, 49, 56, 63, 70]
HALF_LIVES = [730, 365, 180]
RECENT_SEASONS = 2
PHASE_SIGMA = 12.0        # games; the width of the phase-similarity kernel
MIN_WEIGHT = 0.02         # no training row is ever fully silenced


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wf = _load_sibling("walk_forward")
cal = _load_sibling("calibration")


# ---------------------------------------------------------------------------
#  Weight schemes - each returns one weight per training row
# ---------------------------------------------------------------------------
def weights_uniform(train, test):
    return None


def weights_recency(half_life_days):
    """Exponential decay in days before the fold, normalised to mean 1.

    Normalised because LightGBM's leaf-count and early-stopping behaviour track
    the total weight; leaving the sum free would confound "weighted differently"
    with "trained on less".
    """
    def scheme(train, test):
        age = (test["game_date"].min() - train["game_date"]).dt.days.astype(float)
        w = np.power(0.5, age / half_life_days)
        w = np.maximum(w, MIN_WEIGHT)
        return w / w.mean()
    return scheme


def weights_phase(train, test):
    """Weight a training row by how close its season phase is to the test month.

    Phase is the smaller of the two teams' games played, the same quantity
    burn-in filters on. A test month where teams average their 60th game
    upweights training rows from around anyone's 60th game, whatever season
    they came from.
    """
    target = float(np.median(np.minimum(test["home_gp"], test["away_gp"])))
    phase = np.minimum(train["home_gp"], train["away_gp"]).astype(float)
    w = np.exp(-0.5 * ((phase - target) / PHASE_SIGMA) ** 2)
    w = np.maximum(w, MIN_WEIGHT)
    return w / w.mean()


def subset_recent_seasons(train, n_seasons):
    keep = sorted(train["season"].unique())[-n_seasons:]
    return train[train["season"].isin(keep)]


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
def fit_weighted_classifier(train_df, features, seed, weight_scheme, test_df):
    """Same three-way split as the harness, with per-row weights.

    The weights are computed on the FULL training frame first and then split,
    so the fit and early-stopping slices are weighted consistently. Computing
    them per slice would give the early-stopping rows a different scale and
    stop on a different objective than the one being fitted.
    """
    tr = train_df.sort_values("game_date")
    w_all = None if weight_scheme is None else np.asarray(weight_scheme(tr, test_df))

    n = len(tr)
    a, b = int(n * 0.80), int(n * 0.90)
    fit_df, stop_df, cal_df = tr.iloc[:a], tr.iloc[a:b], tr.iloc[b:]

    X_fit, y_fit = cal.Xy(fit_df, features)
    X_stop, y_stop = cal.Xy(stop_df, features)
    model = lgb.LGBMClassifier(random_state=seed, **wf.BASE_PARAMS)
    model.fit(X_fit, y_fit,
              sample_weight=None if w_all is None else w_all[:a],
              eval_set=[(X_stop, y_stop)],
              eval_sample_weight=None if w_all is None else [w_all[a:b]],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
    return model, cal_df


def margin_to_probability(reg, features, cal_df, test_df):
    """Turn a predicted point margin into a win probability.

    The map is a one-variable logistic fitted on the calibration slice - the
    same slice the classifier's calibrator uses, and never the test month. A
    fixed textbook constant would be a guess about this league in these seasons;
    fitting it costs one parameter and makes it a measurement.
    """
    m_cal = reg.predict(cal_df[features].apply(pd.to_numeric, errors="coerce").fillna(0))
    y_cal = cal_df["home_win"].astype(float).values
    if len(np.unique(y_cal)) < 2:
        return None
    mapper = LogisticRegression(C=1e6, solver="lbfgs")
    mapper.fit(m_cal.reshape(-1, 1), y_cal)
    m_test = reg.predict(test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0))
    return np.clip(mapper.predict_proba(m_test.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def calibrated(model, features, cal_df, p_test):
    """Apply the better of isotonic/Platt, chosen on the calibration slice."""
    X_cal, y_cal = cal.Xy(cal_df, features)
    if len(np.unique(y_cal)) < 2:
        return p_test
    p_cal = model.predict_proba(X_cal)[:, 1]
    calibrators = cal.fit_calibrators(p_cal, y_cal)
    pick = min(calibrators, key=lambda k: brier_score_loss(
        y_cal, cal.apply_calibrator(k, calibrators[k], p_cal)))
    return cal.apply_calibrator(pick, calibrators[pick], p_test)


def run_cell(month, train, test, features, seed):
    """One (fold, seed): every protocol arm, scored at every burn-in."""
    probs = {}

    schemes = {"base": None, "phase_match": weights_phase}
    for hl in HALF_LIVES:
        schemes[f"recency_hl{hl}"] = weights_recency(hl)

    base_model = base_cal_df = None
    for name, scheme in schemes.items():
        model, cal_df = fit_weighted_classifier(train, features, seed, scheme, test)
        X_test, _ = cal.Xy(test, features)
        p = model.predict_proba(X_test)[:, 1]
        probs[name] = p
        probs[f"{name}+cal"] = calibrated(model, features, cal_df, p)
        if name == "base":
            base_model, base_cal_df = model, cal_df

    recent = subset_recent_seasons(train, RECENT_SEASONS)
    if len(recent) >= wf.MIN_TRAIN_GAMES:
        model, cal_df = fit_weighted_classifier(recent, features, seed, None, test)
        X_test, _ = cal.Xy(test, features)
        p = model.predict_proba(X_test)[:, 1]
        probs["recent_2s"] = p
        probs["recent_2s+cal"] = calibrated(model, features, cal_df, p)

    reg = wf.fit_regressor(train, features, "point_diff", seed)
    p_margin = margin_to_probability(reg, features, base_cal_df, test)
    if p_margin is not None:
        probs["margin_prob"] = p_margin
        probs["blend"] = 0.5 * (probs["base"] + p_margin)
        probs["blend+cal"] = 0.5 * (probs["base+cal"] + p_margin)

    y = test["home_win"].astype(float).values
    seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
    cell = {"month": month, "seed": seed, "n_train": int(len(train)),
            "by_burn_in": {}}
    for burn in wf.BURN_INS:
        keep = seasoned >= burn
        if keep.sum() < wf.MIN_TEST_GAMES or len(np.unique(y[keep])) < 2:
            continue
        entry = {"n_test": int(keep.sum())}
        for name, p in probs.items():
            entry[name] = wf.clf_scores(y[keep], p[keep])
        cell["by_burn_in"][str(burn)] = entry
    return cell


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------
def paired_delta(cells, burn, arm, base="base", metric="accuracy"):
    """Arm minus base within each cell, which is the only fair comparison.

    Cells differ enormously - a month where the favourites all held is easier
    for every arm at once - so a difference of averages mostly measures which
    months each arm happened to be scored on.
    """
    diffs = []
    for c in cells:
        e = c["by_burn_in"].get(burn)
        if e and arm in e and base in e:
            diffs.append(e[arm][metric] - e[base][metric])
    if not diffs:
        return None
    d = np.array(diffs, dtype=float)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "n": len(d), "wins": int((d > 0).sum())}


def summarise(cells, burn="10"):
    arms = sorted({a for c in cells for a in c["by_burn_in"].get(burn, {})
                   if a != "n_test"})
    rows = []
    for arm in arms:
        vals = [c["by_burn_in"][burn][arm]["accuracy"] for c in cells
                if burn in c["by_burn_in"] and arm in c["by_burn_in"][burn]]
        delta = paired_delta(cells, burn, arm)
        rows.append({"arm": arm, "accuracy": float(np.mean(vals)),
                     "n_cells": len(vals),
                     "delta_vs_base": delta["mean"] if delta else 0.0,
                     "se": delta["se"] if delta else 0.0,
                     "wins": delta["wins"] if delta else 0})
    return sorted(rows, key=lambda r: -r["accuracy"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl"))
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--out", default=REPORT_PATH)
    args = parser.parse_args()

    dataset, features, _groups = wf.load_any_dataset(args.dataset)
    dataset = wf.add_team_game_index(dataset)
    folds = wf.month_folds(dataset, args.months)
    seeds = SEEDS[:args.seeds]
    print(f"{len(folds)} ay x {len(seeds)} seed = {len(folds) * len(seeds)} hucre, "
          f"{len(features)} feature", flush=True)

    t0 = time.time()
    cells = []
    for i, (month, train, test) in enumerate(folds, 1):
        for seed in seeds:
            cells.append(run_cell(month, train, test, features, seed))
        done = i * len(seeds)
        rate = (time.time() - t0) / done
        print(f"  [{i}/{len(folds)}] {month}  n_train={len(train):,} "
              f"n_test={len(test)}  ({rate:.1f} sn/hucre, "
              f"kalan ~{rate * (len(folds) - i) * len(seeds) / 60:.0f} dk)", flush=True)

    report = {"protocol": "training-protocol arms; identical features, weighted fits",
              "seeds": seeds, "half_lives": HALF_LIVES,
              "recent_seasons": RECENT_SEASONS, "phase_sigma": PHASE_SIGMA,
              "n_cells": len(cells), "cells": cells,
              "summary_burn10": summarise(cells, "10"),
              "summary_burn0": summarise(cells, "0")}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"\n=== burn-in 10, {len(cells)} hucre ===")
    print(f"{'kol':18} {'isabet':>8} {'base farki':>12} {'se':>8} {'onde':>10}")
    for r in report["summary_burn10"]:
        print(f"{r['arm']:18} {r['accuracy']:8.4f} {r['delta_vs_base']:+12.4f} "
              f"{r['se']:8.4f} {r['wins']:6}/{r['n_cells']}")
    print(f"\nYazildi: {args.out}  ({time.time() - t0:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
