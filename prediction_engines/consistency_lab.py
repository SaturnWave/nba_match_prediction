"""
Consistency lab: which changes make the forecasts more stable, measured on
pre-game information only.

THE TWO QUESTIONS
    1. How much of the model's measured skill came from reading the roster off
       the finished game? `observed` is the production feature set; `pregame`
       is the same 190 columns with the roster and matchup families rebuilt by
       pregame_roster.py from the pre-game team sheet. Paired within each
       (month, seed) cell, their gap is the size of the illusion.
    2. On the honest features, what makes the model more consistent? Not just
       more accurate on average, but with less spread from month to month and
       seed to seed, fewer contradictions between the winner and the margin,
       and probabilities that stay calibrated.

ARMS
    observed      production features, observed roster              (reference)
    pregame       the same columns, pre-game roster
    pregame+out   plus who is out tonight: roster_avail_minutes and
                  roster_missing_impact, the pre-game injury signal
    diff_only     pregame+out restricted to the diff_ and h2h_ columns - the
                  model sees the gap between the teams, not two levels
    regularized   pregame+out with a smaller, slower, bagged learner
    monotone      pregame+out with monotone constraints on every diff_ column,
                  signed from the training fold - a team that leads on a
                  statistic cannot be made LESS likely to win by leading more
    phase         pregame+out plus how many games each team has played this
                  season, so the model can discount cold October averages

    Round two (--arms, --all-margin), combining what round one found: the
    regularized learner and the monotone constraints each bought ranking
    quality (AUC +0.013 and +0.008 over pregame, Brier -0.003) without moving
    accuracy, and the who-is-out columns paid only inside the blend.
    reg+out       regularized learner on pregame+out
    reg+mono+out  regularized learner with the monotone constraints
    diff_mono     diff_only with the monotone constraints
    diff_reg_mono diff_only, regularized, monotone

PROTOCOL
    walk_forward.py's: expanding-origin monthly folds, a classifier fitted on
    the first 80% of prior games and early-stopped on the next 10%, the final
    10% kept for the calibrator and for the margin-to-probability map; a
    point-differential regressor fitted 90/10. Scored at burn-in 10. The
    observed, pregame and pregame+out arms also get the blended forecaster
    (classifier + margin view + logistic view), since that is what production
    serves.

    Nothing from the test month reaches any fit, and every arm is compared to
    another arm inside the same cell.

Run:  py prediction_engines/consistency_lab.py [--months 21] [--seeds 3] [--arms a,b,c]
                                               [--all-margin] [--base pregame] [--out ...]
Output: output/consistency_lab.json
"""
import argparse
import importlib.util
import itertools
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_absolute_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OBSERVED_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
PREGAME_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "consistency_lab.json")

OBSERVED_PREFIX = "obs_"
BURN_IN = 10
MIN_TEST_GAMES = 40
WITH_MARGIN = ("observed", "pregame", "pregame+out")   # arms that also fit the regressor + blend


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wf = _load_sibling("walk_forward")
cal = _load_sibling("calibration")
pregame = _load_sibling("pregame_roster")

BASE_PARAMS = dict(wf.BASE_PARAMS)
# Smaller leaves, more rows per leaf, half the columns per tree, real bagging
# (subsample needs subsample_freq to do anything), and a slower learning rate
# with room to stop early. Everything here trades fit for variance.
REGULARIZED_PARAMS = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                          min_child_samples=60, reg_lambda=5.0, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
MONOTONE_MIN_CORR = 0.02


# ---------------------------------------------------------------------------
#  Data: one frame carrying both roster regimes
# ---------------------------------------------------------------------------
def load_frame():
    """The pre-game dataset with the observed roster columns kept under obs_."""
    observed = pd.read_pickle(OBSERVED_PATH)
    blob = pd.read_pickle(PREGAME_PATH)
    pre = blob["dataset"]
    obs = observed["dataset"].copy()
    obs["game_id"] = obs["game_id"].astype(str).str.zfill(10)
    keep = ["game_id"] + [c for c in pregame.REPLACED_COLUMNS if c in obs.columns]
    obs_cols = obs[keep].rename(columns={c: OBSERVED_PREFIX + c for c in keep if c != "game_id"})
    frame = pre.merge(obs_cols, on="game_id", how="left")
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = wf.add_team_game_index(frame)
    base = list(blob["features"])
    extra = [c for c in blob["feature_groups"]["pregame_extra"] if c in frame.columns]
    return frame, base, extra


def arm_definitions(base, extra):
    replaced = set(pregame.REPLACED_COLUMNS)
    observed_feats = [OBSERVED_PREFIX + f if f in replaced else f for f in base]
    with_out = base + extra
    diff_only = [f for f in with_out if f.startswith(("diff_", "h2h_"))]
    return {
        "observed": {"features": observed_feats, "params": BASE_PARAMS},
        "pregame": {"features": base, "params": BASE_PARAMS},
        "pregame+out": {"features": with_out, "params": BASE_PARAMS},
        "diff_only": {"features": diff_only, "params": BASE_PARAMS},
        "regularized": {"features": with_out, "params": REGULARIZED_PARAMS},
        "monotone": {"features": with_out, "params": BASE_PARAMS, "monotone": True},
        "phase": {"features": with_out + ["home_gp", "away_gp"], "params": BASE_PARAMS},
        # round two
        "reg+out": {"features": with_out, "params": REGULARIZED_PARAMS},
        "reg+mono+out": {"features": with_out, "params": REGULARIZED_PARAMS, "monotone": True},
        "diff_mono": {"features": diff_only, "params": BASE_PARAMS, "monotone": True},
        "diff_reg_mono": {"features": diff_only, "params": REGULARIZED_PARAMS, "monotone": True},
    }


# ---------------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------------
def X_of(df, features):
    return df[features].apply(pd.to_numeric, errors="coerce").fillna(0)


def monotone_signs(fit_df, features):
    """+1 / -1 for a diff_ column by its Spearman sign against the margin on
    the fitting rows, 0 for everything else and for columns too weak to call."""
    margin = fit_df["point_diff"].astype(float).rank()
    signs = []
    for f in features:
        if not f.startswith("diff_"):
            signs.append(0)
            continue
        x = pd.to_numeric(fit_df[f], errors="coerce").fillna(0)
        if x.std() == 0:
            signs.append(0)
            continue
        rho = np.corrcoef(x.rank(), margin)[0, 1]
        signs.append(int(np.sign(rho)) if abs(rho) >= MONOTONE_MIN_CORR else 0)
    return signs


def fit_arm(train, features, params, seed, monotone=False, with_margin=False):
    """Classifier (+ regressor and blend parts) for one arm in one cell."""
    tr = train.sort_values("game_date")
    fit_df, stop_df, cal_df = cal.three_way(tr)
    extra = {}
    if monotone:
        extra = {"monotone_constraints": monotone_signs(fit_df, features),
                 "monotone_constraints_method": "intermediate"}
    X_fit, X_stop = X_of(fit_df, features), X_of(stop_df, features)
    clf = lgb.LGBMClassifier(random_state=seed, **params, **extra)
    clf.fit(X_fit, fit_df["home_win"].astype(float),
            eval_set=[(X_stop, stop_df["home_win"].astype(float))], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    out = {"clf": clf, "cal_df": cal_df}
    if not with_margin:
        return out

    # 90/10 for the regressor, like walk_forward.fit_regressor; the margin map
    # and the logistic view follow training_protocol.py exactly.
    cut = int(len(tr) * 0.9)
    r_fit, r_stop = tr.iloc[:cut], tr.iloc[cut:]
    reg = lgb.LGBMRegressor(random_state=seed, **params, **extra)
    reg.fit(X_of(r_fit, features), r_fit["point_diff"].astype(float),
            eval_set=[(X_of(r_stop, features), r_stop["point_diff"].astype(float))],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    mapper = LogisticRegression(C=1e6, solver="lbfgs")
    mapper.fit(reg.predict(X_of(cal_df, features)).reshape(-1, 1),
               cal_df["home_win"].astype(float))
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    logistic.fit(X_of(tr, features), tr["home_win"].astype(float))
    out.update({"reg": reg, "mapper": mapper, "logistic": logistic})
    return out


def clf_scores(y, p):
    return {"accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0.0, 1.0]))}


def score_arm(fitted, test, features, keep):
    X = X_of(test, features)
    y = test["home_win"].astype(float).values[keep]
    p = fitted["clf"].predict_proba(X)[:, 1][keep]
    entry = clf_scores(y, p)
    if "reg" in fitted:
        margin = fitted["reg"].predict(X)[keep]
        actual = test["point_diff"].astype(float).values[keep]
        entry["point_diff_mae"] = float(mean_absolute_error(actual, margin))
        entry["margin_sign_accuracy"] = float(np.mean((margin > 0) == (actual > 0)))
        entry["contradiction_rate"] = float(np.mean((p > 0.5) != (margin > 0)))
        p_margin = np.clip(fitted["mapper"].predict_proba(margin.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        p_logit = np.clip(fitted["logistic"].predict_proba(X)[:, 1][keep], 1e-6, 1 - 1e-6)
        entry["blend3"] = clf_scores(y, np.mean([p, p_margin, p_logit], axis=0))
    return entry


def run_cell(month, train, test, arms, seed, with_margin=WITH_MARGIN):
    seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
    keep = seasoned >= BURN_IN
    cell = {"month": month, "seed": seed, "n_train": int(len(train)),
            "n_test": int(keep.sum()), "arms": {}}
    if keep.sum() < MIN_TEST_GAMES or test["home_win"][keep].nunique() < 2:
        return cell
    for name, spec in arms.items():
        t0 = time.time()
        fitted = fit_arm(train, spec["features"], spec["params"], seed,
                         monotone=spec.get("monotone", False),
                         with_margin=name in with_margin)
        cell["arms"][name] = score_arm(fitted, test, spec["features"], keep)
        cell["arms"][name]["n_trees"] = int(fitted["clf"].n_estimators_)
        cell["arms"][name]["seconds"] = round(time.time() - t0, 1)
    return cell


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------
def _metric(entry, metric):
    if metric.startswith("blend3."):
        return entry.get("blend3", {}).get(metric.split(".", 1)[1])
    return entry.get(metric)


def collect(cells, arm, metric):
    vals = [_metric(c["arms"][arm], metric) for c in cells if arm in c["arms"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def paired(cells, arm, base, metric):
    d = [_metric(c["arms"][arm], metric) - _metric(c["arms"][base], metric)
         for c in cells if arm in c["arms"] and base in c["arms"]
         and _metric(c["arms"][arm], metric) is not None and _metric(c["arms"][base], metric) is not None]
    if not d:
        return None
    d = np.array(d, float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return {"mean": float(d.mean()), "se": se, "n": len(d), "ahead": int((d > 0).sum())}


def spreads(cells, arm, metric="accuracy"):
    """Month-to-month sd of the seed-mean, and mean within-month seed range."""
    by_month = {}
    for c in cells:
        if arm in c["arms"]:
            by_month.setdefault(c["month"], []).append(_metric(c["arms"][arm], metric))
    means = [np.mean(v) for v in by_month.values()]
    ranges = [max(v) - min(v) for v in by_month.values() if len(v) > 1]
    return {"month_sd": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "seed_range": float(np.mean(ranges)) if ranges else float("nan")}


PAIRED_METRICS = ("accuracy", "auc", "brier", "log_loss", "point_diff_mae",
                  "blend3.accuracy", "blend3.auc", "blend3.brier")


def summarise(cells, arms, base="pregame"):
    summary = {"_base": base}
    for arm in arms:
        block = {m: collect(cells, arm, m) for m in
                 ("accuracy", "auc", "brier", "log_loss", "point_diff_mae",
                  "contradiction_rate", "margin_sign_accuracy", "n_trees",
                  "blend3.accuracy", "blend3.auc", "blend3.brier")}
        block["spread"] = spreads(cells, arm)
        block["vs_base"] = {m: paired(cells, arm, base, m) for m in PAIRED_METRICS}
        summary[arm] = block
    return summary


def print_summary(summary, arms):
    base = summary.get("_base", "pregame")
    print(f"\n{'kol':13} {'acc':>7} {'auc':>7} {'brier':>7} {'logloss':>8} {'mae':>6} "
          f"{'celiski':>7} {'aySD':>6} {'seedR':>6} {'blend3':>7} {'bl.brier':>8}  "
          f"| {base}'e gore: acc / brier / blend3 acc / blend3 brier")
    for arm in arms:
        s = summary[arm]

        def m(key, fmt="{:.4f}"):
            v = s.get(key)
            return fmt.format(v["mean"]) if v else "-"

        def delta(key):
            d = s["vs_base"].get(key)
            return f"{d['mean']:+.4f}±{d['se']:.4f}" if d else "-"
        print(f"{arm:13} {m('accuracy'):>7} {m('auc'):>7} {m('brier'):>7} {m('log_loss'):>8} "
              f"{m('point_diff_mae', '{:.2f}'):>6} {m('contradiction_rate', '{:.3f}'):>7} "
              f"{s['spread']['month_sd']:6.4f} {s['spread']['seed_range']:6.4f} "
              f"{m('blend3.accuracy'):>7} {m('blend3.brier'):>8}  | {delta('accuracy')} / "
              f"{delta('brier')} / {delta('blend3.accuracy')} / {delta('blend3.brier')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arms", default=None, help="virgulle ayrilmis kol adlari")
    parser.add_argument("--all-margin", action="store_true",
                        help="her kol icin marj modeli ve blend de kurulsun")
    parser.add_argument("--base", default="pregame",
                        help="eslesmis farklarin olculecegi kol")
    parser.add_argument("--out", default=REPORT_PATH)
    args = parser.parse_args()

    frame, base_features, extra = load_frame()
    arms = arm_definitions(base_features, extra)
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in wanted if a not in arms]
        if unknown:
            raise SystemExit(f"bilinmeyen kol: {unknown}; mevcut: {list(arms)}")
        arms = {k: arms[k] for k in wanted}
    else:
        arms = {k: v for k, v in arms.items() if k in
                ("observed", "pregame", "pregame+out", "diff_only",
                 "regularized", "monotone", "phase")}
    if args.base not in arms:
        raise SystemExit(f"--base {args.base} kollar arasinda degil: {list(arms)}")
    with_margin = set(arms) if args.all_margin else set(WITH_MARGIN)
    for name, spec in arms.items():
        print(f"  {name:13} {len(spec['features'])} feature"
              + (" + marj + blend" if name in with_margin else ""))

    folds = wf.month_folds(frame, args.months)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"\n{len(folds)} ay x {len(seeds)} seed = {len(folds) * len(seeds)} hucre, "
          f"{len(arms)} kol", flush=True)

    t0 = time.time()
    cells = []
    total = len(folds) * len(seeds)
    for i, ((month, train, test), seed) in enumerate(itertools.product(folds, seeds), 1):
        cell = run_cell(month, train, test, arms, seed, with_margin)
        if cell["arms"]:
            cells.append(cell)
            line = "  ".join(f"{k}={v['accuracy']:.3f}" for k, v in cell["arms"].items())
            rate = (time.time() - t0) / i
            print(f"[{i}/{total}] {month} seed={seed} n={cell['n_test']:3d}  {line}  "
                  f"(kalan ~{rate * (total - i) / 60:.0f} dk)", flush=True)
        if i % 3 == 0 or i == total:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"protocol": __doc__.split("PROTOCOL")[1].split("Run:")[0].strip(),
                           "burn_in": BURN_IN, "seeds": seeds,
                           "arm_sizes": {k: len(v["features"]) for k, v in arms.items()},
                           "n_cells": len(cells), "cells": cells,
                           "summary": summarise(cells, list(arms), args.base)}, f, indent=1)

    summary = summarise(cells, list(arms), args.base)
    print_summary(summary, list(arms))
    print(f"\nYazildi: {args.out}  ({(time.time() - t0) / 60:.0f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
