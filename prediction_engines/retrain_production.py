"""
Retrain the five production models on pre-game information only.

TWO THINGS THE PREVIOUS PRODUCTION RUN GOT WRONG
    1. The roster was read off the finished game. The impact and matchup
       features were aggregated over the players who APPEARED in the game, and
       how many players appear is a function of the result: the count
       correlates 0.64 with the margin because blowouts empty both benches.
       roster_impact_l10_sum - a sum over that set - was the point-differential
       model's most important feature. pregame_roster.py rebuilds the family
       from the pre-game team sheet (listed players minus the injury-report
       designations), weighted by trailing minutes; the dataset this script
       trains on is that rebuild (output/engineered_dataset_pregame.pkl).
    2. The held-out games chose the tree count. Every member was early-stopped
       against the 308-game test slice, so the "out-of-sample" metrics were
       read off games that had tuned the model. Measured effect: small (about
       0.3 points of accuracy, 0.1 points of margin MAE), but it meant the
       badge on the dashboard was not true. Now the tree count is chosen on the
       last tenth of the TRAINING rows and the member is refitted on all of
       them with that count; the test slice is touched once, to score.

FEATURE SET
    The 190 columns production always used, with the six roster/matchup
    columns in their pre-game form, plus the six "who is out tonight" columns
    (roster_avail_minutes, roster_missing_impact x home/away/diff) when the
    feature set is base+out. Which set to ship is decided by
    consistency_lab.py, which compares them paired over the same walk-forward
    cells; --feature-set overrides.

SEED ENSEMBLE
    Each target is an average of ENSEMBLE_SIZE models differing only in their
    random seed. Over 21 monthly folds accuracy rose monotonically with the
    number of seeds (+0.0070 / +0.0088 / +0.0103 at 3 / 5 / 10), the
    dose-response bagging predicts. It steadies the decision boundary; it does
    not touch the month-to-month spread, 86% of which is genuine difficulty.

CALIBRATOR AND BLEND
    Fitted on a slice held out from both the fit and the early stopping, as
    before. The blend (classifier + margin view + logistic view, averaged in
    probability space) is what the dashboard quotes; it was worth +0.0183
    accuracy over the classifier alone across 430 walk-forward cells.

Run:  py prediction_engines/retrain_production.py [--feature-set base|base+out]
Outputs: models/{target}_model_2025_26.pkl, models/{target}_ensemble_2025_26.pkl,
         models/{target}_feat_imp_2025_26.png, models/home_win_calibrator_2025_26.pkl,
         models/blend_2025_26.pkl, models/feature_list_2025_26.json,
         output/metrics_2025_26.json
"""
import os
import sys
import json
import time
import pickle
import argparse
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss, mean_absolute_error, mean_squared_error)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_2025_26.json")
LAB_REPORT_PATH = os.path.join(OUTPUT_DIR, "consistency_lab.json")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "home_win_calibrator_2025_26.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list_2025_26.json")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]
TARGET_SEASON = "2025_2026"
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
ENSEMBLE_SIZE = 10
SEEDS = [42 + 7 * i for i in range(ENSEMBLE_SIZE)]
STOP_SHARE = 0.10          # last tenth of the training rows chooses the tree count
MIN_TREES = 20
FEATURE_SETS = ("base", "base+out")
DEFAULT_FEATURE_SET = "base+out"

# RECIPE "reg+mono" - the winner of consistency_lab.py's second round (54
# paired cells against pregame+out): classifier accuracy +0.0114 +/- 0.0045,
# Brier -0.0061 +/- 0.0014, log loss 0.6088 (best of every arm), winner-vs-
# margin contradictions 8.9% against 13.0%, and the honest AUC (0.743) back to
# where the leaky roster had put it. A smaller, slower, bagged learner plus
# monotone constraints on every diff_ column, signed from the training rows.
# Only the winner and margin models were measured under it, so the three score
# models keep the base parameters.
REGULARIZED_PARAMS = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15,
                          min_child_samples=60, reg_lambda=5.0, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
RECIPES = ("base", "reg+mono")
DEFAULT_RECIPE = "reg+mono"
RECIPE_TARGETS = ("home_win", "point_diff")
MONOTONE_MIN_CORR = 0.02
LAB_SOURCES = {   # (recipe, feature_set) -> (report, arm) the honest numbers come from
    ("base", "base"): (os.path.join(OUTPUT_DIR, "consistency_lab.json"), "pregame"),
    ("base", "base+out"): (os.path.join(OUTPUT_DIR, "consistency_lab.json"), "pregame+out"),
    ("reg+mono", "base+out"): (os.path.join(OUTPUT_DIR, "consistency_lab_round2.json"),
                               "reg+mono+out"),
}


# SeedEnsemble is imported, not defined here. Defining it in a script that is
# run directly makes every pickled model reference __main__.SeedEnsemble, which
# only this script can resolve - and that broke the dashboard on startup.
_spec = importlib.util.spec_from_file_location(
    "ensemble_model", os.path.join(HERE, "ensemble_model.py"))
_ensemble_module = importlib.util.module_from_spec(_spec)
sys.modules["ensemble_model"] = _ensemble_module
_spec.loader.exec_module(_ensemble_module)
SeedEnsemble = _ensemble_module.SeedEnsemble


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chrono_split(dataset):
    """Context seasons plus the first 75% of the target season train; the last
    25% is held out. Same split the previous production runs used, so the
    headline numbers stay comparable."""
    ctx = dataset[dataset["season"] != TARGET_SEASON]
    tgt = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt) * 0.75)
    train = pd.concat([ctx, tgt.iloc[:split]]) if not ctx.empty else tgt.iloc[:split]
    return train.sort_values("game_date"), tgt.iloc[split:]


def Xy(df, features, target):
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, df[target].astype(float)


def select_features(blob, feature_set):
    base = list(blob["features"])
    if feature_set == "base":
        return base
    extra = [c for c in (blob.get("feature_groups") or {}).get("pregame_extra", [])
             if c in blob["dataset"].columns]
    if not extra:
        raise RuntimeError("dataset'te pregame_extra grubu yok - once "
                           "pregame_roster.py calistirin")
    return base + extra


def monotone_signs(train_df, features):
    """+1 / -1 per diff_ column from its Spearman sign against the margin on
    the training rows, 0 for every other column and for ones too weak to call.
    Same rule consistency_lab.py measured."""
    margin = train_df["point_diff"].astype(float).rank()
    signs = []
    for f in features:
        if not f.startswith("diff_"):
            signs.append(0)
            continue
        x = pd.to_numeric(train_df[f], errors="coerce").fillna(0)
        if x.std() == 0:
            signs.append(0)
            continue
        rho = np.corrcoef(x.rank(), margin)[0, 1]
        signs.append(int(np.sign(rho)) if abs(rho) >= MONOTONE_MIN_CORR else 0)
    return signs


def recipe_settings(recipe, target, train_df, features):
    """(params, extra fit kwargs) for one target under the chosen recipe."""
    if recipe == "reg+mono" and target in RECIPE_TARGETS:
        return REGULARIZED_PARAMS, {"monotone_constraints": monotone_signs(train_df, features),
                                    "monotone_constraints_method": "intermediate"}
    return PARAMS, {}


def fit_members(make, train_df, features, target, metric, params=PARAMS, extra=None):
    """One model per seed, with a tree count the test set had no say in.

    The last tenth of the training rows (chronologically) early-stops a probe
    fit on the first nine tenths; the member is then refitted on every training
    row with the tree count the probe settled on. Two fits per seed, and the
    held-out slice is never an argument to either.
    """
    extra = extra or {}
    cut = int(len(train_df) * (1 - STOP_SHARE))
    fit_df, stop_df = train_df.iloc[:cut], train_df.iloc[cut:]
    X_fit, y_fit = Xy(fit_df, features, target)
    X_stop, y_stop = Xy(stop_df, features, target)
    X_all, y_all = Xy(train_df, features, target)
    members, tree_counts = [], []
    for seed in SEEDS:
        probe = make(seed, {**params, **extra})
        probe.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric=metric,
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        n_trees = max(int(probe.n_estimators_), MIN_TREES)
        final = make(seed, {**params, **extra, "n_estimators": n_trees})
        final.fit(X_all, y_all)
        members.append(final)
        tree_counts.append(n_trees)
    return members, tree_counts


def make_classifier(seed, params):
    return lgb.LGBMClassifier(random_state=seed, **params)


def make_regressor(seed, params):
    return lgb.LGBMRegressor(random_state=seed, **params)


def honest_numbers(recipe, feature_set):
    """Walk-forward numbers for the shipped arm, read from the lab report so
    the dashboard never quotes a figure this script did not measure."""
    source = LAB_SOURCES.get((recipe, feature_set))
    if source is None or not os.path.exists(source[0]):
        return None
    path, arm = source
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    block = (report.get("summary") or {}).get(arm)
    if not block or not block.get("accuracy"):
        return None

    def mean(key):
        v = block.get(key)
        return round(v["mean"], 4) if v else None
    return {"arm": arm, "n_cells": block["accuracy"]["n"],
            "accuracy_classifier": mean("accuracy"),
            "accuracy_blend3": mean("blend3.accuracy"),
            "auc": mean("auc"), "brier": mean("brier"),
            "point_diff_mae": mean("point_diff_mae"),
            "month_sd": round(block["spread"]["month_sd"], 4),
            "seed_range": round(block["spread"]["seed_range"], 4),
            "protocol": report.get("protocol", "")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=DEFAULT_FEATURE_SET)
    parser.add_argument("--recipe", choices=RECIPES, default=DEFAULT_RECIPE,
                        help="base: eski parametreler; reg+mono: duzenlilestirilmis + "
                             "monoton kisit (consistency_lab 2. tur kazanani)")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    blob = pd.read_pickle(args.dataset)
    if blob.get("roster") != "pregame":
        raise SystemExit(f"{args.dataset} mac-oncesi kadro ile kurulmamis "
                         "(roster != 'pregame'); once pregame_roster.py calistirin")
    dataset = blob["dataset"]
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    features = select_features(blob, args.feature_set)
    print(f"dataset: {len(dataset):,} mac, {dataset['season'].nunique()} sezon, "
          f"{len(features)} feature ({args.feature_set}, kadro: mac oncesi)")

    train_df, test_df = chrono_split(dataset)
    print(f"train {len(train_df):,} | held-out {len(test_df):,} "
          f"({test_df.game_date.min().date()} -> {test_df.game_date.max().date()}) "
          f"- held-out hic bir fit'e girmiyor")

    metrics, models, trees = {}, {}, {}
    for target in TARGETS:
        t0 = time.time()
        X_test, y_test = Xy(test_df, features, target)
        params, extra = recipe_settings(args.recipe, target, train_df, features)
        if target == "home_win":
            members, counts = fit_members(make_classifier, train_df, features, target, "auc",
                                          params, extra)
            win_params, win_extra = params, extra
            model = SeedEnsemble(members, "clf")
            probs = model.predict_proba(X_test)[:, 1]
            metrics[target] = {
                "accuracy": float(accuracy_score(y_test, (probs > 0.5).astype(int))),
                "auc": float(roc_auc_score(y_test, probs)),
                "brier": float(brier_score_loss(y_test, probs)),
                "log_loss": float(log_loss(y_test, probs, labels=[0.0, 1.0])),
                "baseline_home_rate": float(y_test.mean())}
            print(f"  home_win     acc={metrics[target]['accuracy']:.4f} "
                  f"auc={metrics[target]['auc']:.4f} "
                  f"brier={metrics[target]['brier']:.4f} "
                  f"agac={int(np.mean(counts))} ({time.time()-t0:.0f} sn)")
        else:
            members, counts = fit_members(make_regressor, train_df, features, target, "mae",
                                          params, extra)
            model = SeedEnsemble(members, "reg")
            preds = model.predict(X_test)
            metrics[target] = {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))}
            print(f"  {target:12} mae={metrics[target]['mae']:.3f} "
                  f"rmse={metrics[target]['rmse']:.3f} agac={int(np.mean(counts))} "
                  f"({time.time()-t0:.0f} sn)")

        models[target] = model
        trees[target] = counts
        # SeedEnsemble lives in ensemble_model, but a consumer that only knows
        # LightGBM can still rebuild it from the members file.
        with open(os.path.join(MODEL_DIR, f"{target}_model_2025_26.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(MODEL_DIR, f"{target}_ensemble_2025_26.pkl"), "wb") as f:
            pickle.dump({"members": model.models, "kind": model.kind,
                         "seeds": SEEDS, "features": features}, f)
        imp = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
               .sort_values("importance", ascending=False).head(20))
        plt.figure(figsize=(9, 7))
        sns.barplot(x="importance", y="feature", data=imp)
        plt.title(f"Top features - {target} (9 sezon, mac oncesi kadro)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, f"{target}_feat_imp_2025_26.png"))
        plt.close()

    # Calibrator on a slice the classifier never saw: the last 10% of train.
    # It is fitted against an ensemble of the same size, so the mapping it
    # learns matches the shape of the probabilities production actually emits.
    cal = _load_sibling("calibration")
    fit_df, stop_df, cal_df = cal.three_way(train_df)
    X_fit, y_fit = Xy(fit_df, features, "home_win")
    X_stop, y_stop = Xy(stop_df, features, "home_win")
    # Same recipe as the shipped classifier, so the calibrator learns the shape
    # of the probabilities production actually emits.
    warm_members = []
    for seed in SEEDS:
        w = lgb.LGBMClassifier(random_state=seed, **win_params, **win_extra)
        w.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        warm_members.append(w)
    warm = SeedEnsemble(warm_members, "clf")
    X_cal, y_cal = Xy(cal_df, features, "home_win")
    p_cal = warm.predict_proba(X_cal)[:, 1]
    calibrators = cal.fit_calibrators(p_cal, y_cal)
    pick = min(calibrators, key=lambda k: brier_score_loss(
        y_cal, cal.apply_calibrator(k, calibrators[k], p_cal)))
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump({"method": pick, "calibrator": calibrators[pick]}, f)
    print(f"\nkalibrator: {pick} ({len(cal_df):,} mac uzerinde)")

    # ---- the blended forecaster -------------------------------------------
    # The margin view's map and the calibrator share the held-out slice, and
    # both are read from models that never saw it - a mapper fitted on the
    # fitting rows would be reading its own training error as uncertainty.
    warm_margin = []
    y_fit_m = fit_df["point_diff"].astype(float)
    y_stop_m = stop_df["point_diff"].astype(float)
    margin_params, margin_extra = recipe_settings(args.recipe, "point_diff", train_df, features)
    for seed in SEEDS:
        r = lgb.LGBMRegressor(random_state=seed, **margin_params, **margin_extra)
        r.fit(X_fit, y_fit_m, eval_set=[(X_stop, y_stop_m)], eval_metric="mae",
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
        warm_margin.append(r)
    mapper = LogisticRegression(C=1e6, solver="lbfgs")
    mapper.fit(SeedEnsemble(warm_margin, "reg").predict(X_cal).reshape(-1, 1), y_cal)

    X_train_win, y_train_win = Xy(train_df, features, "home_win")
    logistic = make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000))
    logistic.fit(X_train_win, y_train_win)

    blend = _ensemble_module.BlendedForecaster(
        models["home_win"], models["point_diff"], mapper, logistic)
    with open(os.path.join(MODEL_DIR, "blend_2025_26.pkl"), "wb") as f:
        pickle.dump({"classifier_members": models["home_win"].models,
                     "margin_members": models["point_diff"].models,
                     "margin_mapper": mapper, "logistic": logistic,
                     "features": features}, f)

    X_hold, y_hold = Xy(test_df, features, "home_win")
    p_blend = blend.predict_proba(X_hold)[:, 1]
    metrics["blend"] = {
        "accuracy": float(accuracy_score(y_hold, (p_blend > 0.5).astype(int))),
        "auc": float(roc_auc_score(y_hold, p_blend)),
        "brier": float(brier_score_loss(y_hold, p_blend))}
    print(f"blend        acc={metrics['blend']['accuracy']:.4f} "
          f"auc={metrics['blend']['auc']:.4f} "
          f"brier={metrics['blend']['brier']:.4f}")

    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_set": args.feature_set, "recipe": args.recipe,
                   "roster": "pregame", "dataset": os.path.basename(args.dataset),
                   "features": features}, f, indent=1)

    payload = {
        "trained_on": sorted(dataset["season"].unique().tolist()),
        "dataset": os.path.basename(args.dataset), "roster": "pregame",
        "feature_set": args.feature_set, "recipe": args.recipe,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_features": len(features),
        "split_date": str(test_df.game_date.min().date()),
        "ensemble_size": ENSEMBLE_SIZE,
        "tree_counts": {t: [int(c) for c in counts] for t, counts in trees.items()},
        "early_stopping": "chronological last 10% of train chooses the tree count; "
                          "members refitted on all training rows; held-out slice "
                          "used only for scoring",
        "metrics": metrics,
        "calibrator": pick,
        "honest_walk_forward": honest_numbers(args.recipe, args.feature_set),
        "note": "tek-split rakamlari sezonun son bes haftasindan geliyor ve iyimser; "
                "baglayici olan honest_walk_forward",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Yazildi: {METRICS_PATH}")
    print(f"Yazildi: {FEATURE_LIST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
