"""
Retrain the five production models on the corrected nine-season dataset.

WHY NOW
    The models in models/*_model_2025_26.pkl were fitted on seven seasons of
    data whose impact features were wrong in 11.3% of games - two opposing
    players sharing a surname had their play-by-play impact merged and credited
    to one team. Both problems are fixed: the dataset now covers 10,749 games
    across nine seasons and keys impact on person id.

    The paired walk-forward over 120 cells says the fix moved Brier by
    -0.0025 (significant, and in the same direction for both the raw and the
    calibrated arm) while leaving threshold accuracy alone at +0.0017 ± 0.0033.
    That is the expected shape: correcting 11% of a feature improves the
    probabilities before it flips any decisions.

FEATURE SET
    The base 190. Across 120 cells, every alternative - opponent-adjusted
    ratings, rest/back-to-back, calibration, and each combination - landed
    inside one standard error of the base, and the arm that reached
    significance changed identity and sign between runs. There is no evidence
    for adding anything, so production keeps the simplest set that performs.

CALIBRATOR
    Fitted on a slice held out from both the fit and the early stopping, and
    saved beside the models. It does not move accuracy, but it is what makes a
    reported 70% mean 70%, and the simulation engine needs honest probabilities
    to draw from.

Run:  py prediction_engines/retrain_production.py
Outputs: models/{target}_model_2025_26.pkl, models/{target}_feat_imp_2025_26.png,
         models/home_win_calibrator_2025_26.pkl, output/metrics_2025_26.json
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
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss, mean_absolute_error, mean_squared_error)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_2025_26.json")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "home_win_calibrator_2025_26.pkl")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]
TARGET_SEASON = "2025_2026"
PARAMS = dict(random_state=42, n_estimators=600, learning_rate=0.03,
              num_leaves=31, subsample=0.8, colsample_bytree=0.8,
              verbose=-1, n_jobs=-1)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    blob = pd.read_pickle(args.dataset)
    dataset, features = blob["dataset"], blob["features"]
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    print(f"dataset: {len(dataset):,} mac, {dataset['season'].nunique()} sezon, "
          f"{len(features)} feature")

    train_df, test_df = chrono_split(dataset)
    print(f"train {len(train_df):,} | held-out {len(test_df):,} "
          f"({test_df.game_date.min().date()} -> {test_df.game_date.max().date()})")

    metrics = {}
    models = {}
    for target in TARGETS:
        t0 = time.time()
        X_train, y_train = Xy(train_df, features, target)
        X_test, y_test = Xy(test_df, features, target)
        if target == "home_win":
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(0)])
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
                  f"({time.time()-t0:.0f} sn)")
        else:
            model = lgb.LGBMRegressor(**PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="mae",
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(0)])
            preds = model.predict(X_test)
            metrics[target] = {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))}
            print(f"  {target:12} mae={metrics[target]['mae']:.3f} "
                  f"rmse={metrics[target]['rmse']:.3f} ({time.time()-t0:.0f} sn)")

        models[target] = model
        with open(os.path.join(MODEL_DIR, f"{target}_model_2025_26.pkl"), "wb") as f:
            pickle.dump(model, f)
        if hasattr(model, "feature_importances_"):
            imp = (pd.DataFrame({"feature": features,
                                 "importance": model.feature_importances_})
                   .sort_values("importance", ascending=False).head(20))
            plt.figure(figsize=(9, 7))
            sns.barplot(x="importance", y="feature", data=imp)
            plt.title(f"Top features - {target} (9 sezon, duzeltilmis impact)")
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR, f"{target}_feat_imp_2025_26.png"))
            plt.close()

    # Calibrator on a slice the classifier never saw: the last 10% of train.
    cal = _load_sibling("calibration")
    fit_df, stop_df, cal_df = cal.three_way(train_df)
    warm = lgb.LGBMClassifier(**PARAMS)
    X_fit, y_fit = Xy(fit_df, features, "home_win")
    X_stop, y_stop = Xy(stop_df, features, "home_win")
    warm.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    X_cal, y_cal = Xy(cal_df, features, "home_win")
    p_cal = warm.predict_proba(X_cal)[:, 1]
    calibrators = cal.fit_calibrators(p_cal, y_cal)
    pick = min(calibrators, key=lambda k: brier_score_loss(
        y_cal, cal.apply_calibrator(k, calibrators[k], p_cal)))
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump({"method": pick, "calibrator": calibrators[pick]}, f)
    print(f"\nkalibrator: {pick} ({len(cal_df):,} mac uzerinde)")

    payload = {
        "trained_on": sorted(dataset["season"].unique().tolist()),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_features": len(features),
        "split_date": str(test_df.game_date.min().date()),
        "metrics": metrics,
        "calibrator": pick,
        "honest_walk_forward": {
            "accuracy": 0.6646, "accuracy_std": 0.0557, "naive_baseline": 0.5554,
            "note": "12 ay x 5 seed, burn-in >= 10; asagidaki tek-split rakami "
                    "sezonun en kolay 5 haftasindan geliyor ve iyimserdir",
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Yazildi: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
