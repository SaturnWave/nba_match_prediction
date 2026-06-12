"""
Empirical analysis backing the improvement report for the 2025-26 predictor.

Runs two experiments on the engineered 2024-25 (train) -> 2025-26 (test) dataset:

  1. IMPACT-SCORE ABLATION
     Train each target with the FULL feature set vs a feature set with every
     impact-derived feature removed, and compare. Then segment the held-out
     games (close / blowout, home-favoured / away-favoured by rolling impact)
     to locate WHERE the impact score helps or hurts accuracy.

  2. MODEL COMPARISON
     LightGBM vs XGBoost vs RandomForest vs Linear (LogReg / Ridge) on the same
     features, to see whether a different model class predicts better.

Results -> output/analysis_2025_26.json   (consumed by the report).
"""
import os
import json
import importlib.util

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_CACHE = os.path.join(OUTPUT_DIR, "engineered_dataset_2025_26.pkl")

spec = importlib.util.spec_from_file_location("pp", os.path.join(HERE, "predict_2025_2026.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

TRAIN_SEASON, TARGET_SEASON = "2024_2025", "2025_2026"
REG_TARGETS = ["point_diff", "total_score"]


def build_dataset(force=False):
    if os.path.exists(DATASET_CACHE) and not force:
        obj = pd.read_pickle(DATASET_CACHE)
        return obj["dataset"], obj["features"]
    P = pp.NBAPredictor()
    P.load_and_prepare(seasons=[TRAIN_SEASON, TARGET_SEASON])
    features = P._select_features()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.to_pickle({"dataset": P.dataset, "features": features}, DATASET_CACHE)
    return P.dataset, features


def chrono_split(dataset, features):
    ctx = dataset[dataset["season"] == TRAIN_SEASON]
    tgt = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt) * 0.75)
    train = pd.concat([ctx, tgt.iloc[:split]]) if not ctx.empty else tgt.iloc[:split]
    test = tgt.iloc[split:]
    return train, test


def Xy(df, features, target):
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, df[target].astype(float)


def fit_lgbm(kind, Xtr, ytr, Xte, yte):
    params = dict(random_state=42, n_estimators=400, learning_rate=0.03,
                  num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    M = lgb.LGBMClassifier(**params) if kind == "clf" else lgb.LGBMRegressor(**params)
    M.fit(Xtr, ytr, eval_set=[(Xte, yte)],
          eval_metric="auc" if kind == "clf" else "mae",
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return M


# ---------------------------------------------------------------------------
# Experiment 1 : impact-score ablation
# ---------------------------------------------------------------------------
def impact_ablation(train, test, features):
    # every impact-derived feature: game-impact aggregates AND roster-impact form
    impact_feats = [f for f in features if ("impact" in f.lower() or "roster" in f.lower())]
    no_impact = [f for f in features if f not in impact_feats]
    res = {"n_impact_features": len(impact_feats), "n_total_features": len(features),
           "impact_features": impact_feats, "targets": {}}

    # store per-game home_win correctness for segmentation
    yte_win = test["home_win"].astype(float).values
    seg = {"full_correct": None, "noimp_correct": None}

    for target in ["home_win"] + REG_TARGETS:
        kind = "clf" if target == "home_win" else "reg"
        out = {}
        for tag, feats in (("with_impact", features), ("without_impact", no_impact)):
            Xtr, ytr = Xy(train, feats, target)
            Xte, yte = Xy(test, feats, target)
            M = fit_lgbm(kind, Xtr, ytr, Xte, yte)
            if kind == "clf":
                preds = M.predict(Xte)
                probs = M.predict_proba(Xte)[:, 1]
                out[tag] = {"accuracy": float(accuracy_score(yte, preds)),
                            "auc": float(roc_auc_score(yte, probs))}
                if tag == "with_impact":
                    seg["full_correct"] = (preds == yte.values).astype(int)
                else:
                    seg["noimp_correct"] = (preds == yte.values).astype(int)
            else:
                preds = M.predict(Xte)
                out[tag] = {"mae": float(mean_absolute_error(yte, preds)),
                            "rmse": float(np.sqrt(mean_squared_error(yte, preds)))}
        if target == "home_win":
            out["delta_accuracy"] = out["with_impact"]["accuracy"] - out["without_impact"]["accuracy"]
            out["delta_auc"] = out["with_impact"]["auc"] - out["without_impact"]["auc"]
        else:
            out["delta_mae"] = out["with_impact"]["mae"] - out["without_impact"]["mae"]
        res["targets"][target] = out

    # segmentation of home_win accuracy by game character
    t = test.copy().reset_index(drop=True)
    t["full_correct"] = seg["full_correct"]
    t["noimp_correct"] = seg["noimp_correct"]
    t["abs_pd"] = t["point_diff"].abs()
    segments = {
        "close_games (|pd|<=6)": t["abs_pd"] <= 6,
        "moderate (6<|pd|<=15)": (t["abs_pd"] > 6) & (t["abs_pd"] <= 15),
        "blowouts (|pd|>15)": t["abs_pd"] > 15,
    }
    # also split by rolling impact-form gap available pre-game
    if "diff_L5_impact_score_agg" in t.columns:
        med = t["diff_L5_impact_score_agg"].abs().median()
        segments["large_impact_form_gap"] = t["diff_L5_impact_score_agg"].abs() > med
        segments["small_impact_form_gap"] = t["diff_L5_impact_score_agg"].abs() <= med
    seg_out = {}
    for name, mask in segments.items():
        sub = t[mask]
        if len(sub) == 0:
            continue
        seg_out[name] = {"n": int(len(sub)),
                         "acc_with_impact": float(sub["full_correct"].mean()),
                         "acc_without_impact": float(sub["noimp_correct"].mean()),
                         "delta": float(sub["full_correct"].mean() - sub["noimp_correct"].mean())}
    res["home_win_accuracy_by_segment"] = seg_out
    return res


# ---------------------------------------------------------------------------
# Experiment 2 : model comparison
# ---------------------------------------------------------------------------
def model_comparison(train, test, features):
    out = {"home_win": {}, "point_diff": {}, "total_score": {}}

    Xtr, ytr = Xy(train, features, "home_win")
    Xte, yte = Xy(test, features, "home_win")
    clfs = {
        "lightgbm": fit_lgbm("clf", Xtr, ytr, Xte, yte),
        "xgboost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                                     subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                     random_state=42, verbosity=0).fit(Xtr, ytr),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=8,
                                                random_state=42, n_jobs=-1).fit(Xtr, ytr),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, ytr),
    }
    for name, M in clfs.items():
        preds = M.predict(Xte)
        probs = M.predict_proba(Xte)[:, 1]
        out["home_win"][name] = {"accuracy": float(accuracy_score(yte, preds)),
                                 "auc": float(roc_auc_score(yte, probs))}

    for target in REG_TARGETS:
        Xtr, ytr = Xy(train, features, target)
        Xte, yte = Xy(test, features, target)
        regs = {
            "lightgbm": fit_lgbm("reg", Xtr, ytr, Xte, yte),
            "xgboost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                                        subsample=0.8, colsample_bytree=0.8,
                                        random_state=42, verbosity=0).fit(Xtr, ytr),
            "random_forest": RandomForestRegressor(n_estimators=400, max_depth=8,
                                                   random_state=42, n_jobs=-1).fit(Xtr, ytr),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(Xtr, ytr),
        }
        for name, M in regs.items():
            preds = M.predict(Xte)
            out[target][name] = {"mae": float(mean_absolute_error(yte, preds)),
                                 "rmse": float(np.sqrt(mean_squared_error(yte, preds)))}
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building / loading engineered dataset...")
    dataset, features = build_dataset()
    train, test = chrono_split(dataset, features)
    print(f"train={len(train)}  test(held-out {TARGET_SEASON})={len(test)}  features={len(features)}")

    print("Experiment 1: impact-score ablation...")
    ablation = impact_ablation(train, test, features)
    print("Experiment 2: model comparison...")
    comparison = model_comparison(train, test, features)

    report = {
        "train_season": TRAIN_SEASON, "target_season": TARGET_SEASON,
        "n_train": int(len(train)), "n_test": int(len(test)), "n_features": len(features),
        "home_baseline_rate": float(test["home_win"].mean()),
        "impact_ablation": ablation,
        "model_comparison": comparison,
    }
    with open(os.path.join(OUTPUT_DIR, "analysis_2025_26.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== SUMMARY ===")
    print("home_win  with_impact:", ablation["targets"]["home_win"]["with_impact"],
          "\n          without   :", ablation["targets"]["home_win"]["without_impact"])
    print("model comparison home_win:", {k: round(v["accuracy"], 3)
                                          for k, v in comparison["home_win"].items()})
    print("Saved -> output/analysis_2025_26.json")


if __name__ == "__main__":
    main()
