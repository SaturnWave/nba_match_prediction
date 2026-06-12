"""
One-pass retrain after the BoxScoreMatchupsV3 download: rebuild the engineered
dataset WITH the 6 matchup features, refresh output/engineered_dataset_2025_26.pkl
(what app.py serves), retrain the five production models, and run a FAIR matchup
ablation.

Fair = both arms early-stop on a validation slice carved from the train tail
(last 10% chronologically), never on the held-out test set — the leak-free
protocol the improvement report calls for. The production train() keeps the old
protocol so its headline metrics stay comparable with metrics_2025_26.json.

Run:  py prediction_engines/retrain_with_matchups.py
Outputs: models/*_model_2025_26.pkl, output/engineered_dataset_2025_26.pkl,
         output/metrics_2025_26.json, output/matchup_ablation.json
"""
import os
import json
import importlib.util

import numpy as np
import pandas as pd
import lightgbm as lgb
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


def chrono_split(dataset):
    ctx = dataset[dataset["season"] == TRAIN_SEASON]
    tgt = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt) * 0.75)
    train = pd.concat([ctx, tgt.iloc[:split]]) if not ctx.empty else tgt.iloc[:split]
    test = tgt.iloc[split:]
    return train, test


def Xy(df, features, target):
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, df[target].astype(float)


def fit_fair(kind, train_df, test_df, features, target):
    """Leak-free early stopping: validation = chronological tail 10% of train."""
    tr = train_df.sort_values("game_date")
    cut = int(len(tr) * 0.9)
    fit_df, val_df = tr.iloc[:cut], tr.iloc[cut:]
    Xf, yf = Xy(fit_df, features, target)
    Xv, yv = Xy(val_df, features, target)
    Xte, yte = Xy(test_df, features, target)
    params = dict(random_state=42, n_estimators=600, learning_rate=0.03,
                  num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    M = lgb.LGBMClassifier(**params) if kind == "clf" else lgb.LGBMRegressor(**params)
    M.fit(Xf, yf, eval_set=[(Xv, yv)],
          eval_metric="auc" if kind == "clf" else "mae",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    if kind == "clf":
        preds = M.predict(Xte)
        probs = M.predict_proba(Xte)[:, 1]
        return M, {"accuracy": float(accuracy_score(yte, preds)),
                   "auc": float(roc_auc_score(yte, probs))}, (preds == yte.values).astype(int)
    preds = M.predict(Xte)
    return M, {"mae": float(mean_absolute_error(yte, preds)),
               "rmse": float(np.sqrt(mean_squared_error(yte, preds)))}, None


def matchup_ablation(dataset, features):
    matchup_feats = [f for f in features if "matchup_" in f]
    without = [f for f in features if f not in matchup_feats]
    train, test = chrono_split(dataset)
    res = {"matchup_features": matchup_feats,
           "n_features_with": len(features), "n_features_without": len(without),
           "protocol": "leak-free early stopping (val = chronological tail 10% of train)",
           "targets": {}, "home_win_by_segment": {}}

    correct = {}
    for target in ["home_win", "point_diff", "total_score"]:
        kind = "clf" if target == "home_win" else "reg"
        out = {}
        for tag, feats in (("with_matchup", features), ("without_matchup", without)):
            M, metrics, corr = fit_fair(kind, train, test, feats, target)
            out[tag] = metrics
            if target == "home_win":
                correct[tag] = corr
                if tag == "with_matchup":
                    imp = (pd.DataFrame({"feature": feats, "importance": M.feature_importances_})
                           .sort_values("importance", ascending=False).reset_index(drop=True))
                    ranks = {f: int(imp.index[imp["feature"] == f][0]) + 1 for f in matchup_feats}
                    out["matchup_feature_importance_rank"] = ranks
        if target == "home_win":
            out["delta_accuracy"] = out["with_matchup"]["accuracy"] - out["without_matchup"]["accuracy"]
            out["delta_auc"] = out["with_matchup"]["auc"] - out["without_matchup"]["auc"]
        else:
            out["delta_mae"] = out["with_matchup"]["mae"] - out["without_matchup"]["mae"]
        res["targets"][target] = out

    t = test.copy().reset_index(drop=True)
    t["with_correct"] = correct["with_matchup"]
    t["without_correct"] = correct["without_matchup"]
    t["abs_pd"] = t["point_diff"].abs()
    segments = {
        "close_games (|pd|<=6)": t["abs_pd"] <= 6,
        "moderate (6<|pd|<=15)": (t["abs_pd"] > 6) & (t["abs_pd"] <= 15),
        "blowouts (|pd|>15)": t["abs_pd"] > 15,
    }
    if "diff_matchup_edge" in t.columns:
        gap = t["diff_matchup_edge"].abs()
        med = gap.median()
        segments["large_matchup_edge_gap"] = gap > med
        segments["small_matchup_edge_gap"] = gap <= med
    for name, mask in segments.items():
        sub = t[mask]
        if len(sub) == 0:
            continue
        res["home_win_by_segment"][name] = {
            "n": int(len(sub)),
            "acc_with_matchup": float(sub["with_correct"].mean()),
            "acc_without_matchup": float(sub["without_correct"].mean()),
            "delta": float(sub["with_correct"].mean() - sub["without_correct"].mean())}
    return res


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_files = 0
    for season in (TRAIN_SEASON, TARGET_SEASON):
        sdir = os.path.join(PROJECT_ROOT, "nba_data", season)
        if os.path.isdir(sdir):
            n_files += sum(1 for g in os.listdir(sdir)
                           if os.path.exists(os.path.join(sdir, g, "box_scores",
                                                          f"{g}box_score_matchups.csv")))
    print(f"Matchup CSVs on disk (both seasons): {n_files}")

    print("\n[1/4] Rebuilding engineered dataset with matchup features...")
    P = pp.NBAPredictor()
    P.load_and_prepare(seasons=[TRAIN_SEASON, TARGET_SEASON])
    features = P._select_features()
    matchup_feats = [f for f in features if "matchup_" in f]
    print(f"  {len(features)} features ({len(matchup_feats)} matchup: {matchup_feats})")
    if not matchup_feats:
        raise RuntimeError("No matchup features in dataset — integration failed, aborting "
                           "(old pickle/models left untouched).")

    print("\n[2/4] Saving dataset pickle for app.py...")
    pd.to_pickle({"dataset": P.dataset, "features": features}, DATASET_CACHE)

    print("\n[3/4] Retraining production models (original protocol, comparable metrics)...")
    P.train(train_season=TRAIN_SEASON, target_season=TARGET_SEASON)

    print("\n[4/4] Fair matchup ablation (leak-free early stopping)...")
    ablation = matchup_ablation(P.dataset, features)
    with open(os.path.join(OUTPUT_DIR, "matchup_ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    print("\n=== MATCHUP ABLATION (fair protocol) ===")
    for tgt_name, out in ablation["targets"].items():
        if tgt_name == "home_win":
            print(f"  home_win  with={out['with_matchup']['accuracy']:.4f} "
                  f"without={out['without_matchup']['accuracy']:.4f} "
                  f"delta={out['delta_accuracy']:+.4f}  (auc delta {out['delta_auc']:+.4f})")
        else:
            print(f"  {tgt_name:12s} with={out['with_matchup']['mae']:.3f} "
                  f"without={out['without_matchup']['mae']:.3f} delta={out['delta_mae']:+.3f}")
    for seg, v in ablation["home_win_by_segment"].items():
        print(f"    {seg:28s} n={v['n']:4d} delta={v['delta']:+.4f}")

    print("\n--- Single-game sanity test (held-out) ---")
    print(json.dumps(P.test_single_game(), indent=2))
    print("\nDone. Restart app.py to serve the new models/dataset.")


if __name__ == "__main__":
    main()
