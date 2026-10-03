"""
Score the production models against the closing line on the held-out games.

WHY THE MARKET IS THE YARDSTICK
    Nothing here is betting advice, and nothing will be. A closing line is the
    one forecast of an NBA game that is published before tip-off, priced by
    people who lose money when it is wrong, and archived afterwards. That makes
    it the only external benchmark this model can be measured against on the
    same games, in the same units: a win probability, a margin, a total.

WHAT IS COMPARED (DraftKings close, vig removed proportionally)
    winner        model P(home) vs market P(home): accuracy at 0.5, Brier
    margin        model point_diff vs minus the home spread: MAE against the
                  real margin, and the side of the line the model takes
    total         model total vs the posted total: MAE against the real total
    The against-the-spread tally counts how often the model's side of the
    closing spread covered. Breaking even at standard prices needs ~52.4%; the
    number is reported with a binomial standard error, because 308 games is a
    small sample and most things look like a trend at that size.

Inputs: odds_data/nba_2025_26_heldout_odds.csv (theScore + ESPN/DraftKings),
        models/*_2025_26.pkl, output/engineered_dataset_pregame.pkl
Run:    py prediction_engines/market_check.py
Output: output/market_check.json
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ODDS_PATH = os.path.join(PROJECT_ROOT, "odds_data", "nba_2025_26_heldout_odds.csv")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list_2025_26.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "market_check.json")
BREAK_EVEN = 0.524


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_inputs():
    odds = pd.read_csv(ODDS_PATH, dtype={"game_id": str})
    odds["game_id"] = odds["game_id"].str.zfill(10)
    blob = pd.read_pickle(DATASET_PATH)
    dataset = blob["dataset"]
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    features = list(blob["features"])
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, encoding="utf-8") as f:
            features = json.load(f)["features"]
    rows = dataset[dataset["game_id"].isin(odds["game_id"])].copy()
    return odds.merge(rows[["game_id"] + features], on="game_id", how="inner"), features


def model_predictions(frame, features):
    ensemble = _load_sibling("ensemble_model")
    X = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    blend = ensemble.load_blend(MODEL_DIR)
    clf = ensemble.load_model("home_win", MODEL_DIR)
    out = pd.DataFrame({"game_id": frame["game_id"]})
    out["p_home"] = (blend.predict_proba(X)[:, 1] if blend is not None
                     else clf.predict_proba(X)[:, 1])
    out["margin"] = ensemble.load_model("point_diff", MODEL_DIR).predict(X)
    out["total"] = ensemble.load_model("total_score", MODEL_DIR).predict(X)
    return out


def binomial_se(rate, n):
    return float(np.sqrt(rate * (1 - rate) / n)) if n else float("nan")


def compare(frame, pred):
    d = frame.merge(pred, on="game_id")
    d = d.dropna(subset=["dk_home_prob_novig_close", "dk_spread_home_close", "dk_total_close"])
    actual_margin = (d["home_pts"] - d["away_pts"]).astype(float)
    actual_total = (d["home_pts"] + d["away_pts"]).astype(float)
    home_won = d["home_won"].astype(int)
    market_p = d["dk_home_prob_novig_close"].astype(float)
    market_margin = -d["dk_spread_home_close"].astype(float)      # spread -7.5 => home by 7.5
    market_total = d["dk_total_close"].astype(float)

    model_side_home = d["margin"] > market_margin               # model likes home vs the line
    cover_margin = actual_margin - market_margin
    decided = cover_margin != 0                                  # pushes are not bets
    model_covered = np.where(model_side_home, cover_margin > 0, cover_margin < 0)[decided]
    ats_rate = float(model_covered.mean())
    gap = (d["margin"] - market_margin).abs()
    confident = decided & (gap > 3)
    conf_covered = np.where(model_side_home, cover_margin > 0, cover_margin < 0)[confident]

    return {
        "n_games": int(len(d)),
        "winner": {
            "model_accuracy": float(((d["p_home"] > 0.5).astype(int) == home_won).mean()),
            "market_accuracy": float(((market_p > 0.5).astype(int) == home_won).mean()),
            "model_brier": float(brier_score_loss(home_won, d["p_home"])),
            "market_brier": float(brier_score_loss(home_won, market_p)),
            "agreement_rate": float(((d["p_home"] > 0.5) == (market_p > 0.5)).mean()),
        },
        "margin": {
            "model_mae": float(mean_absolute_error(actual_margin, d["margin"])),
            "market_mae": float(mean_absolute_error(actual_margin, market_margin)),
            "mean_abs_model_minus_line": float(gap.mean()),
        },
        "total": {
            "model_mae": float(mean_absolute_error(actual_total, d["total"])),
            "market_mae": float(mean_absolute_error(actual_total, market_total)),
        },
        "against_the_spread": {
            "n": int(decided.sum()), "cover_rate": ats_rate,
            "se": binomial_se(ats_rate, int(decided.sum())), "break_even": BREAK_EVEN,
            "when_model_differs_by_3plus": {
                "n": int(confident.sum()),
                "cover_rate": float(conf_covered.mean()) if confident.sum() else None,
                "se": binomial_se(float(conf_covered.mean()), int(confident.sum()))
                if confident.sum() else None,
            },
        },
    }


def main():
    frame, features = load_inputs()
    pred = model_predictions(frame, features)
    report = compare(frame, pred)
    report["features"] = len(features)
    w, m, t, a = report["winner"], report["margin"], report["total"], report["against_the_spread"]
    print(f"{report['n_games']} mac, {len(features)} feature (mac oncesi kadro)")
    print(f"  kazanan   model {w['model_accuracy']:.3f}  piyasa {w['market_accuracy']:.3f}  "
          f"| Brier model {w['model_brier']:.4f}  piyasa {w['market_brier']:.4f}  "
          f"| ayni tarafi secme {w['agreement_rate']:.3f}")
    print(f"  marj MAE  model {m['model_mae']:.2f}  piyasa {m['market_mae']:.2f}  "
          f"| model-cizgi ort. fark {m['mean_abs_model_minus_line']:.2f}")
    print(f"  toplam MAE model {t['model_mae']:.2f}  piyasa {t['market_mae']:.2f}")
    c = a["when_model_differs_by_3plus"]
    print(f"  handikap  {a['cover_rate']:.3f} +/- {a['se']:.3f} ({a['n']} mac; basabas {BREAK_EVEN}); "
          f"fark > 3 sayi: {c['cover_rate'] if c['cover_rate'] is None else round(c['cover_rate'], 3)} "
          f"({c['n']} mac)")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Yazildi: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
