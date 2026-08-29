"""
Two-arm retrain after the data backfill.

  ARM A  current feature set (rolling form + season averages + H2H + streaks +
         roster impact + matchup features) — exactly what _select_features()
         returns today.
  ARM B  ARM A plus the 24 trailing defensive-box + player-tracking features
         from defensive_tracking_features.py.

Both arms are fitted under the SAME leak-free protocol (fit_fair: validation is
the chronological tail 10% of train, never the held-out test set), so the
difference between them is the feature set and nothing else. This is the only
comparison that answers "do defensive + tracking data earn their place".

Separately, phase 2 refreshes the PRODUCTION models with ARM A's feature set
under the original protocol (P.train, which early-stops on the test set). That
protocol is optimistic and known to be so — it is kept only because
metrics_2025_26.json and the improvement report's headline numbers were
produced with it, and changing it would silently break that comparison.

Why this runs now: the backfill closed two data gaps that fed the training set —
2024-25 gained its 146 missing traditional / 147 advanced box scores (those
games previously trained with their stats zeroed) and both 2024-25 and 2025-26
gained their missing defensive box scores + player tracking.

Run:  py prediction_engines/retrain_two_arms.py
Outputs:
  models/*_model_2025_26.pkl          ARM A, production (original protocol)
  models/*_model_2025_26_dt.pkl       ARM B, fair protocol
  output/engineered_dataset_2025_26.pkl   dataset app.py serves (ARM A features)
  output/metrics_2025_26.json         ARM A production metrics
  output/two_arm_comparison.json      the A/B result
"""
import os
import json
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "two_arm_comparison.json")


def _load_sibling(name):
    """Import a sibling module by path — these files are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# fit_fair / chrono_split / Xy are imported, never copied: ARM A and ARM B must
# run the identical protocol, and a second copy would drift from the original.
rw = _load_sibling("retrain_with_matchups")
dt = _load_sibling("defensive_tracking_features")
pp = rw.pp

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]


def build_arm_b(dataset):
    """ARM A dataset + the trailing defensive/tracking columns.

    NaN (a team's first game, or a game whose source CSVs are missing) is left
    for Xy() to fill with 0, matching how every other feature in this pipeline
    is handled. A 0 is a poor sentinel for a rate like fg_pct_allowed; it is
    used here only for consistency with the existing convention, and the
    coverage line printed below is what says whether it matters.
    """
    out = dt.add_defensive_tracking_features(dataset, data_dir=pp.BASE_DATA_DIR)
    if len(out) != len(dataset):
        # A duplicated (game_id, team) key on the right side would multiply rows
        # and silently desynchronise ARM B from ARM A. Fail loudly instead.
        raise RuntimeError(f"ARM B merge changed the row count: {len(dataset)} -> {len(out)}")
    present = [c for c in dt.DT_FEATURE_COLS if c in out.columns]
    return out, present


def evaluate_arm(tag, train_df, test_df, features, save_suffix=None):
    """Fit every target for one arm under the fair protocol; return its metrics."""
    metrics, correct, importances = {}, None, {}
    for target in TARGETS:
        kind = "clf" if target == "home_win" else "reg"
        model, scores, corr = rw.fit_fair(kind, train_df, test_df, features, target)
        metrics[target] = scores
        if target == "home_win":
            correct = corr
        if save_suffix and hasattr(model, "feature_importances_"):
            ranked = (pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                      .sort_values("importance", ascending=False).reset_index(drop=True))
            importances[target] = ranked
            path = os.path.join(MODEL_DIR, f"{target}_model_2025_26{save_suffix}.pkl")
            pd.to_pickle(model, path)
    print(f"  [{tag}] " + "  ".join(
        f"{t}=" + (f"{m['accuracy']:.4f}/{m['auc']:.4f}" if t == "home_win" else f"{m['mae']:.3f}")
        for t, m in metrics.items()))
    return metrics, correct, importances


def segment_ledger(test_df, correct_a, correct_b):
    """Where ARM B helps or hurts, by margin band — the same bands the
    improvement report uses, so the numbers stay comparable."""
    if not (len(test_df) == len(correct_a) == len(correct_b)):
        raise RuntimeError("Segment ledger needs both arms' test rows to line up: "
                           f"{len(test_df)} / {len(correct_a)} / {len(correct_b)}")
    t = test_df.copy().reset_index(drop=True)
    t["a_correct"], t["b_correct"] = correct_a, correct_b
    t["abs_pd"] = t["point_diff"].abs()
    bands = {
        "close (|pd|<=6)": t["abs_pd"] <= 6,
        "moderate (6<|pd|<=15)": (t["abs_pd"] > 6) & (t["abs_pd"] <= 15),
        "blowout (|pd|>15)": t["abs_pd"] > 15,
    }
    ledger = {}
    for name, mask in bands.items():
        sub = t[mask]
        if sub.empty:
            continue
        ledger[name] = {"n": int(len(sub)),
                        "acc_arm_a": float(sub["a_correct"].mean()),
                        "acc_arm_b": float(sub["b_correct"].mean()),
                        "delta": float(sub["b_correct"].mean() - sub["a_correct"].mean())}
    return ledger


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seasons = rw.TRAIN_SEASONS + [rw.TARGET_SEASON]

    print(f"[1/5] Building engineered dataset over {len(seasons)} seasons...")
    P = pp.NBAPredictor()
    P.load_and_prepare(seasons=seasons)
    features_a = P._select_features()
    matchup_feats = [f for f in features_a if "matchup_" in f]
    print(f"  ARM A: {len(features_a)} features ({len(matchup_feats)} matchup)")
    if not matchup_feats:
        raise RuntimeError("No matchup features in dataset — integration broken, aborting "
                           "so the existing pickle and models are left untouched.")

    print("\n[2/5] Saving dataset pickle for app.py (ARM A features)...")
    pd.to_pickle({"dataset": P.dataset, "features": features_a}, rw.DATASET_CACHE)

    print("\n[3/5] Refreshing PRODUCTION models — ARM A, original protocol...")
    P.train(train_season=rw.TRAIN_SEASONS, target_season=rw.TARGET_SEASON)
    production_metrics = dict(P.metrics)

    print("\n[4/5] Building ARM B (defensive + tracking features)...")
    dataset_b, dt_feats = build_arm_b(P.dataset)
    if not dt_feats:
        raise RuntimeError("No defensive/tracking features produced — check that the "
                           "backfill wrote box_score_defensive.csv and team_tracking.csv.")
    features_b = features_a + dt_feats
    coverage = {c: float(dataset_b[c].notna().mean()) for c in dt_feats}
    print(f"  ARM B: {len(features_b)} features (+{len(dt_feats)} defensive/tracking), "
          f"coverage {min(coverage.values()):.3f}-{max(coverage.values()):.3f}")

    print("\n[5/5] Fair A/B — identical protocol, feature set is the only difference...")
    train_a, test_a = rw.chrono_split(P.dataset)
    train_b, test_b = rw.chrono_split(dataset_b)
    metrics_a, correct_a, _ = evaluate_arm("ARM A", train_a, test_a, features_a)
    metrics_b, correct_b, imps_b = evaluate_arm("ARM B", train_b, test_b, features_b,
                                                save_suffix="_dt")

    dt_ranks = {}
    if "home_win" in imps_b:
        ranked = imps_b["home_win"]
        dt_ranks = {f: int(ranked.index[ranked["feature"] == f][0]) + 1 for f in dt_feats}

    deltas = {}
    for target in TARGETS:
        if target == "home_win":
            deltas[target] = {
                "delta_accuracy": metrics_b[target]["accuracy"] - metrics_a[target]["accuracy"],
                "delta_auc": metrics_b[target]["auc"] - metrics_a[target]["auc"]}
        else:
            deltas[target] = {"delta_mae": metrics_b[target]["mae"] - metrics_a[target]["mae"]}

    result = {
        "protocol": "leak-free early stopping (val = chronological tail 10% of train); "
                    "identical for both arms",
        "n_train": int(len(train_a)), "n_test": int(len(test_a)),
        "arm_a": {"n_features": len(features_a), "metrics": metrics_a},
        "arm_b": {"n_features": len(features_b), "metrics": metrics_b,
                  "dt_features": dt_feats, "dt_feature_coverage": coverage,
                  "dt_feature_importance_rank_home_win": dt_ranks},
        "delta_b_minus_a": deltas,
        "home_win_by_segment": segment_ledger(test_a, correct_a, correct_b),
        "production_metrics_arm_a_original_protocol": production_metrics,
    }
    with open(COMPARISON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== TWO-ARM COMPARISON (fair protocol) ===")
    for target in TARGETS:
        if target == "home_win":
            print(f"  home_win     A acc={metrics_a[target]['accuracy']:.4f} "
                  f"B acc={metrics_b[target]['accuracy']:.4f} "
                  f"delta={deltas[target]['delta_accuracy']:+.4f} "
                  f"| auc delta {deltas[target]['delta_auc']:+.4f}")
        else:
            print(f"  {target:12s} A mae={metrics_a[target]['mae']:.3f} "
                  f"B mae={metrics_b[target]['mae']:.3f} "
                  f"delta={deltas[target]['delta_mae']:+.3f}  (negative = ARM B better)")
    print("\n  home_win by margin band:")
    for name, seg in result["home_win_by_segment"].items():
        print(f"    {name:24s} n={seg['n']:4d}  A={seg['acc_arm_a']:.4f} "
              f"B={seg['acc_arm_b']:.4f}  delta={seg['delta']:+.4f}")
    if dt_ranks:
        top = sorted(dt_ranks.items(), key=lambda kv: kv[1])[:6]
        print("\n  best-ranked defensive/tracking features (home_win importance):")
        for name, rank in top:
            print(f"    #{rank:<4d} {name}")
    print(f"\nWrote {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
