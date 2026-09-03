"""What accuracy is even reachable with the information this model has?

WHY ASK
    The target is 70% on a full-season basis and the model sits at 66.9%
    single / 67.9% ensembled. Six feature families have been screened and none
    moved it. The seed-ensemble dose-response then showed something sharper:
    going from one model to ten lifts accuracy by 0.0103 while AUC moves from
    0.7405 to 0.7409 - which is nothing. Ensembling is not finding more signal,
    it is only steadying the decision boundary.

    That distinction decides what to do next. If accuracy is limited by noise
    around the threshold, better averaging keeps paying. If it is limited by
    how much the features actually separate winners from losers, no amount of
    averaging, calibrating or threshold-tuning gets to 70% and the only honest
    move is to find genuinely new information.

WHAT IS COMPUTED
    For a perfectly calibrated forecaster, the best possible accuracy is
    E[max(p, 1-p)]: on a game you call at 0.62 you are right 62% of the time no
    matter what you do, so the ceiling is the average of that quantity over the
    games. Compare it against the accuracy actually achieved:

      * achieved well BELOW ceiling  -> the threshold or the calibration is
        leaving points on the table, and they are cheap to collect
      * achieved AT the ceiling      -> the model is already extracting
        everything its probabilities contain, and 70% needs new features

    The ceiling is only meaningful if the probabilities are calibrated, so
    calibration is measured alongside it rather than assumed - reliability by
    decile, plus the Brier decomposition into resolution and reliability.

    Also reported: the best fixed decision threshold, chosen on the training
    side of each fold and applied to the test month, so it is a real
    out-of-sample question and not a threshold fitted to the answer.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "information_ceiling.json")


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wf = _load_sibling("walk_forward")
cal = _load_sibling("calibration")


def collect_predictions(dataset, features, seeds, months, burn_in=10):
    """Out-of-sample probabilities for every game in every fold's test month.

    Averaged over seeds, which is the production configuration - the ceiling
    should be measured on the forecaster that actually ships, not on a single
    noisier one.
    """
    folds = wf.month_folds(dataset, months)
    rows = []
    t0 = time.time()
    for i, (month, train, test) in enumerate(folds, 1):
        raw, calibrated = [], []
        for seed in seeds:
            model, cal_df = wf.fit_classifier(train, features, seed)
            X_test, _ = cal.Xy(test, features)
            p = model.predict_proba(X_test)[:, 1]
            raw.append(p)

            X_cal, y_cal = cal.Xy(cal_df, features)
            p_cal = model.predict_proba(X_cal)[:, 1]
            cals = cal.fit_calibrators(p_cal, y_cal)
            pick = min(cals, key=lambda k: brier_score_loss(
                y_cal, cal.apply_calibrator(k, cals[k], p_cal)))
            calibrated.append(cal.apply_calibrator(pick, cals[pick], p))

        seasoned = np.minimum(test["home_gp"].values, test["away_gp"].values)
        rows.append(pd.DataFrame({
            "month": month,
            "game_id": test["game_id"].values,
            "seasoned": seasoned,
            "y": test["home_win"].astype(float).values,
            "p_raw": np.mean(raw, axis=0),
            "p_cal": np.mean(calibrated, axis=0),
        }))
        print(f"  [{i}/{len(folds)}] {month}  {len(test)} mac "
              f"({time.time() - t0:.0f} sn)", flush=True)
    out = pd.concat(rows, ignore_index=True)
    return out[out["seasoned"] >= burn_in].reset_index(drop=True)


def reliability(p, y, bins=10):
    """Predicted versus observed, by decile of the forecast."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() < 10:
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "predicted": float(p[m].mean()),
                    "observed": float(y[m].mean())})
    return out


def brier_decomposition(p, y, bins=10):
    """Brier = reliability - resolution + uncertainty (Murphy).

    Resolution is the part that measures information: how far the conditional
    outcome rates sit from the base rate. Reliability is the part a calibrator
    can fix. Splitting them says which of the two is costing accuracy.
    """
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    base = y.mean()
    rel = res = 0.0
    n = len(y)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        nk, pk, ok = m.sum(), p[m].mean(), y[m].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    return {"reliability": float(rel / n), "resolution": float(res / n),
            "uncertainty": float(base * (1 - base)),
            "brier": float(brier_score_loss(y, p))}


def threshold_scan(p, y):
    """Accuracy across thresholds, and the best one in hindsight.

    In hindsight deliberately: this is the upper bound on what threshold tuning
    could ever be worth. If the best hindsight threshold barely beats 0.5, then
    picking one honestly out-of-sample cannot be worth chasing.
    """
    grid = np.arange(0.35, 0.66, 0.01)
    accs = [(float(t), float(accuracy_score(y, (p > t).astype(int)))) for t in grid]
    best = max(accs, key=lambda kv: kv[1])
    return {"at_half": float(accuracy_score(y, (p > 0.5).astype(int))),
            "best_threshold": best[0], "best_accuracy": best[1],
            "curve": accs}


def analyse(preds):
    out = {"n_games": int(len(preds)), "home_win_rate": float(preds["y"].mean())}
    for tag in ("p_raw", "p_cal"):
        p, y = preds[tag].values, preds["y"].values
        ceiling = float(np.maximum(p, 1 - p).mean())
        achieved = float(accuracy_score(y, (p > 0.5).astype(int)))
        out[tag] = {
            "accuracy": achieved,
            "auc": float(roc_auc_score(y, p)),
            "ceiling_if_calibrated": ceiling,
            "gap_to_ceiling": ceiling - achieved,
            "brier_parts": brier_decomposition(p, y),
            "threshold": threshold_scan(p, y),
            "reliability": reliability(p, y),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",
                        default=os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--out", default=REPORT_PATH)
    args = parser.parse_args()

    dataset, features, _ = wf.load_any_dataset(args.dataset)
    dataset = wf.add_team_game_index(dataset)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"{len(features)} feature, {len(seeds)} seed, burn-in {args.burn_in}",
          flush=True)

    preds = collect_predictions(dataset, features, seeds, args.months, args.burn_in)
    report = analyse(preds)
    report["seeds"] = seeds
    report["burn_in"] = args.burn_in

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"\n=== {report['n_games']:,} out-of-sample mac "
          f"(ev kazanma orani {report['home_win_rate']:.3f}) ===")
    for tag, label in (("p_raw", "ham"), ("p_cal", "kalibre")):
        r = report[tag]
        b = r["brier_parts"]
        print(f"\n{label}:")
        print(f"  isabet                 {r['accuracy']:.4f}")
        print(f"  AUC                    {r['auc']:.4f}")
        print(f"  kalibreyse tavan       {r['ceiling_if_calibrated']:.4f}")
        print(f"  tavana uzaklik         {r['gap_to_ceiling']:+.4f}")
        print(f"  brier {b['brier']:.4f} = guvenilirlik {b['reliability']:.4f} "
              f"- ayirt edicilik {b['resolution']:.4f} "
              f"+ belirsizlik {b['uncertainty']:.4f}")
        t = r["threshold"]
        print(f"  esik 0.50 -> {t['at_half']:.4f}, en iyi esik "
              f"{t['best_threshold']:.2f} -> {t['best_accuracy']:.4f} "
              f"(gecmise bakarak, ust sinir)")
    print(f"\nYazildi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
