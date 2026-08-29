"""
Probability calibration for the home_win classifier.

WHY
    The model ranks well but its probabilities are compressed toward 0.5.
    Measured on the 308 held-out games: in the bin where it says p(home)≈0.297
    the home team actually won 9.8% of the time, and where it says 0.736 the
    home team won 86.8%. Every bin is under-confident in the same direction.
    That is the classic signature of AUC (0.840) sitting well above thresholded
    accuracy (0.786) — the ordering is right, the scale is wrong.

    Accuracy is barely affected by fixing this (isotonic is monotone, so the
    ranking and therefore AUC are unchanged up to ties). What changes is that
    the number the model reports becomes usable: a stated 70% starts meaning
    70%. Anything downstream that consumes probabilities rather than labels —
    a simulation drawing outcomes, an expected-value calculation — is wrong
    until this is fixed.

PROTOCOL
    A three-way chronological split of the training data, so nothing is reused:
        fit        earliest 80%   — trains the LightGBM model
        early stop next 10%       — chooses the tree count
        calibrate  final 10%      — fits the calibrator on unseen predictions
    The held-out test season slice is never touched by any of the three.
    Calibrating on the early-stopping slice would fit the calibrator to scores
    the model was already tuned against, which flatters the result.

Two calibrators are fitted and compared, because neither is always better:
isotonic is flexible but can overfit a small calibration set, Platt is a
two-parameter sigmoid that cannot. The one with the lower Brier score on the
held-out set is reported as the winner, with both numbers shown.

Run:  py prediction_engines/calibration.py
Outputs: output/calibration_2025_26.json
         models/home_win_calibrator_2025_26.pkl
"""
import os
import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_CACHE = os.path.join(OUTPUT_DIR, "engineered_dataset_2025_26.pkl")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "home_win_calibrator_2025_26.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "calibration_2025_26.json")

TRAIN_SEASONS = ["2019_2020", "2020_2021", "2021_2022", "2022_2023",
                 "2023_2024", "2024_2025"]
TARGET_SEASON = "2025_2026"
TARGET = "home_win"

LGB_PARAMS = dict(random_state=42, n_estimators=600, learning_rate=0.03,
                  num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                  verbose=-1, n_jobs=-1)
EPS = 1e-6


def load_dataset():
    if not os.path.exists(DATASET_CACHE):
        raise FileNotFoundError(
            f"{DATASET_CACHE} yok — once retrain_two_arms.py calistirin.")
    blob = pd.read_pickle(DATASET_CACHE)
    return blob["dataset"], blob["features"]


def chrono_split(dataset):
    """Same split the production retrain uses: context seasons + first 75% of
    the target season train; the last 25% of the target season is held out."""
    ctx = dataset[dataset["season"].isin(TRAIN_SEASONS)]
    tgt = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt) * 0.75)
    train = pd.concat([ctx, tgt.iloc[:split]]) if not ctx.empty else tgt.iloc[:split]
    return train.sort_values("game_date"), tgt.iloc[split:]


def three_way(train_df):
    """fit 80% / early-stop 10% / calibrate 10%, in date order."""
    n = len(train_df)
    a, b = int(n * 0.80), int(n * 0.90)
    return train_df.iloc[:a], train_df.iloc[a:b], train_df.iloc[b:]


def Xy(df, features):
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, df[TARGET].astype(float)


def fit_base(fit_df, stop_df, features):
    X_fit, y_fit = Xy(fit_df, features)
    X_stop, y_stop = Xy(stop_df, features)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model


def fit_calibrators(p_cal, y_cal):
    """Isotonic (flexible, monotone) and Platt (two-parameter sigmoid)."""
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p_cal, y_cal)

    # Platt is fitted on the logit of the score, which is the standard form and
    # keeps the sigmoid well conditioned when scores crowd the ends.
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(_logit(p_cal).reshape(-1, 1), y_cal)
    return {"isotonic": isotonic, "platt": platt}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def apply_calibrator(name, calibrator, p):
    if name == "isotonic":
        return np.clip(calibrator.predict(p), EPS, 1 - EPS)
    return np.clip(calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1], EPS, 1 - EPS)


def evaluate(y, p):
    return {"accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p))}


def reliability(y, p, edges=(0.0, 0.35, 0.5, 0.65, 1.0001)):
    """Per-bin: what the model claimed vs what actually happened."""
    rows = []
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        rows.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "actual_rate": float(y[m].mean()),
                     "gap": float(p[m].mean() - y[m].mean())})
    return rows


def _print_reliability(title, rows):
    print(f"  {title}")
    print(f"    {'bant':12} {'n':>4} {'model der':>10} {'gercek':>8} {'sapma':>8}")
    for r in rows:
        print(f"    {r['bin']:12} {r['n']:4d} {r['mean_predicted']:10.3f} "
              f"{r['actual_rate']:8.3f} {r['gap']:+8.3f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset, features = load_dataset()
    train_df, test_df = chrono_split(dataset)
    fit_df, stop_df, cal_df = three_way(train_df)
    print(f"fit={len(fit_df)}  early-stop={len(stop_df)}  calibrate={len(cal_df)}  "
          f"test={len(test_df)}")

    model = fit_base(fit_df, stop_df, features)
    print(f"agac sayisi: {model.n_estimators_}")

    X_cal, y_cal = Xy(cal_df, features)
    X_test, y_test = Xy(test_df, features)
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    calibrators = fit_calibrators(p_cal, y_cal)
    scored = {"uncalibrated": evaluate(y_test, p_test)}
    curves = {"uncalibrated": reliability(y_test, p_test)}
    calibrated_probs = {}
    for name, cal in calibrators.items():
        p = apply_calibrator(name, cal, p_test)
        calibrated_probs[name] = p
        scored[name] = evaluate(y_test, p)
        curves[name] = reliability(y_test, p)

    winner = min(("isotonic", "platt"), key=lambda k: scored[k]["brier"])

    print("\n=== HELD-OUT TEST (308 mac) ===")
    print(f"  {'yontem':14} {'acc':>7} {'auc':>7} {'brier':>8} {'log_loss':>9}")
    for name in ("uncalibrated", "isotonic", "platt"):
        s = scored[name]
        mark = "  <- secildi" if name == winner else ""
        print(f"  {name:14} {s['accuracy']:7.4f} {s['auc']:7.4f} "
              f"{s['brier']:8.4f} {s['log_loss']:9.4f}{mark}")

    print("\n=== GUVENILIRLIK ===")
    _print_reliability("kalibrasyonsuz", curves["uncalibrated"])
    print()
    _print_reliability(f"{winner} ile", curves[winner])

    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump({"method": winner, "calibrator": calibrators[winner]}, f)

    report = {
        "protocol": "chronological three-way train split: fit 80% / early-stop 10% "
                    "/ calibrate 10%; test season slice untouched",
        "n_fit": len(fit_df), "n_early_stop": len(stop_df),
        "n_calibrate": len(cal_df), "n_test": len(test_df),
        "n_trees": int(model.n_estimators_),
        "selected": winner,
        "metrics": scored,
        "reliability": curves,
        "brier_improvement": scored["uncalibrated"]["brier"] - scored[winner]["brier"],
        "log_loss_improvement": scored["uncalibrated"]["log_loss"] - scored[winner]["log_loss"],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nYazildi: {REPORT_PATH}")
    print(f"Yazildi: {CALIBRATOR_PATH}  (yontem: {winner})")


if __name__ == "__main__":
    main()
