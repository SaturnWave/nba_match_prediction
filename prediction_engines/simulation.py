"""
Generative game simulator: one process produces all five outputs.

WHY REPLACE FIVE INDEPENDENT MODELS
    Today home_win, point_diff, total_score, home_score and away_score are five
    separately fitted models, so nothing forces them to agree. Measured on the
    308 held-out games:
        home_win and point_diff disagree on the winner in 9.1% of games (28)
        |(home_score - away_score) - point_diff|  averages 2.86, peaks at 12.0
        |(home_score + away_score) - total_score| averages 1.88, peaks at  8.1
    A simulator cannot contradict itself: every quantity is a statistic of the
    same simulated draws.

    The second problem is spread. A regressor trained on MAE predicts the
    conditional MEAN, and the conditional mean is correctly compressed - the
    predicted home score spans 102-126 (std 4.5) against an actual 86-157
    (std 13.7). Reporting that mean as "the predicted score" is the error, not
    the model. Drawing from a distribution restores the spread and, more
    usefully, gives intervals that can be checked for coverage.

HOW
    1. Two LightGBM regressors with objective="poisson" give each team's
       expected score lambda. Poisson is the right objective for a non-negative
       count with mean-variance coupling; plain L2 is mildly misspecified.
    2. NBA scores are OVERdispersed relative to Poisson: at lambda≈115 Poisson
       implies sd≈10.7 while the real residual sd is larger. So the marginals
       are negative binomial, with the dispersion r fitted by moments on a
       held-out slice rather than assumed.
    3. The two scores are correlated - pace lifts both teams at once - so the
       draws are coupled with a Gaussian copula whose correlation is estimated
       from the same slice. Drawing independently would give the right margins
       and the wrong totals.
    4. Everything else is read off the draws: P(home win), the margin
       distribution, the total, and any interval the caller wants.

PROTOCOL
    Chronological, matching the rest of the project: fit 80% / early-stop 10% /
    estimate dispersion and correlation on the final 10%. The test slice is
    never involved in fitting, stopping, or moment estimation.

Run:  py prediction_engines/simulation.py [--sims N]
Output: output/simulation_2025_26.json
        models/simulator_2025_26.pkl
"""
import os
import json
import pickle
import argparse
import importlib.util

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss, mean_absolute_error)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_PATH = os.path.join(OUTPUT_DIR, "simulation_2025_26.json")
SIMULATOR_PATH = os.path.join(MODEL_DIR, "simulator_2025_26.pkl")

POISSON_PARAMS = dict(objective="poisson", random_state=42, n_estimators=600,
                      learning_rate=0.03, num_leaves=31, subsample=0.8,
                      colsample_bytree=0.8, verbose=-1, n_jobs=-1)
DEFAULT_SIMS = 4000
SEED = 42
MIN_DISPERSION = 1e-3


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_sibling("calibration")


# ===========================================================================
#  Fitting
# ===========================================================================
def fit_rate_model(fit_df, stop_df, features, target):
    """LightGBM with a Poisson objective -> expected score for one side."""
    X_fit = fit_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_stop = stop_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    model = lgb.LGBMRegressor(**POISSON_PARAMS)
    model.fit(X_fit, fit_df[target].astype(float),
              eval_set=[(X_stop, stop_df[target].astype(float))], eval_metric="poisson",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return model


def estimate_dispersion(y, lam):
    """Negative-binomial r by moments: Var(Y) = lam + lam^2 / r.

    Returns None when the observed variance is at or below the Poisson floor,
    which means no extra dispersion is needed and Poisson marginals are used.
    """
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    excess = np.mean((y - lam) ** 2 - lam)
    if excess <= MIN_DISPERSION:
        return None
    return float(np.mean(lam ** 2) / excess)


def estimate_correlation(y_home, lam_home, y_away, lam_away):
    """Correlation of the two sides' Pearson residuals on the held-out slice."""
    r_home = (np.asarray(y_home, float) - lam_home) / np.sqrt(lam_home)
    r_away = (np.asarray(y_away, float) - lam_away) / np.sqrt(lam_away)
    return float(np.corrcoef(r_home, r_away)[0, 1])


# ===========================================================================
#  Simulating
# ===========================================================================
def _draw_counts(lam, r, uniforms):
    """Invert the count CDF at the given uniforms: NB if dispersed, else Poisson.

    lam is (n_games,) and uniforms is (n_games, n_sims); the result matches
    uniforms' shape.
    """
    lam2d = lam[:, None]
    if r is None:
        return stats.poisson.ppf(uniforms, mu=lam2d)
    p = r / (r + lam2d)
    return stats.nbinom.ppf(uniforms, n=r, p=p)


def simulate(lam_home, lam_away, dispersion, rho, n_sims=DEFAULT_SIMS, seed=SEED):
    """Draw n_sims correlated (home, away) score pairs per game.

    A Gaussian copula supplies the dependence: correlated normals -> uniforms
    -> the count marginals. This keeps each side's marginal exactly as fitted
    while giving the pair the correlation measured on real games.
    """
    rng = np.random.default_rng(seed)
    n = len(lam_home)
    rho = float(np.clip(rho, -0.99, 0.99))
    z_home = rng.standard_normal((n, n_sims))
    z_indep = rng.standard_normal((n, n_sims))
    z_away = rho * z_home + np.sqrt(1.0 - rho ** 2) * z_indep

    u_home = stats.norm.cdf(z_home)
    u_away = stats.norm.cdf(z_away)
    home = _draw_counts(np.asarray(lam_home, float), dispersion["home"], u_home)
    away = _draw_counts(np.asarray(lam_away, float), dispersion["away"], u_away)
    return home, away


def summarise_draws(home, away):
    """Every reported quantity is a statistic of the same draws, so they agree."""
    margin = home - away
    total = home + away
    return {
        "p_home_win": (margin > 0).mean(axis=1) + 0.5 * (margin == 0).mean(axis=1),
        "home_score": home.mean(axis=1),
        "away_score": away.mean(axis=1),
        "point_diff": margin.mean(axis=1),
        "total_score": total.mean(axis=1),
        "home_score_lo": np.percentile(home, 10, axis=1),
        "home_score_hi": np.percentile(home, 90, axis=1),
        "margin_lo": np.percentile(margin, 10, axis=1),
        "margin_hi": np.percentile(margin, 90, axis=1),
        "total_lo": np.percentile(total, 10, axis=1),
        "total_hi": np.percentile(total, 90, axis=1),
    }


# ===========================================================================
#  Scoring
# ===========================================================================
def coverage(actual, lo, hi):
    """Share of games whose real value fell inside the stated interval."""
    return float(((actual >= lo) & (actual <= hi)).mean())


def evaluate(test, draws, home, away):
    y_win = test["home_win"].astype(float).values
    p = np.clip(draws["p_home_win"], 1e-6, 1 - 1e-6)
    margin_pred = draws["point_diff"]

    contradictions = int((((p > 0.5) != (margin_pred > 0)) &
                          (np.abs(margin_pred) > 1e-9)).sum())
    return {
        "home_win": {"accuracy": float(accuracy_score(y_win, (p > 0.5).astype(int))),
                     "auc": float(roc_auc_score(y_win, p)),
                     "brier": float(brier_score_loss(y_win, p)),
                     "log_loss": float(log_loss(y_win, p, labels=[0.0, 1.0]))},
        "mae": {t: float(mean_absolute_error(test[t].astype(float), draws[t]))
                for t in ("point_diff", "total_score", "home_score", "away_score")},
        "coherence": {
            "win_vs_margin_contradictions": contradictions,
            "margin_identity_max_error": float(np.max(np.abs(
                (draws["home_score"] - draws["away_score"]) - draws["point_diff"]))),
            "total_identity_max_error": float(np.max(np.abs(
                (draws["home_score"] + draws["away_score"]) - draws["total_score"]))),
        },
        "spread": {
            "sim_home_score_sd": float(home.std(axis=1).mean()),
            "actual_home_score_sd": float(test["home_score"].astype(float).std()),
            "sim_margin_sd": float((home - away).std(axis=1).mean()),
            "actual_margin_sd": float(test["point_diff"].astype(float).std()),
        },
        "interval_coverage_80pct": {
            "home_score": coverage(test["home_score"].astype(float).values,
                                   draws["home_score_lo"], draws["home_score_hi"]),
            "point_diff": coverage(test["point_diff"].astype(float).values,
                                   draws["margin_lo"], draws["margin_hi"]),
            "total_score": coverage(test["total_score"].astype(float).values,
                                    draws["total_lo"], draws["total_hi"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset, features = cal.load_dataset()
    train, test = cal.chrono_split(dataset)
    fit_df, stop_df, moment_df = cal.three_way(train.sort_values("game_date"))
    print(f"fit={len(fit_df)} early-stop={len(stop_df)} moment={len(moment_df)} test={len(test)}")

    models = {side: fit_rate_model(fit_df, stop_df, features, f"{side}_score")
              for side in ("home", "away")}
    print(f"agac sayisi: home={models['home'].n_estimators_} away={models['away'].n_estimators_}")

    X_moment = moment_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    lam_m = {s: np.clip(models[s].predict(X_moment), 1.0, None) for s in models}
    dispersion = {s: estimate_dispersion(moment_df[f"{s}_score"].astype(float), lam_m[s])
                  for s in models}
    rho = estimate_correlation(moment_df["home_score"].astype(float), lam_m["home"],
                               moment_df["away_score"].astype(float), lam_m["away"])
    for side, r in dispersion.items():
        implied = "Poisson" if r is None else f"NB r={r:.1f}"
        print(f"  {side}: {implied}")
    print(f"  ev/deplasman artik korelasyonu rho={rho:+.3f}")

    X_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    lam_test = {s: np.clip(models[s].predict(X_test), 1.0, None) for s in models}
    home, away = simulate(lam_test["home"], lam_test["away"], dispersion, rho, args.sims)
    draws = summarise_draws(home, away)
    scored = evaluate(test, draws, home, away)

    print(f"\n=== SIMULASYON ({args.sims} cekilis/mac, {len(test)} mac) ===")
    hw = scored["home_win"]
    print(f"  home_win  acc={hw['accuracy']:.4f}  auc={hw['auc']:.4f}  "
          f"brier={hw['brier']:.4f}  log_loss={hw['log_loss']:.4f}")
    for t, v in scored["mae"].items():
        print(f"  {t:12} MAE={v:.3f}")
    c = scored["coherence"]
    print(f"\n  celiski (kazanan vs marj) : {c['win_vs_margin_contradictions']} mac "
          f"(bes bagimsiz modelde 28 idi)")
    print(f"  marj kimligi max hata     : {c['margin_identity_max_error']:.2e}")
    print(f"  toplam kimligi max hata   : {c['total_identity_max_error']:.2e}")
    s = scored["spread"]
    print(f"\n  ev skoru sd  : simulasyon {s['sim_home_score_sd']:.2f} vs gercek "
          f"{s['actual_home_score_sd']:.2f}")
    print(f"  marj sd      : simulasyon {s['sim_margin_sd']:.2f} vs gercek "
          f"{s['actual_margin_sd']:.2f}")
    print("\n  %80 araligin gercek kapsamasi (hedef 0.80):")
    for k, v in scored["interval_coverage_80pct"].items():
        print(f"    {k:12} {v:.3f}")

    with open(SIMULATOR_PATH, "wb") as f:
        pickle.dump({"models": models, "dispersion": dispersion, "rho": rho,
                     "features": features, "n_sims": args.sims}, f)
    with open(REPORT_PATH, "w") as f:
        json.dump({"protocol": "chronological fit 80% / early-stop 10% / moments 10%; "
                               "Poisson-objective rates, negative-binomial marginals, "
                               "Gaussian copula",
                   "n_sims": args.sims, "n_test": int(len(test)),
                   "dispersion_r": {k: v for k, v in dispersion.items()},
                   "residual_correlation": rho,
                   "metrics": scored}, f, indent=2)
    print(f"\nYazildi: {REPORT_PATH}")
    print(f"Yazildi: {SIMULATOR_PATH}")


if __name__ == "__main__":
    main()
