"""
Forecasts for the games not played yet, from the production models.

Reads output/engineered_dataset_asof.pkl (build_dataset_asof.py) and scores
its unplayed rows with the models retrain_production.py saved: the blended
home-win probability the dashboard shows, the point-difference model's
margin and the total-score model's total. An unplayed game's team sheets are
carried forward from each team's last game until the injury report is
archived, so a player ruled out since then is still counted as available.

Each forecast is stamped with when it was made and each game's tip-off, both
in UTC. A forecast made after tip-off still uses nothing from the game, but
it is not a pre-game forecast; `before_tipoff` keeps the two apart for any
later scoring.

It also scores the season's games played so far, as a first look at the
models on 2026-27. Those rows are built like the training rows - each team's
sheet read from that game's box score - so that is the backtest's regime, not
a forecast made before tip-off.

The forecasts are information about the games, not betting advice.

Run:  py prediction_engines/forecast_upcoming.py
Output: output/forecasts/forecast_<first date>_<made, UTC>.csv and a printed table
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ASOF_PATH = os.path.join(PROJECT_ROOT, "output", "engineered_dataset_asof.pkl")
FORECAST_DIR = os.path.join(PROJECT_ROOT, "output", "forecasts")
SHOWN = ["game_date", "away_team", "home_team", "home_win_prob", "margin", "total", "before_tipoff"]


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models():
    """(blend, margin model, total model) as retrain_production.py saved them."""
    ensemble = _load_sibling("ensemble_model")
    blend = ensemble.load_blend(MODEL_DIR)
    if blend is None:
        raise FileNotFoundError("models/blend_2025_26.pkl yok - once retrain_production.py")
    return blend, ensemble.load_model("point_diff", MODEL_DIR), ensemble.load_model("total_score", MODEL_DIR)


def tipoffs():
    """{game_id: tip-off time, UTC} from the newest schedule file."""
    _season, schedule = _load_sibling("build_dataset_asof").current_schedule()
    return dict(zip(schedule["game_id"].str.zfill(10), pd.to_datetime(schedule["tipoff_utc"], utc=True)))


def score(rows, features, models):
    """Probability, margin and total for each row, home team's side."""
    blend, margin_model, total_model = models
    X = rows[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    out = rows[["game_id", "game_date", "season", "home_team", "away_team"]].copy()
    out["home_win_prob"] = blend.predict_proba(X)[:, 1]
    out["margin"] = margin_model.predict(X)
    out["total"] = total_model.predict(X)
    return out


def season_so_far(dataset, features, models):
    """The current season's played games, scored, with their results; empty when there are none."""
    season = max(dataset["season"])
    played = dataset[dataset["is_played"] & (dataset["season"] == season)]
    if played.empty:
        return played, season
    scored = score(played, features, models)
    scored["home_won"] = played["home_win"].astype(int).values
    scored["real_margin"] = played["point_diff"].values
    return scored, season


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=ASOF_PATH)
    args = parser.parse_args()
    blob = pd.read_pickle(args.in_path)
    dataset, features = blob["dataset"], blob["production_features"]
    models = load_models()

    upcoming = dataset[~dataset["is_played"]]
    if upcoming.empty:
        print("oynanmamis mac yok - build_dataset_asof.py fiksturde gelecek tarih bulamadi")
    else:
        forecasts = score(upcoming, features, models).sort_values(["game_date", "game_id"])
        now = pd.Timestamp.now(tz="UTC").floor("s")
        forecasts["generated_at"] = now.isoformat()
        forecasts["dataset_as_of"] = blob.get("as_of")
        forecasts["tipoff_utc"] = forecasts["game_id"].map(tipoffs())
        forecasts["before_tipoff"] = forecasts["tipoff_utc"] > now
        os.makedirs(FORECAST_DIR, exist_ok=True)
        first = pd.to_datetime(forecasts["game_date"]).min().date().isoformat()
        # One file per run: a later run must not overwrite the forecast made before tip-off.
        path = os.path.join(FORECAST_DIR, f"forecast_{first}_{now:%Y%m%dT%H%M%SZ}.csv")
        forecasts.to_csv(path, index=False)
        view = forecasts[SHOWN].assign(game_date=pd.to_datetime(forecasts["game_date"]).dt.date)
        with pd.option_context("display.width", 140):
            print(view.round({"home_win_prob": 3, "margin": 1, "total": 1}).to_string(index=False))
        print(f"Yazildi: {path}")

    scored, season = season_so_far(dataset, features, models)
    if len(scored):
        p, won = scored["home_win_prob"].to_numpy(), scored["home_won"].to_numpy()
        print(f"\n{season}, oynanmis {len(scored)} mac (kutu skorundan kadro, yani geri test kosullari): "
              f"isabet {np.mean((p > 0.5) == (won == 1)):.3f}, Brier {brier_score_loss(won, p):.3f}, "
              f"fark MAE {np.mean(np.abs(scored['margin'] - scored['real_margin'])):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
