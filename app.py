"""
NBA 2025-26 prediction dashboard â€” served over Tailscale, built for a phone.

WHAT CHANGED AND WHY
    The old dashboard showed one number per target from five independently
    fitted models. Everything measured since then says that presentation was
    misleading in three specific ways, and this rebuild fixes each:

      * the raw win probability is badly scaled. Where the classifier says 30%
        the home team actually wins ~10%, and where it says 74% they win ~87%.
        The page now shows the CALIBRATED probability, with the raw one kept
        beside it so the correction is visible rather than hidden.
      * a single point score is not a prediction. The point regressor's spread
        is a third of reality (sd 4.5 against 13.7), because a model trained on
        MAE correctly predicts the conditional mean. The simulator draws from a
        distribution instead, so the page can show an 80% interval that
        actually covers 80%.
      * the headline accuracy was 0.786 measured on the season's easiest five
        weeks. Walk-forward over 12 months puts it at 0.667 +/- 0.066 against a
        0.550 baseline. /api/meta reports the honest number.

    Opponent-adjusted ratings (Elo, Massey) are also surfaced, because they are
    the one signal shown to improve the model and they explain a prediction far
    better than a rolling average does.

SERVING
    Binds to this machine's Tailscale address by default, so the dashboard is
    reachable from other devices on the tailnet and from nothing else â€” not
    from the local Wi-Fi network, not from the internet. Uses waitress rather
    than Flask's development server, which is single-threaded and explicitly
    not meant to face a network.

Run:  py app.py                 (auto-detects the Tailscale IP)
      py app.py --host 0.0.0.0  (also expose on LAN â€” only if you mean it)
      py app.py --dev           (Flask dev server, localhost, for debugging)
"""
import os
import sys
import pickle
import argparse
import subprocess
import importlib.util

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_2025_26.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "home_win_calibrator_2025_26.pkl")
SIMULATOR_PATH = os.path.join(MODEL_DIR, "simulator_2025_26.pkl")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]
TARGET_SEASON = "2025_2026"
SIM_DRAWS = 4000

# Measured over 12 monthly walk-forward folds x 3 seeds, burn-in >= 10 games.
# Hardcoded rather than recomputed: the page must not imply these came from the
# games being browsed.
HONEST_METRICS = {
    "accuracy": 0.6673, "accuracy_std": 0.0659,
    "accuracy_calibrated": 0.6693,
    "accuracy_with_ratings_calibrated": 0.6814,
    "naive_baseline": 0.5499,
    "auc": 0.7340,
    "protocol": "12 aylik walk-forward x 3 seed, burn-in >= 10 mac",
    "single_split_accuracy": 0.7857,
    "single_split_note": "tek split rakami sezonun en kolay 5 haftasindan geliyor",
}

BADGE_IN_SAMPLE = {
    "key": "in-sample",
    "label": "Egitim verisinde â€” model bu sonucu gordu, tahmin iyimser",
}
BADGE_OUT_OF_SAMPLE = {
    "key": "out-of-sample",
    "label": "Held-out â€” gercek out-of-sample tahmin",
}

app = Flask(__name__)
_STATE = None


def _load_sibling(name):
    """prediction_engines/ holds scripts, not a package â€” import them by path."""
    path = os.path.join(PROJECT_ROOT, "prediction_engines", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
#  Startup loading
# ===========================================================================
def _build_player_games(dataset):
    """Per-player impact history from the v3 impact cache.

    Joins game_impact_cache_v4.pkl to the dataset's game metadata so every
    record carries (player, team, game_id, game_date, season, impact),
    sorted by player then date. Powers the roster-impact panel.
    """
    cache = {}
    if os.path.exists(IMPACT_CACHE):
        try:
            with open(IMPACT_CACHE, "rb") as f:
                cache = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            cache = {}
    meta = dataset[["game_id", "game_date", "season", "home_team", "away_team"]]
    records = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        imp = cache.get(gid)
        if not isinstance(imp, dict):
            continue
        for key, d in (imp.get("players") or {}).items():
            if not isinstance(d, dict):
                continue
            team = d.get("team")
            if team not in (home, away):
                continue
            # The cache keys on person id; the surname is display only, because
            # surnames collide (two Wigginses on opposing teams, LeBron and
            # Bronny on the same one) and would merge distinct players.
            person_id = d.get("person_id")
            records.append({"player": person_id if person_id is not None else str(key),
                            "name": d.get("name") or str(key),
                            "team": team, "game_id": gid,
                            "game_date": gdate, "season": season,
                            "impact": float(d.get("impact", 0.0))})
    pg = pd.DataFrame(records)
    if pg.empty:
        return pg
    return pg.sort_values(["player", "game_date"]).reset_index(drop=True)


def _load_optional(path, label):
    """Load a pickle that the dashboard degrades gracefully without."""
    if not os.path.exists(path):
        print(f"[startup] {label} yok ({os.path.basename(path)}) â€” o panel kapali")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        print(f"[startup] {label} okunamadi: {exc} â€” o panel kapali")
        return None


def _load_state():
    with open(DATASET_PATH, "rb") as f:
        bundle = pickle.load(f)
    dataset = bundle["dataset"].copy()
    features = bundle["features"]
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])

    # 75% chronological split of the target season, exactly as NBAPredictor.train
    # does it: split the season subset in its AS-SAVED row order. This must
    # happen before the display re-sort below â€” sort_values is not stable, so
    # sorting the full frame first permutes ties on the boundary date and would
    # hold out a different set than the trainer actually did.
    tgt_rows = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt_rows) * 0.75)
    test_game_ids = set(tgt_rows.iloc[split:]["game_id"])
    split_date = (str(tgt_rows.iloc[split]["game_date"].date())
                  if split < len(tgt_rows) else None)

    dataset = dataset.sort_values("game_date").reset_index(drop=True)
    dataset["month"] = dataset["game_date"].dt.strftime("%Y-%m")

    # Through ensemble_model rather than pickle.load: the production models are
    # SeedEnsemble instances, and a pickle records the class by module path. The
    # trainer runs as __main__, so a direct load here looks for
    # __main__.SeedEnsemble in app.py and does not find it.
    ensemble_module = _load_sibling("ensemble_model")
    models = {tgt: ensemble_module.load_model(tgt, MODEL_DIR) for tgt in TARGETS}

    calibrator = _load_optional(CALIBRATOR_PATH, "kalibrator")
    simulator = _load_optional(SIMULATOR_PATH, "simulator")

    ratings_module = _load_sibling("team_ratings")
    rated = ratings_module.add_rating_features(dataset)
    rating_cols = [c for c in ratings_module.RATING_FEATURE_COLS if c in rated.columns]
    ratings_by_game = rated.set_index("game_id")[rating_cols] if rating_cols else None

    player_games = _build_player_games(dataset)
    print(f"[startup] {dataset.shape[0]} mac x {dataset.shape[1]} sutun, "
          f"{len(features)} feature, {len(models)} model, "
          f"{len(player_games)} oyuncu-mac kaydi, "
          f"held-out {len(test_game_ids)} mac ({split_date} sonrasi)")
    return {"dataset": dataset, "features": features, "models": models,
            "test_game_ids": test_game_ids, "split_date": split_date,
            "player_games": player_games, "calibrator": calibrator,
            "simulator": simulator, "ratings": ratings_by_game,
            "simulation_module": _load_sibling("simulation") if simulator else None,
            "calibration_module": _load_sibling("calibration") if calibrator else None}


def get_state():
    """Create-once accessor â€” nothing is reloaded per request."""
    global _STATE
    if _STATE is None:
        _STATE = _load_state()
    return _STATE


# ===========================================================================
#  Prediction helpers
# ===========================================================================
def _calibrated(state, raw_prob):
    """Raw classifier probability passed through the fitted calibrator."""
    blob, module = state["calibrator"], state["calibration_module"]
    if not blob or module is None:
        return None
    value = module.apply_calibrator(blob["method"], blob["calibrator"],
                                    np.array([raw_prob]))
    return {"prob": round(float(value[0]), 3), "method": blob["method"]}


def _simulate_game(state, row):
    """Draw the game SIM_DRAWS times and read every quantity off the draws.

    Returns None when the simulator has not been trained yet.
    """
    blob, module = state["simulator"], state["simulation_module"]
    if not blob or module is None:
        return None
    features = blob["features"]
    X = row[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    lam_home = np.clip(blob["models"]["home"].predict(X), 1.0, None)
    lam_away = np.clip(blob["models"]["away"].predict(X), 1.0, None)
    home, away = module.simulate(lam_home, lam_away, blob["dispersion"],
                                 blob["rho"], n_sims=SIM_DRAWS)
    draws = module.summarise_draws(home, away)
    return {
        "home_win_prob": round(float(draws["p_home_win"][0]), 3),
        "home_score": round(float(draws["home_score"][0]), 1),
        "away_score": round(float(draws["away_score"][0]), 1),
        "point_diff": round(float(draws["point_diff"][0]), 1),
        "total_score": round(float(draws["total_score"][0]), 1),
        "home_score_range": [int(draws["home_score_lo"][0]), int(draws["home_score_hi"][0])],
        "margin_range": [int(draws["margin_lo"][0]), int(draws["margin_hi"][0])],
        "total_range": [int(draws["total_lo"][0]), int(draws["total_hi"][0])],
        "draws": SIM_DRAWS,
        "interval": "%80",
    }


def _ratings_for_game(state, game_id):
    """Opponent-adjusted ratings as they stood before this game."""
    table = state["ratings"]
    if table is None or game_id not in table.index:
        return None
    row = table.loc[game_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def val(col):
        v = row.get(col)
        return None if v is None or pd.isna(v) else round(float(v), 1)

    return {
        "elo": {"home": val("home_rating_elo"), "away": val("away_rating_elo"),
                "diff": val("diff_rating_elo")},
        "massey": {"home": val("home_rating_massey"), "away": val("away_rating_massey"),
                   "diff": val("diff_rating_massey")},
        "sos": {"home": val("home_rating_sos"), "away": val("away_rating_sos")},
    }


# ===========================================================================
#  Display helpers
# ===========================================================================
def _badge(state, game_id, season):
    if season == TARGET_SEASON and game_id in state["test_game_ids"]:
        return BADGE_OUT_OF_SAMPLE
    return BADGE_IN_SAMPLE


def _game_summary(state, g):
    return {
        "game_id": g["game_id"],
        "date": str(g["game_date"].date()),
        "season": g["season"],
        "matchup": f"{g['away_team']} @ {g['home_team']}",
        "home_team": g["home_team"],
        "away_team": g["away_team"],
        "home_score": float(g["home_score"]),
        "away_score": float(g["away_score"]),
        "badge": _badge(state, g["game_id"], g["season"]),
    }


def _roster_impact_for_team(state, team, as_of_date, top_n=6):
    """Top players by trailing-10-game mean impact strictly before a date.

    A player belongs to the team's roster if their most recent appearance
    before `as_of_date` was for that team (handles trades); trailing means are
    taken over their own history regardless of jersey, mirroring
    get_current_roster_impact's strictly-before-the-game windows.
    """
    pg = state["player_games"]
    if pg is None or pg.empty:
        return []
    hist = pg[pg["game_date"] < as_of_date]
    if hist.empty:
        return []
    latest = hist.groupby("player", sort=False).tail(1)
    roster = set(latest.loc[latest["team"] == team, "player"])
    if not roster:
        return []
    out = []
    for player, grp in hist[hist["player"].isin(roster)].groupby("player", sort=False):
        imp = grp["impact"]
        # `player` is the person id; the readable surname rides along separately.
        display = grp["name"].iloc[-1] if "name" in grp.columns else str(player)
        out.append({"name": display,
                    "l10_mean": round(float(imp.tail(10).mean()), 2),
                    "l3_mean": round(float(imp.tail(3).mean()), 2),
                    "n_games": int(len(imp))})
    out.sort(key=lambda r: r["l10_mean"], reverse=True)
    return out[:top_n]


def _matchup_data(season, game_id, top_n=8):
    """Top defender-vs-scorer assignments, or None when the csv is absent."""
    path = os.path.join(BASE_DATA_DIR, season, game_id, "box_scores",
                        f"{game_id}box_score_matchups.csv")
    if not (os.path.exists(path) and os.path.getsize(path) > 64):
        return None
    try:
        m = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    needed = {"firstNameOff", "familyNameOff", "firstNameDef", "familyNameDef",
              "partialPossessions", "playerPoints",
              "matchupFieldGoalsMade", "matchupFieldGoalsAttempted"}
    if m.empty or not needed.issubset(m.columns):
        return None
    # A partially written file is coerced, not crashed on. NaN is truthy, so
    # `pd.to_numeric(x) or 0` would NOT guard int(NaN).
    for col in ("partialPossessions", "playerPoints",
                "matchupFieldGoalsMade", "matchupFieldGoalsAttempted"):
        m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0)
    rows = []
    for _, r in m.sort_values("partialPossessions", ascending=False).head(top_n).iterrows():

        def name(first, family):
            parts = [r[c] for c in (first, family) if isinstance(r[c], str)]
            return " ".join(p.strip() for p in parts).strip()

        rows.append({
            "offName": name("firstNameOff", "familyNameOff"),
            "defName": name("firstNameDef", "familyNameDef"),
            "partialPossessions": round(float(r["partialPossessions"]), 1),
            "playerPoints": int(r["playerPoints"]),
            "matchupFieldGoalsMade": int(r["matchupFieldGoalsMade"]),
            "matchupFieldGoalsAttempted": int(r["matchupFieldGoalsAttempted"]),
        })
    return rows


# ===========================================================================
#  Routes
# ===========================================================================
@app.route("/")
def index():
    get_state()
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    state = get_state()
    df = state["dataset"]
    return jsonify({
        "seasons": sorted(df["season"].unique().tolist()),
        "teams": sorted(set(df["home_team"]) | set(df["away_team"])),
        "months": sorted(df["month"].unique().tolist(), reverse=True),
        "split_date": state["split_date"],
        "n_held_out": len(state["test_game_ids"]),
        "n_games": int(len(df)),
        "metrics": HONEST_METRICS,
        "has_calibrator": state["calibrator"] is not None,
        "has_simulator": state["simulator"] is not None,
    })


@app.route("/api/games")
def api_games():
    state = get_state()
    df = state["dataset"]
    season = request.args.get("season", "").strip()
    team = request.args.get("team", "").strip().upper()
    month = request.args.get("month", "").strip()
    if season:
        df = df[df["season"] == season]
    if team:
        df = df[(df["home_team"] == team) | (df["away_team"] == team)]
    if month:
        df = df[df["month"] == month]
    df = df.sort_values(["game_date", "game_id"], ascending=False)
    # A phone should not be handed 8000 rows; the filters exist to narrow it.
    limit = min(int(request.args.get("limit", 300) or 300), 1000)
    games = [_game_summary(state, g) for _, g in df.head(limit).iterrows()]
    return jsonify({"count": int(len(df)), "shown": len(games), "games": games})


@app.route("/api/predict/<game_id>")
def api_predict(game_id):
    state = get_state()
    gid = str(game_id).zfill(10)
    row = state["dataset"][state["dataset"]["game_id"] == gid]
    if row.empty:
        return jsonify({"error": f"game_id {gid} datasette yok"}), 404
    g = row.iloc[0]

    X = row[state["features"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    predicted = {}
    raw_prob = None
    for name, model in state["models"].items():
        if name == "home_win":
            raw_prob = float(model.predict_proba(X)[:, 1][0])
            predicted["home_win_prob_raw"] = round(raw_prob, 3)
        else:
            predicted[name] = round(float(model.predict(X)[0]), 2)

    calibrated = _calibrated(state, raw_prob)
    shown_prob = calibrated["prob"] if calibrated else raw_prob
    predicted["home_win_prob"] = round(shown_prob, 3)
    predicted["calibrated"] = calibrated
    predicted["predicted_winner"] = g["home_team"] if shown_prob > 0.5 else g["away_team"]

    actual_winner = g["home_team"] if int(g["home_win"]) else g["away_team"]
    out = _game_summary(state, g)
    out["predicted"] = predicted
    out["simulation"] = _simulate_game(state, row)
    out["ratings"] = _ratings_for_game(state, gid)
    out["actual"] = {
        "home_score": float(g["home_score"]),
        "away_score": float(g["away_score"]),
        "point_diff": float(g["point_diff"]),
        "total_score": float(g["total_score"]),
        "home_win": int(g["home_win"]),
        "winner": actual_winner,
        "correct_winner": bool(predicted["predicted_winner"] == actual_winner),
    }
    out["roster_impact"] = {
        "home": _roster_impact_for_team(state, g["home_team"], g["game_date"]),
        "away": _roster_impact_for_team(state, g["away_team"], g["game_date"]),
    }
    out["matchup_data"] = _matchup_data(g["season"], gid)
    return jsonify(out)


# ===========================================================================
#  Serving
# ===========================================================================
def tailscale_ip():
    """This machine's tailnet address, or None when Tailscale isn't running."""
    for candidate in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            result = subprocess.run([candidate, "ip", "-4"], capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            first = result.stdout.strip().splitlines()
            if first:
                return first[0].strip()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None,
                        help="bind adresi (varsayilan: Tailscale IP)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dev", action="store_true",
                        help="Flask gelistirme sunucusu, sadece localhost")
    args = parser.parse_args()

    if args.dev:
        get_state()
        print(f"\n  gelistirme modu -> http://127.0.0.1:{args.port}\n")
        app.run(host="127.0.0.1", port=args.port, debug=False)
        return

    host = args.host or tailscale_ip()
    if host is None:
        print("Tailscale IP bulunamadi. Tailscale calisiyor mu? "
              "Ya da --host ile acikca belirtin.", file=sys.stderr)
        return 1

    get_state()
    from waitress import serve
    print(f"\n  dashboard -> http://{host}:{args.port}")
    if host == "0.0.0.0":
        print("  UYARI: 0.0.0.0 tum arayuzlere acar, yerel agdan da erisilir")
    else:
        print("  yalnizca tailnet uzerinden erisilebilir")
    print("  durdurmak icin Ctrl+C\n")
    serve(app, host=host, port=args.port, threads=8)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
