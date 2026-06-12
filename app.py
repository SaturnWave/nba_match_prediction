"""
NBA 2025-26 prediction dashboard (Flask).

Browse any game from the engineered dataset and see what the five trained
LightGBM models (home_win / point_diff / total_score / home_score /
away_score) predict for it, side-by-side with the actual result, plus:

  * an in-sample vs held-out badge — games before the 75% chronological
    split of 2025-26 (and all of 2024-25) were inside the training set, so
    their predictions are optimistic; the last 25% are true out-of-sample;
  * a forward-looking roster-impact panel: each team's top players by
    trailing-10-game mean play-by-play impact computed strictly BEFORE the
    game date (the get_current_roster_impact signal);
  * defender-vs-scorer matchup assignments when the BoxScoreMatchupsV3 csv
    has been downloaded for the game (graceful when it hasn't yet).

Startup only unpickles output/engineered_dataset_2025_26.pkl, the five
models/<target>_model_2025_26.pkl files and game_impact_cache_v3.pkl —
no retraining, no per-game file scans — so it loads in seconds.

Run:  py app.py        then open  http://127.0.0.1:5000
"""
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_PATH = os.path.join(PROJECT_ROOT, "output", "engineered_dataset_2025_26.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v3.pkl")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]
TARGET_SEASON = "2025_2026"

BADGE_IN_SAMPLE = {
    "key": "in-sample",
    "label": "In training data — model saw this outcome; treat the prediction as optimistic",
}
BADGE_OUT_OF_SAMPLE = {
    "key": "out-of-sample",
    "label": "Held-out — true out-of-sample prediction",
}

app = Flask(__name__)

# module-level create-once state (filled by get_state on first use)
_STATE = None


# ===========================================================================
#  Startup loading
# ===========================================================================
def _build_player_games(dataset):
    """Per-player impact history from the v3 impact cache.

    Joins game_impact_cache_v3.pkl to the dataset's game metadata so every
    record carries (player, team, game_id, game_date, season, impact),
    sorted by player then date. Powers the roster-impact panel.
    """
    cache = {}
    if os.path.exists(IMPACT_CACHE):
        try:
            with open(IMPACT_CACHE, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    meta = dataset[["game_id", "game_date", "season", "home_team", "away_team"]]
    records = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        imp = cache.get(gid)
        if not isinstance(imp, dict):
            continue
        for player, d in (imp.get("players") or {}).items():
            if isinstance(d, dict):
                team, impact = d.get("team"), d.get("impact", 0.0)
            else:
                team, impact = None, d
            if team not in (home, away):  # drops None / stray tricodes
                continue
            records.append({"player": player, "team": team, "game_id": gid,
                            "game_date": gdate, "season": season,
                            "impact": float(impact)})
    pg = pd.DataFrame(records)
    if pg.empty:
        return pg
    return pg.sort_values(["player", "game_date"]).reset_index(drop=True)


def _load_state():
    """Load dataset, models and player impact history exactly once."""
    with open(DATASET_PATH, "rb") as f:
        bundle = pickle.load(f)
    dataset = bundle["dataset"].copy()
    features = bundle["features"]
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])

    # 75% chronological split of the target season — exactly like
    # NBAPredictor.train(): sort the season subset of the dataset in its
    # AS-SAVED row order by game_date, split index int(n * 0.75). This must
    # happen before the display re-sort below: sort_values is not stable, so
    # sorting the full frame first permutes ties on the boundary date and
    # would put a different 4 of that date's 6 games in the held-out set
    # than the trainer actually held out.
    tgt_rows = dataset[dataset["season"] == TARGET_SEASON].sort_values("game_date")
    split = int(len(tgt_rows) * 0.75)
    test_game_ids = set(tgt_rows.iloc[split:]["game_id"])
    split_date = (str(tgt_rows.iloc[split]["game_date"].date())
                  if split < len(tgt_rows) else None)

    dataset = dataset.sort_values("game_date").reset_index(drop=True)
    dataset["month"] = dataset["game_date"].dt.strftime("%Y-%m")

    models = {}
    for tgt in TARGETS:
        with open(os.path.join(MODEL_DIR, f"{tgt}_model_2025_26.pkl"), "rb") as f:
            models[tgt] = pickle.load(f)

    player_games = _build_player_games(dataset)
    print(f"[startup] dataset {dataset.shape[0]} games x {dataset.shape[1]} cols, "
          f"{len(features)} features, {len(models)} models, "
          f"{len(player_games)} player-game impact records, "
          f"held-out games: {len(test_game_ids)} (from {split_date})")
    return {"dataset": dataset, "features": features, "models": models,
            "test_game_ids": test_game_ids, "split_date": split_date,
            "player_games": player_games}


def get_state():
    """Create-once accessor — nothing is reloaded per request."""
    global _STATE
    if _STATE is None:
        _STATE = _load_state()
    return _STATE


# ===========================================================================
#  Helpers
# ===========================================================================
def _badge(state, game_id, season):
    """In-sample vs held-out label for a game (matches the training split)."""
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
    before `as_of_date` was for that team (handles trades); trailing means
    are taken over their own history regardless of jersey, mirroring
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
        imp = grp["impact"]  # already date-sorted within player
        out.append({"name": player,
                    "l10_mean": round(float(imp.tail(10).mean()), 2),
                    "l3_mean": round(float(imp.tail(3).mean()), 2),
                    "n_games": int(len(imp))})
    out.sort(key=lambda r: r["l10_mean"], reverse=True)
    return out[:top_n]


def _matchup_data(season, game_id, top_n=8):
    """Top defender-vs-scorer assignments from the matchups box score.

    Returns None when the csv hasn't been downloaded yet (the background
    retrieval job is still filling these in) or can't be parsed.
    """
    path = os.path.join(BASE_DATA_DIR, season, game_id, "box_scores",
                        f"{game_id}box_score_matchups.csv")
    if not (os.path.exists(path) and os.path.getsize(path) > 64):
        return None
    try:
        m = pd.read_csv(path)
    except Exception:
        return None
    needed = {"firstNameOff", "familyNameOff", "firstNameDef", "familyNameDef",
              "partialPossessions", "playerPoints",
              "matchupFieldGoalsMade", "matchupFieldGoalsAttempted"}
    if m.empty or not needed.issubset(m.columns):
        return None
    # the background download may have written only part of the file — coerce
    # everything and treat unparsable cells as 0 / "" instead of crashing
    # (NaN is truthy, so `pd.to_numeric(x) or 0` does NOT guard int(NaN))
    for col in ("partialPossessions", "playerPoints",
                "matchupFieldGoalsMade", "matchupFieldGoalsAttempted"):
        m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0)
    top = m.sort_values("partialPossessions", ascending=False).head(top_n)
    rows = []
    for _, r in top.iterrows():

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
    """Dropdown values (seasons, teams, months) plus the split date."""
    state = get_state()
    df = state["dataset"]
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return jsonify({
        "seasons": sorted(df["season"].unique().tolist()),
        "teams": teams,
        "months": sorted(df["month"].unique().tolist(), reverse=True),
        "split_date": state["split_date"],
        "n_held_out": len(state["test_game_ids"]),
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
    games = [_game_summary(state, g) for _, g in df.iterrows()]
    return jsonify({"count": len(games), "games": games})


@app.route("/api/predict/<game_id>")
def api_predict(game_id):
    state = get_state()
    gid = str(game_id).zfill(10)
    row = state["dataset"][state["dataset"]["game_id"] == gid]
    if row.empty:
        return jsonify({"error": f"game_id {gid} not found in dataset"}), 404
    g = row.iloc[0]

    X = row[state["features"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    predicted = {}
    for name, model in state["models"].items():
        if name == "home_win":
            p = float(model.predict_proba(X)[:, 1][0])
            predicted["home_win_prob"] = round(p, 3)
            predicted["predicted_winner"] = g["home_team"] if p > 0.5 else g["away_team"]
        else:
            predicted[name] = round(float(model.predict(X)[0]), 2)

    actual_winner = g["home_team"] if int(g["home_win"]) else g["away_team"]
    actual = {
        "home_score": float(g["home_score"]),
        "away_score": float(g["away_score"]),
        "point_diff": float(g["point_diff"]),
        "total_score": float(g["total_score"]),
        "home_win": int(g["home_win"]),
        "winner": actual_winner,
        "correct_winner": bool(predicted["predicted_winner"] == actual_winner),
    }

    out = _game_summary(state, g)
    out["predicted"] = predicted
    out["actual"] = actual
    out["roster_impact"] = {
        "home": _roster_impact_for_team(state, g["home_team"], g["game_date"]),
        "away": _roster_impact_for_team(state, g["away_team"], g["game_date"]),
    }
    out["matchup_data"] = _matchup_data(g["season"], gid)
    return jsonify(out)


if __name__ == "__main__":
    get_state()
    app.run(host="127.0.0.1", port=5000, debug=False)
