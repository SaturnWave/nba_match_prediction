"""
NBA 2025-26 prediction dashboard - served over Tailscale, built for a phone.

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
    reachable from other devices on the tailnet and from nothing else - not
    from the local Wi-Fi network, not from the internet. Uses waitress rather
    than Flask's development server, which is single-threaded and explicitly
    not meant to face a network.

Run:  py app.py                 (auto-detects the Tailscale IP)
      py app.py --host 0.0.0.0  (also expose on LAN - only if you mean it)
      py app.py --dev           (Flask dev server, localhost, for debugging)
"""
import os
import sys
import glob
import json
import pickle
import argparse
import datetime as dt
import subprocess
import importlib.util

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
# The SAME pickle retrain_production trains on. It has to be: the roster and
# matchup columns keep their names across the dataset builds but not their
# meaning - the older builds aggregated them over the players who appeared in
# the game (which leaks the result), this one over the pre-game team sheet.
# Serving predictions from features the models were never fitted on is a silent
# error; nothing raises, the numbers just mean less than they appear to.
#
# Two builds satisfy that: engineered_dataset_asof.pkl is the one served, and
# output/asof_parity.json records it equal feature-for-feature to the
# production build over all 10,749 games it shares with it - which is exactly
# the check the paragraph above demands. What it adds is the season in
# progress and the games not played yet (is_played False, with the five result
# columns NaN and all 196 trained features present).
#
# It is rebuilt daily by build_dataset_asof.py and is gitignored, so after a
# fresh clone it is absent. Then the production pickle is served instead and
# /api/meta says which one it is, rather than the page refusing to start.
ASOF_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_asof.pkl")
TRAINED_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
# Deliberately the OLD engine's cache. The models train on v5 (value pricing),
# but the rankings show production and value side by side and flag the gap as
# "empty stats"; production has to come from the engine that counted it.
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "home_win_calibrator_2025_26.pkl")
SIMULATOR_PATH = os.path.join(MODEL_DIR, "simulator_2025_26.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list_2025_26.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_2025_26.json")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]
# How far back the roster panel will look for a team's last game before it
# says it does not know. Longer than any in-season gap (the All-Star break is
# about nine days), shorter than an off-season.
ROSTER_STALE_DAYS = 30
# The last season the models were fitted through, NOT "the current season" -
# they stopped being the same thing the moment 2026-27 entered the dataset.
# Everything about held-out and in-sample is anchored here; what season the
# page is showing is read from the data instead. Season names sort
# chronologically because they are fixed-width ("2026_2027" > "2025_2026"),
# which is what lets a later season be recognised as one the model never saw.
TARGET_SEASON = "2025_2026"
# The simulated season the page shows beside the real ones, when it has been
# run on this machine (prediction_engines/season_sim.py forecast). It lives in
# sim_sandbox/, is only ever read here, and is never mixed into the real data:
# its own routes, its own section of the page, labelled as a simulation.
SIM_RUN = "forecast_2026_2027"
SIM_DRAWS = 4000

# What the page quotes as the model's accuracy. Read from the trainer's metrics
# file, whose honest_walk_forward block is copied from consistency_lab.py's
# monthly walk-forward - never recomputed from the games being browsed, so the
# page cannot imply the number came from them. The fallback is the last
# measurement made by hand, kept so the page still renders without the file.
# The naive baseline (always pick the side that wins more often that month) is
# a property of the games, not the model, so it does not change between runs.
NAIVE_BASELINE_21_MONTHS = 0.5554
FALLBACK_METRICS = {
    "accuracy": 0.6673, "accuracy_std": 0.0659,
    "naive_baseline": 0.5499, "auc": 0.7340,
    "protocol": "12 aylik walk-forward x 3 seed, burn-in >= 10 mac",
    "single_split_accuracy": 0.7857,
    "single_split_note": "tek split rakami sezonun en kolay 5 haftasindan geliyor",
}


def _honest_metrics():
    """The walk-forward numbers for the models actually on disk."""
    if not os.path.exists(METRICS_PATH):
        return FALLBACK_METRICS
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return FALLBACK_METRICS
    honest = payload.get("honest_walk_forward")
    if not honest or honest.get("accuracy_blend3") is None:
        return FALLBACK_METRICS
    return {
        "accuracy": honest["accuracy_blend3"],
        "accuracy_std": honest["month_sd"],
        "naive_baseline": NAIVE_BASELINE_21_MONTHS,
        "auc": honest["auc"],
        "protocol": (f"{honest['n_cells']} hucre aylik walk-forward (mac oncesi kadro, "
                     f"burn-in >= 10 mac); siniflandirici tek basina "
                     f"{honest['accuracy_classifier']:.3f}"),
        # Chained indexing here used to be unguarded: a metrics file with a
        # walk-forward block but no blend block (a retrain with the blend off)
        # raised KeyError inside /api/meta, which the page calls before
        # anything else, so the whole dashboard failed to open over one
        # optional number.
        "single_split_accuracy": (payload.get("metrics", {}).get("blend", {}).get("accuracy")
                                  or FALLBACK_METRICS["single_split_accuracy"]),
        "single_split_note": "tek split rakami sezonun son 5 haftasindan geliyor",
    }

# Four states, and nothing falls through to "the model trained on this". The
# page explains its labels, so a wrong one is worse than a missing one: the
# previous two-state version called every 2026-27 game training data, which is
# the exact inverse of the truth - the models were fitted through 2025-26 only.
#
# "Geri test" and "tahmin" are kept apart on purpose. A played game's team
# sheet is read from the box score of that same game, so the model is told who
# actually played while a forecast only knows who was expected to. Only the
# unplayed rows are forecasts; everything else is a backtest, held-out or not.
BADGE_UPCOMING = {
    "key": "upcoming",
    "label": "Henuz oynanmadi - tip-off oncesi gercek tahmin",
}
BADGE_UNSEEN = {
    "key": "unseen",
    "label": "Modelin gormedigi sezon - geri test (kadro kutu skorundan)",
}
BADGE_OUT_OF_SAMPLE = {
    "key": "out-of-sample",
    "label": "Held-out - egitimde yok, geri test (kadro kutu skorundan)",
}
BADGE_IN_SAMPLE = {
    "key": "in-sample",
    "label": "Egitim verisinde - model bu sonucu gordu, tahmin iyimser",
}
# Tip-off oncesi yazildi, sonra oynandi: panodaki tek dogrulanmis tahmin
# cinsi. Kadro mac oncesi listeden geldi, kutu skorundan degil - bu yuzden
# "unseen" rozetinin parantezi bu maclar icin yanlis olurdu.
BADGE_PRE_TIPOFF = {
    "key": "pretip",
    "label": "Tip-off oncesi yazildi, sonucu sonra geldi - dogrulanmis tahmin",
}

app = Flask(__name__)
_STATE = None


def _load_sibling(name):
    """prediction_engines/ holds scripts, not a package - import them by path."""
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


def _trained_features(bundle):
    """The columns the models on disk were fitted on, in their order.

    The trainer writes them beside the models because the shipped set may be
    wider than the dataset's base list (the who-is-out columns), and a column
    the model never saw must not be handed to it in another one's place.
    """
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, encoding="utf-8") as f:
            listed = json.load(f)["features"]
        missing = [c for c in listed if c not in bundle["dataset"].columns]
        if missing:
            raise RuntimeError(f"model {len(missing)} feature bekliyor ama dataset'te yok: "
                               f"{missing[:5]} - dataset ve modeller ayni build'den mi?")
        return listed
    # Without the file, the widest list the build itself calls production. The
    # asof blob carries both: "features" is 190 names, "production_features"
    # the 196 the models were actually fitted on. Reaching for the 190 would
    # hand the models a frame they were never fitted on - silently, which is
    # the failure this file's header was written about.
    return bundle.get("production_features") or bundle["features"]


def _load_registered(name):
    """A sibling module that defines dataclasses, loaded once.

    Registered in sys.modules before it runs: @dataclass resolves annotations
    through it, and season_sim loads its own siblings the same way.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(PROJECT_ROOT, "prediction_engines", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _load_sim_season():
    """The simulated season, or None when there is none to show."""
    try:
        season = _load_registered("season_sim").SimSeason.load(SIM_RUN)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[startup] simule sezon okunamadi ({exc}) - o bolum kapali")
        return None
    if season is None:
        print(f"[startup] simule sezon yok (sim_sandbox/{SIM_RUN}) - o bolum kapali")
    else:
        season.ranking_engine()       # built now, so the first click does not wait for it
        print(f"[startup] simule sezon: {season.season}, {len(season.games)} mac")
    return season


def _load_optional(path, label):
    """Load a pickle that the dashboard degrades gracefully without."""
    if not os.path.exists(path):
        print(f"[startup] {label} yok ({os.path.basename(path)}) - o panel kapali")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        print(f"[startup] {label} okunamadi: {exc} - o panel kapali")
        return None


def _dataset_choice():
    """(path, name) of the dataset to serve: the as-of build when it is there."""
    if os.path.exists(ASOF_PATH):
        return ASOF_PATH, "asof"
    print(f"[startup] {os.path.basename(ASOF_PATH)} yok "
          f"(build_dataset_asof.py kosturulmadi) - uretim veri seti servis ediliyor, "
          f"yurumekte olan sezon gorunmeyecek")
    return TRAINED_PATH, "trained"


SCHEDULE_COLUMNS = ["game_id", "date", "season", "season_type", "status",
                    "tipoff_utc", "home_team", "away_team",
                    "home_score", "away_score"]


def _schedule_frame():
    """Every game the NBA has scheduled, read from game_ids/schedule_*.csv.

    Two readers need this file. The game list needs the tip-off time, because
    the dataset's game_date is a midnight datetime and carries no time, so a
    day's games cannot be ordered from it. The calendar needs the days
    themselves: the dataset only holds the games the builder has seen - every
    game played, plus the single slate it was built for - so a month drawn
    from it alone would show an empty April, and would lose a day's games the
    moment the build went stale. Reading the file once here keeps the two from
    disagreeing about what is on the schedule.
    """
    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "game_ids", "schedule_*.csv")))
    frames = []
    for path in paths:
        name = os.path.basename(path)
        season = name[len("schedule_"):-len(".csv")]
        try:
            raw = pd.read_csv(path, dtype={"game_id": str})
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"[startup] fikstur okunamadi ({name}: {exc}) - atlandi")
            continue
        if not {"game_id", "game_date", "home", "away"}.issubset(raw.columns):
            print(f"[startup] {name} beklenen sutunlari tasimiyor - atlandi")
            continue
        def column(col, default):
            return raw[col] if col in raw.columns else pd.Series(default, index=raw.index)
        frames.append(pd.DataFrame({
            "game_id": raw["game_id"].astype(str).str.zfill(10),
            "date": pd.to_datetime(raw["game_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "season": season,
            "season_type": column("season_type", "regular_season").astype(str),
            "status": column("status", "").astype(str),
            "tipoff_utc": pd.to_datetime(column("tipoff_utc", pd.NaT),
                                         utc=True, errors="coerce"),
            "home_team": raw["home"].astype(str),
            "away_team": raw["away"].astype(str),
            "home_score": pd.to_numeric(column("home_score", np.nan), errors="coerce"),
            "away_score": pd.to_numeric(column("away_score", np.nan), errors="coerce"),
        }))
    if not frames:
        print("[startup] game_ids/schedule_*.csv yok - tip-off saati gosterilmeyecek, "
              "takvim yalnizca veri setindeki maclari gosterir")
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[out["date"].notna()].drop_duplicates("game_id").reset_index(drop=True)


def _schedule_results(schedule):
    """{game_id: (home_score, away_score)} for every finished game on file.

    The one source of a result that is newer than the nightly build, and the
    reason the page can grade a prediction the same day it was made.
    """
    if schedule.empty:
        return {}
    done = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()]
    return {gid: (float(h), float(a)) for gid, h, a
            in zip(done["game_id"], done["home_score"], done["away_score"])}


def _tipoffs(schedule):
    """{game_id: tip-off, UTC} off the schedule frame, or {} without one."""
    if schedule.empty:
        return {}
    return {gid: t.isoformat()
            for gid, t in zip(schedule["game_id"], schedule["tipoff_utc"])
            if pd.notna(t)}


def _calendar_spine(dataset, schedule):
    """One row per game a calendar month can draw, predictions where they exist.

    Three kinds of row land here and the page has to tell them apart: a game
    with a result and a prediction, a game whose result the current build has
    not reached yet (played, p_home missing), and a fixture (neither). Nothing
    is coerced - a missing probability stays missing - so no fixture can be
    counted as a hit or a miss, which is the same rule _game_summary follows.
    """
    played = dataset["is_played"].to_numpy()
    spine = pd.DataFrame({
        "game_id": dataset["game_id"].to_numpy(),
        "date": dataset["game_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "season": dataset["season"].to_numpy(),
        "season_type": "regular_season",
        "status": "",
        "home_team": dataset["home_team"].to_numpy(),
        "away_team": dataset["away_team"].to_numpy(),
        "home_score": dataset["home_score"].where(dataset["is_played"]).to_numpy(),
        "away_score": dataset["away_score"].where(dataset["is_played"]).to_numpy(),
        "is_played": played,
        "p_home": dataset["model_home_prob"].to_numpy(),
    })
    rest = schedule[~schedule["game_id"].isin(set(spine["game_id"]))]
    if not rest.empty:
        spine = pd.concat([spine, pd.DataFrame({
            "game_id": rest["game_id"].to_numpy(),
            "date": rest["date"].to_numpy(),
            "season": rest["season"].to_numpy(),
            "season_type": rest["season_type"].to_numpy(),
            "status": rest["status"].to_numpy(),
            "home_team": rest["home_team"].to_numpy(),
            "away_team": rest["away_team"].to_numpy(),
            "home_score": rest["home_score"].to_numpy(),
            "away_score": rest["away_score"].to_numpy(),
            # The score is the evidence, not the status string: a file that
            # says "final" with no score is not a played game.
            "is_played": rest["home_score"].notna().to_numpy(),
            "p_home": np.nan,
        })], ignore_index=True)
    # The dataset is rebuilt once a day, so between two builds it holds the
    # night's games as fixtures while the schedule file already has their
    # final scores. Without this the calendar would show a finished game as
    # "not played yet" for a day, and - worse - would leave the one
    # prediction that was written before tip-off ungraded forever.
    if not schedule.empty:
        sch = schedule.set_index("game_id")
        for col in ("home_score", "away_score"):
            spine[col] = spine[col].where(spine[col].notna(),
                                          spine["game_id"].map(sch[col]))
        for col in ("season_type", "status"):
            spine[col] = spine["game_id"].map(sch[col]).fillna(spine[col])
    scored = spine["home_score"].notna() & spine["away_score"].notna()
    # A prediction the model produced while the builder still called the game
    # a fixture, on a game that has since finished, is a true pre-tip-off
    # forecast: no part of the result could have reached its features. That is
    # a different and much stronger claim than a backtest row, so it is
    # carried to the page rather than averaged in silently.
    spine["pre_tipoff"] = (~spine["is_played"]) & scored & spine["p_home"].notna()
    spine["is_played"] = spine["is_played"] | scored
    spine["month"] = spine["date"].str.slice(0, 7)
    return spine.sort_values(["date", "game_id"]).reset_index(drop=True)


def _load_state():
    dataset_path, dataset_name = _dataset_choice()
    try:
        with open(dataset_path, "rb") as f:
            bundle = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        # One readable line instead of a traceback. The builder writes this
        # file atomically now, so a half-written pickle should not happen -
        # but if it ever does, the reader must say what to run.
        raise SystemExit(f"{os.path.basename(dataset_path)} okunamadi ({exc}). "
                         f"Yeniden kurmak icin: py prediction_engines/build_dataset_asof.py") from exc
    dataset = bundle["dataset"].copy()
    features = _trained_features(bundle)
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    # Every branch below asks "is there a result here" through this one column,
    # so the older build - which has no such column and no unplayed games - is
    # given one rather than leaving each reader to guess.
    if "is_played" in dataset.columns:
        dataset["is_played"] = dataset["is_played"].astype(bool)
    else:
        dataset["is_played"] = True

    # 75% chronological split of the target season, exactly as NBAPredictor.train
    # does it: split the season subset in its AS-SAVED row order. This must
    # happen before the display re-sort below - sort_values is not stable, so
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
    # The blended forecaster is what the page quotes when it exists: +0.0183
    # accuracy and +0.0150 AUC over the classifier alone across 430
    # walk-forward cells. None until retrain_production has been run since the
    # blend was added, and the page falls back to the classifier.
    blend = ensemble_module.load_blend(MODEL_DIR)

    calibrator = _load_optional(CALIBRATOR_PATH, "kalibrator")
    simulator = _load_optional(SIMULATOR_PATH, "simulator")

    ratings_module = _load_sibling("team_ratings")
    rated = ratings_module.add_rating_features(dataset)
    rating_cols = [c for c in ratings_module.RATING_FEATURE_COLS if c in rated.columns]
    ratings_by_game = rated.set_index("game_id")[rating_cols] if rating_cols else None

    rankings_module = _load_sibling("power_rankings")
    player_games = rankings_module.attach_value(_attach_minutes(_build_player_games(dataset)))
    # Played games only. A ranking describes what teams have done, so a
    # fixture has nothing to contribute - and it does harm: the engine builds
    # its periods from the games it is given, so today's fixtures created a
    # week period of their own and the page opened that period by default,
    # showing all 30 teams with 0 games and an empty form bar.
    rankings = rankings_module.RankingEngine(dataset[dataset["is_played"]].copy(), player_games)
    calibration_module = _load_sibling("calibration") if calibrator else None
    # One forecast per game up front, so the list can show the model's pick
    # beside every result without a model call per card. ~10,000 rows through
    # the blend is a few seconds once; per request it would be a few seconds
    # every time someone scrolls.
    dataset["model_home_prob"] = _forecast_all(dataset, features, models, blend,
                                               calibrator, calibration_module)
    n_played = int(dataset["is_played"].sum())
    current_season = max(dataset["season"])
    print(f"[startup] {dataset.shape[0]} mac x {dataset.shape[1]} sutun "
          f"({dataset_name}, as-of {bundle.get('as_of') or '-'}), "
          f"oynanmis {n_played}, oynanmamis {len(dataset) - n_played}, "
          f"{len(features)} feature, {len(models)} model, "
          f"{len(player_games)} oyuncu-mac kaydi, "
          f"held-out {len(test_game_ids)} mac ({split_date} sonrasi), "
          f"son sezon {current_season}")
    if current_season > TARGET_SEASON:
        seen = int((dataset["season"] > TARGET_SEASON).sum())
        print(f"[startup] {seen} mac modellerin egitildigi sezondan sonra "
              f"({TARGET_SEASON} sonrasi) - hepsi out-of-sample")
    # The impact cache is a separate file that nothing rebuilds automatically.
    # When it falls behind the games, the rankings and the roster panel quietly
    # describe an older league; say so here rather than letting the page look
    # merely empty.
    # Built here rather than per request: it is a 12,000-row join of two
    # frames that are both already in memory, and the calendar is the view the
    # page opens on.
    schedule = _schedule_frame()
    results = _schedule_results(schedule)
    # Games the build still calls fixtures although the schedule file has
    # their final score. Counted once here so every number on the page -
    # the headline count, the list's denominator, the last-played date -
    # agrees with what the cards show.
    late_ids = {gid for gid in dataset.loc[~dataset["is_played"], "game_id"]
                if gid in results}
    if late_ids:
        print(f"[startup] {len(late_ids)} mac veri setinde fikstur ama fiksturde "
              f"skoru var - tip-off oncesi yazilmis tahminler, sonuclari "
              f"fikstur dosyasindan okunuyor")
    calendar = _calendar_spine(dataset, schedule)
    cal_months = sorted(calendar["month"].unique().tolist())
    print(f"[startup] takvim {len(calendar)} mac, {len(cal_months)} ay "
          f"({cal_months[0] if cal_months else '-'} -> "
          f"{cal_months[-1] if cal_months else '-'}), "
          f"fikstur dosyasindan gelen fazladan mac "
          f"{len(calendar) - len(dataset)}")
    last_played = dataset.loc[dataset["is_played"], "game_date"].max()
    if len(player_games) and player_games["game_date"].max() < last_played:
        print(f"[startup] UYARI: impact onbellegi "
              f"{player_games['game_date'].max().date()} tarihinde bitiyor, son mac "
              f"{last_played.date()} - guc siralamasi ve kadro paneli geride. "
              f"Duzeltmek icin: py prediction_engines/backfill_impact_from_db.py "
              f"--source csv --seasons {current_season}")
    return {"dataset": dataset, "features": features, "models": models,
            "blend": blend, "team_ids": _team_ids(dataset), "rankings": rankings,
            "sim": _load_sim_season(),
            "test_game_ids": test_game_ids, "split_date": split_date,
            "player_games": player_games, "calibrator": calibrator,
            "simulator": simulator, "ratings": ratings_by_game,
            "dataset_name": dataset_name, "as_of": bundle.get("as_of"),
            # How far the impact cache reaches. The roster panel and the
            # rankings both come from it and it does not grow by itself, so
            # this date is the difference between "no players to show" and
            # "nobody rebuilt the cache".
            "player_games_through": (str(player_games["game_date"].max().date())
                                     if len(player_games) else None),
            "tipoffs": _tipoffs(schedule), "schedule": schedule,
            "schedule_results": results, "late_ids": late_ids,
            "calendar": calendar, "calendar_months": cal_months,
            "n_played": n_played,
            "current_season": current_season,
            "simulation_module": _load_sibling("simulation") if simulator else None,
            "calibration_module": calibration_module}


def _attach_minutes(player_games):
    """Minutes and full names per player-game from the box scores (csv_tables.py).

    The rankings weight a roster by minutes and refuse to rate a player who
    barely played; the impact cache only knows surnames, and a ranking that
    says "Williams" is not a ranking.
    """
    if player_games.empty:
        return player_games
    pregame = _load_sibling("pregame_roster")
    box = pregame.load_box()[["game_id", "player_id", "player_name", "min"]]
    box = pd.DataFrame({
        "game_id": box["game_id"].astype(str).str.zfill(10),
        "key": box["player_id"].astype(str),
        "full_name": box["player_name"].astype(str),
        "minutes": pregame.parse_minutes(box["min"]),
    })
    pg = player_games.copy()
    pg["key"] = pg["player"].astype(str)
    pg = pg.merge(box, on=["game_id", "key"], how="left")
    pg["name"] = pg["full_name"].where(pg["full_name"].notna(), pg["name"])
    return pg.drop(columns=["key", "full_name"])


def _forecast_all(dataset, features, models, blend, calibrator, calibration_module):
    """The shown win probability for every game: blend, else calibrated, else raw.

    Same order of preference api_predict uses, so a card and its detail page
    never disagree about the pick.
    """
    X = dataset[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    if blend is not None:
        return blend.predict_proba(X)[:, 1]
    raw = models["home_win"].predict_proba(X)[:, 1]
    if calibrator and calibration_module is not None:
        return calibration_module.apply_calibrator(calibrator["method"],
                                                   calibrator["calibrator"], raw)
    return raw


def _team_ids(dataset):
    """{tricode: NBA team id}, read off the games rather than hardcoded, so a
    relocated or renamed franchise shows up the moment its games do."""
    sides = pd.concat([
        dataset[["home_team", "home_team_id"]].set_axis(["abbr", "team_id"], axis=1),
        dataset[["away_team", "away_team_id"]].set_axis(["abbr", "team_id"], axis=1),
    ]).dropna().drop_duplicates("abbr")
    return {str(a): int(t) for a, t in zip(sides["abbr"], sides["team_id"])}


def get_state():
    """Create-once accessor - nothing is reloaded per request."""
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
def _badge(state, game_id, season, is_played, pre_tipoff=False):
    """Which of the five states this game is in. Nothing falls through."""
    if not is_played:
        return BADGE_UPCOMING
    if pre_tipoff:
        return BADGE_PRE_TIPOFF
    if season > TARGET_SEASON:
        return BADGE_UNSEEN
    if season == TARGET_SEASON and game_id in state["test_game_ids"]:
        return BADGE_OUT_OF_SAMPLE
    return BADGE_IN_SAMPLE


def _game_summary(state, g):
    """One card's worth of a game, played or not.

    A game with no result yet sends JSON null for the five result fields and
    for model_correct, never a coerced value: bool(float('nan')) is True and
    g["home_win"] == 1 is False, so coercing would make every fixture either
    "the home team won" or "the model got it wrong", and the page would count
    it in the accuracy on screen. is_played rides along so the card and every
    counter can tell the two apart without inspecting the fields.
    """
    p_home = float(g["model_home_prob"])
    built_played = bool(g["is_played"])
    if built_played:
        home_score, away_score = float(g["home_score"]), float(g["away_score"])
    else:
        # The build is a day old at most, but that is long enough: last night's
        # games sit in it as fixtures while the schedule file already has their
        # scores. Taking the result from there is what lets the page grade the
        # predictions that were written before tip-off - the only ones on the
        # dashboard that are forecasts rather than backtests - instead of
        # printing "not played yet" under a final score.
        late = state["schedule_results"].get(g["game_id"])
        home_score, away_score = late if late else (None, None)
    played = home_score is not None
    home_won = (home_score > away_score) if played else None
    return {
        "game_id": g["game_id"],
        "date": str(g["game_date"].date()),
        "season": g["season"],
        "is_played": played,
        "pre_tipoff": played and not built_played,
        "tipoff_utc": state["tipoffs"].get(g["game_id"]),
        "matchup": f"{g['away_team']} @ {g['home_team']}",
        "home_team": g["home_team"],
        "away_team": g["away_team"],
        "home_team_id": state["team_ids"].get(g["home_team"]),
        "away_team_id": state["team_ids"].get(g["away_team"]),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_won,
        "model_home_prob": round(p_home, 3),
        "model_pick": g["home_team"] if p_home > 0.5 else g["away_team"],
        # Guven = max(p, 1-p): modelin SECTIGI tarafa ne kadar emin oldugu.
        # has_pred burada her zaman True - bu uc nokta yalnizca veri setindeki
        # maclari servis eder - ama alan takvimdekiyle ayni adi tasiyor, cunku
        # sayfadaki guven filtresi ikisini de ayni kodla suzuyor.
        "confidence": round(max(p_home, 1.0 - p_home), 3),
        "has_pred": True,
        "model_correct": ((p_home > 0.5) == home_won) if played else None,
        "badge": _badge(state, g["game_id"], g["season"], played,
                        played and not built_played),
    }


def _roster_impact_for_team(state, team, as_of_date, top_n=6):
    """Top players by trailing-10-game mean impact strictly before a date.

    A player belongs to the team's roster if their most recent appearance
    before `as_of_date` was for that team (handles trades); trailing means are
    taken over their own history regardless of jersey, mirroring
    get_current_roster_impact's strictly-before-the-game windows.

    It refuses to answer from a stale cache. The player history comes from
    game_impact_cache_v4.pkl on disk, which does not grow by itself: when it
    stops before the game being shown, "the most recent appearance" is from
    last season and this returned whoever finished that season on the team -
    every trade and signing since then invisible, under a panel the reader
    takes for today's team sheet. Nothing errored and nothing looked wrong.
    So if the team has no player-game within ROSTER_STALE_DAYS before the
    game, the answer is nothing rather than a guess, and the caller can say
    the cache needs rebuilding (backfill_impact_from_db.py).
    """
    pg = state["player_games"]
    if pg is None or pg.empty:
        return []
    hist = pg[pg["game_date"] < as_of_date]
    if hist.empty:
        return []
    fresh = hist[hist["team"] == team]
    if fresh.empty or (as_of_date - fresh["game_date"].max()).days > ROSTER_STALE_DAYS:
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
        "team_ids": state["team_ids"],
        "months": sorted(df["month"].unique().tolist(), reverse=True),
        "split_date": state["split_date"],
        "n_held_out": len(state["test_game_ids"]),
        # Played games only. The fixtures are counted separately, because the
        # brand line read "10,903 mac" while 8 of them had not been played.
        "n_games": state["n_played"] + len(state["late_ids"]),
        "n_unplayed": int(len(df)) - state["n_played"] - len(state["late_ids"]),
        # Every game the models were never fitted on: the held-out slice of the
        # season they were trained through, plus every season after it. The
        # page used to name only the first group, which made its own honesty
        # paragraph wrong about the 146 games of the season in progress.
        "n_unseen_season": int((df["season"] > TARGET_SEASON).sum()),
        "trained_through": TARGET_SEASON,
        "current_season": state["current_season"],
        # The build is rebuilt daily and gitignored, so staleness is a real
        # failure mode: an old build would show finished games as fixtures
        # forever with nothing on screen to say so.
        "dataset": state["dataset_name"],
        "as_of": state["as_of"],
        "player_games_through": state["player_games_through"],
        # So the page can see for itself whether the impact cache has fallen
        # behind the games, instead of trusting a server-side print nobody
        # reads when the dashboard runs detached.
        "last_played": str(df.loc[df["is_played"] | df["game_id"].isin(state["late_ids"]),
                                  "game_date"].max().date()),
        "metrics": _honest_metrics(),
        "has_calibrator": state["calibrator"] is not None,
        "has_simulator": state["simulator"] is not None,
        "sim_season": state["sim"].season if state["sim"] is not None else None,
        # Named so it cannot be read as the live season: both are 2026-27 now.
        "sim_is_simulation": True,
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
    # Counted apart so the accuracy strip on the page has a denominator of
    # games that were actually played - including the ones whose score
    # arrived after the build, which the cards already grade.
    played = int((df["is_played"] | df["game_id"].isin(state["late_ids"])).sum())
    return jsonify({"count": int(len(df)), "count_played": played,
                    "count_unplayed": int(len(df)) - played,
                    "shown": len(games), "games": games})


def _calendar_entry(state, row):
    """One game as a calendar cell shows it: pick, confidence, outcome or none.

    Confidence is max(p, 1-p) - how sure the model is about the side it named,
    not about the home team. It is sent from here so the page never has to
    re-derive it and the two cannot drift apart.
    """
    played = bool(row.is_played)
    has_pred = bool(pd.notna(row.p_home))
    p_home = float(row.p_home) if has_pred else None
    scored = pd.notna(row.home_score) and pd.notna(row.away_score)
    home_won = bool(float(row.home_score) > float(row.away_score)) if scored else None
    pick = (row.home_team if p_home > 0.5 else row.away_team) if has_pred else None
    return {
        "game_id": row.game_id,
        "date": row.date,
        "season": row.season,
        "season_type": row.season_type,
        "tipoff_utc": state["tipoffs"].get(row.game_id),
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_score": float(row.home_score) if pd.notna(row.home_score) else None,
        "away_score": float(row.away_score) if pd.notna(row.away_score) else None,
        "is_played": played,
        "home_win": home_won,
        # False means "this game has no prediction", which is not the same as
        # "the model was wrong" and not the same as "not played yet". A game
        # played after the last build is in exactly this state.
        "has_pred": has_pred,
        # The prediction was written before tip-off and the game is over: the
        # only kind of row that is a forecast rather than a backtest.
        "pre_tipoff": bool(row.pre_tipoff),
        "model_home_prob": round(p_home, 3) if has_pred else None,
        "model_pick": pick,
        "confidence": round(max(p_home, 1.0 - p_home), 3) if has_pred else None,
        "model_correct": ((p_home > 0.5) == home_won)
                         if (has_pred and home_won is not None) else None,
        "badge": _badge(state, row.game_id, row.season, played,
                        bool(row.pre_tipoff)) if has_pred else None,
    }


def _default_month(months):
    """The month the calendar opens on: this one when it has games, else the
    next one that does. Opening on an empty grid reads as a broken page."""
    today = dt.date.today().strftime("%Y-%m")
    if not months or today in months:
        return today
    ahead = [m for m in months if m > today]
    return ahead[0] if ahead else months[-1]


@app.route("/api/calendar")
def api_calendar():
    """A month of the schedule, day by day, with the model's pick on each game.

    The whole month is sent in one response - a day at a time would be 30
    requests to page through a season - but a month is at most ~230 games, far
    short of what the game list already sends.
    """
    state = get_state()
    cal = state["calendar"]
    months = state["calendar_months"]
    month = request.args.get("month", "").strip()
    if month not in months:
        month = _default_month(months)
    team = request.args.get("team", "").strip().upper()
    rows = cal[cal["month"] == month]
    if team:
        rows = rows[(rows["home_team"] == team) | (rows["away_team"] == team)]
    by_day = {}
    for row in rows.itertuples(index=False):
        by_day.setdefault(row.date, []).append(_calendar_entry(state, row))
    days, n, graded, hits = [], 0, 0, 0
    for date in sorted(by_day):
        games = sorted(by_day[date], key=lambda g: (g["tipoff_utc"] or "", g["game_id"]))
        # Graded = played AND predicted. The two reasons a game is ungraded -
        # not played, or played without a prediction - are both counted apart
        # so the month header can say which it is.
        day_graded = sum(1 for g in games if g["model_correct"] is not None)
        day_hits = sum(1 for g in games if g["model_correct"])
        days.append({"date": date, "n": len(games), "games": games,
                     "graded": day_graded, "hits": day_hits})
        n += len(games)
        graded += day_graded
        hits += day_hits
    return jsonify({
        "month": month, "months": months, "team": team or None,
        "today": dt.date.today().isoformat(),
        "n": n, "graded": graded, "hits": hits,
        "n_unplayed": sum(1 for d in days for g in d["games"] if not g["is_played"]),
        "n_unpredicted": sum(1 for d in days for g in d["games"]
                             if g["is_played"] and not g["has_pred"]),
        "n_pre_tipoff": sum(1 for d in days for g in d["games"] if g["pre_tipoff"]),
        "days": days,
    })


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
    predicted["calibrated"] = calibrated

    # Order of preference: blend, then calibrated classifier, then raw. The
    # blend is not a calibrated version of the classifier - it is a different
    # forecast, and it is the only one of the three measured to carry more
    # information rather than to read the same information better.
    blend = state.get("blend")
    if blend is not None:
        views = blend.views(X)
        predicted["views"] = {k: round(float(v[0]), 3) for k, v in views.items()}
        predicted["view_spread"] = round(
            float(max(v[0] for v in views.values())
                  - min(v[0] for v in views.values())), 3)
        shown_prob = float(blend.predict_proba(X)[:, 1][0])
        predicted["source"] = "blend"
    else:
        shown_prob = calibrated["prob"] if calibrated else raw_prob
        predicted["source"] = "calibrated" if calibrated else "raw"
    predicted["home_win_prob"] = round(shown_prob, 3)
    predicted["predicted_winner"] = g["home_team"] if shown_prob > 0.5 else g["away_team"]

    out = _game_summary(state, g)
    out["predicted"] = predicted
    out["simulation"] = _simulate_game(state, row)
    out["ratings"] = _ratings_for_game(state, gid)
    # The block stays, with every field null, instead of becoming null itself:
    # the page reads `actual.home_score` in six places, so a missing object
    # would throw where a missing number shows a dash. `correct_winner` is
    # null, never False - grading the model against a game nobody has played
    # is how the page came to print "wrong" next to tonight's forecast.
    # Read back off the summary rather than off the dataset row, so a game
    # whose result arrived after the last build is scored here exactly as the
    # card above it scores it.
    home_score, away_score = out["home_score"], out["away_score"]
    played = home_score is not None
    actual_winner = None
    if played:
        actual_winner = g["home_team"] if out["home_win"] else g["away_team"]
    out["actual"] = {
        "home_score": home_score,
        "away_score": away_score,
        "point_diff": (home_score - away_score) if played else None,
        "total_score": (home_score + away_score) if played else None,
        "home_win": int(bool(out["home_win"])) if played else None,
        "winner": actual_winner,
        "correct_winner": (bool(predicted["predicted_winner"] == actual_winner)
                           if played else None),
    }
    out["roster_impact"] = {
        "home": _roster_impact_for_team(state, g["home_team"], g["game_date"]),
        "away": _roster_impact_for_team(state, g["away_team"], g["game_date"]),
    }
    out["matchup_data"] = _matchup_data(g["season"], gid)
    return jsonify(out)


RANKING_WINDOWS = ("week", "month", "season")


@app.route("/api/rankings/periods")
def api_ranking_periods():
    engine = get_state()["rankings"]
    return jsonify({w: engine.periods(w) for w in RANKING_WINDOWS})


@app.route("/api/rankings")
def api_rankings():
    engine = get_state()["rankings"]
    window = request.args.get("window", "season").strip()
    if window not in RANKING_WINDOWS:
        return jsonify({"error": f"pencere week/month/season olmali: {window}"}), 400
    key = request.args.get("key", "").strip() or engine.periods(window)[-1]["key"]
    try:
        return jsonify(engine.rank(window, key))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/rankings/players")
def api_ranking_players():
    engine = get_state()["rankings"]
    window = request.args.get("window", "season").strip()
    if window not in RANKING_WINDOWS:
        return jsonify({"error": f"pencere week/month/season olmali: {window}"}), 400
    key = request.args.get("key", "").strip() or engine.periods(window)[-1]["key"]
    team = request.args.get("team", "").strip().upper() or None
    limit = min(int(request.args.get("limit", 40) or 40), 200)
    try:
        return jsonify({"window": window, "key": key, "team": team,
                        "players": engine.players(window, key, team, limit)})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


# ===========================================================================
#  The simulated season (read-only, sim_sandbox/)
# ===========================================================================
NO_SIM = ({"error": "simule sezon yok - once season_sim.py forecast calistirin"}, 404)


def _sim():
    return get_state()["sim"]


@app.route("/api/sim/meta")
def api_sim_meta():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    return jsonify(sim.meta())


@app.route("/api/sim/standings")
def api_sim_standings():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    return jsonify({"teams": sim.standings_rows()})


@app.route("/api/sim/leaders")
def api_sim_leaders():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    team = request.args.get("team", "").strip().upper() or None
    limit = min(int(request.args.get("limit", 10) or 10), 50)
    return jsonify(sim.leaders(team, limit))


@app.route("/api/sim/games")
def api_sim_games():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    team = request.args.get("team", "").strip().upper() or None
    month = request.args.get("month", "").strip() or None
    limit = min(int(request.args.get("limit", 300) or 300), 1500)
    return jsonify(sim.games_rows(team, month, limit))


@app.route("/api/sim/game/<game_id>")
def api_sim_game(game_id):
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    game = sim.game(game_id)
    if game is None:
        return jsonify({"error": f"simule sezonda {game_id} yok"}), 404
    return jsonify(game)


@app.route("/api/sim/rankings/periods")
def api_sim_ranking_periods():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    return jsonify({w: sim.ranking_periods(w) for w in RANKING_WINDOWS})


@app.route("/api/sim/rankings")
def api_sim_rankings():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    window = request.args.get("window", "season").strip()
    if window not in RANKING_WINDOWS:
        return jsonify({"error": f"pencere week/month/season olmali: {window}"}), 400
    periods = sim.ranking_periods(window)
    key = request.args.get("key", "").strip() or periods[-1]["key"]
    if key not in {p["key"] for p in periods}:
        return jsonify({"error": f"simule sezonda {window} donemi yok: {key}"}), 404
    return jsonify(sim.ranking_engine().rank(window, key))


@app.route("/api/sim/rankings/players")
def api_sim_ranking_players():
    sim = _sim()
    if sim is None:
        return jsonify(NO_SIM[0]), NO_SIM[1]
    window = request.args.get("window", "season").strip()
    if window not in RANKING_WINDOWS:
        return jsonify({"error": f"pencere week/month/season olmali: {window}"}), 400
    periods = sim.ranking_periods(window)
    key = request.args.get("key", "").strip() or periods[-1]["key"]
    if key not in {p["key"] for p in periods}:
        return jsonify({"error": f"simule sezonda {window} donemi yok: {key}"}), 404
    team = request.args.get("team", "").strip().upper() or None
    limit = min(int(request.args.get("limit", 40) or 40), 200)
    return jsonify({"window": window, "key": key, "team": team,
                    "players": sim.ranking_engine().players(window, key, team, limit)})


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
