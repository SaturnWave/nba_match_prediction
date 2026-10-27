"""
The production dataset rebuilt from the CSVs, with the next unplayed games added.

WHY
    The production dataset (output/engineered_dataset_pregame.pkl) ends with
    2025-26 and was built through a chain of scripts, two of which read the
    phone database's copy. Nothing could put a 2026-27 game into it - played
    or not - so the trained models could not forecast a single game of the
    season. This rebuilds the same dataset from nba_data/ alone and adds the
    games still to come, so tonight's games get a row built from what is
    known before tonight.

WHAT IT DOES, IN ORDER
    1. csv_tables.update(): the tables current with nba_data/.
    2. value_engine on games missing from output/value_cache_unweighted.pkl,
       and on games whose tracking arrived after they were valued, without
       leverage weighting, as the model's cache has always been built; then
       impact_v5 writes game_impact_cache_v5.pkl from it.
    3. build_dataset_db.build(source="csv") with the unplayed games appended,
       then pregame_roster's replacement of the roster family. An unplayed
       game's team sheet is carried forward from each team's last game
       (pregame_roster.carried_forward_listings), until the injury report is
       archived.
    4. Checks. A played game with no score stops the build, so does a box-score
       comment nobody has classified. With --check the played rows must
       reproduce the production dataset: every production feature and every
       label, game by game, once the matchup files that arrived after
       production was built (LATE_MATCHUPS) are left out of the comparison.

UNPLAYED GAMES
    The unplayed games are the schedule's regular-season games that nba_data/
    does not have yet, from the earliest date on. A forecast can only use
    games already played, so every unplayed game is built on the same
    history: all games played so far, nothing between now and its date. The
    games of one date are built together (a team plays at most once a day).
    Several dates are built one pass each, every date as if it were the next.

    A game that is over but missing from nba_data/ - final in the schedule,
    or tipped off more than OVER_AFTER ago - stops the build: every later
    game would be built on a gap. Refresh the schedule and the data first.

Run:  py prediction_engines/build_dataset_asof.py              (the next date with games)
      py prediction_engines/build_dataset_asof.py --days 3     (the next three dates)
      py prediction_engines/build_dataset_asof.py --check      (and the parity test)
Output: output/engineered_dataset_asof.pkl, output/asof_parity.json with --check
"""
import argparse
import datetime as dt
import glob
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
OUT_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_asof.pkl")
PRODUCTION_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
PARITY_PATH = os.path.join(OUTPUT_DIR, "asof_parity.json")
FEATURE_LIST_PATH = os.path.join(PROJECT_ROOT, "models", "feature_list_2025_26.json")
VALUE_CACHE = os.path.join(OUTPUT_DIR, "value_cache_unweighted.pkl")
PARITY_TOLERANCE = 1e-9
ID_COLUMNS = ["game_id", "season", "game_date", "home_team", "away_team"]
OVER_AFTER = pd.Timedelta(hours=4)       # tip-off to final buzzer, with room for overtime

# Matchup files that reached nba_data/ after the production dataset was built.
# The rebuild uses them, and through each player's accumulated matchups they
# move the matchup columns of every later game those players are in. The parity
# test sweeps the roster again without them. Empty this once production is
# rebuilt from this pipeline.
LATE_MATCHUPS = {
    "0022400856": "2026-10-24 20:34'te indi; uretim veri seti 12:49'da kurulmustu",
}

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import csv_tables  # noqa: E402 - one instance per process, shared with the modules below


def _load_sibling(name):
    """A sibling module, registered under its name so later loads reuse it."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                  # value_engine's dataclasses resolve through this
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
#  1-2. Inputs brought up to date
# ---------------------------------------------------------------------------
def refresh_impact(verbose=True):
    """Value every unvalued (or late-tracked) game, then rewrite the v5 impact cache."""
    value_engine = _load_sibling("value_engine")
    impact_v5 = _load_sibling("impact_v5")
    seasons = sorted(csv_tables.load("games")["season"].unique())
    cache = value_engine.load_cache(VALUE_CACHE)
    before = len(cache)
    cache = value_engine.run_seasons(seasons, cache, verbose=verbose, cache_path=VALUE_CACHE,
                                     use_leverage=False)
    games = csv_tables.load("game_summary")
    v5 = impact_v5.build_v5(cache, games)
    with open(impact_v5.V5_PATH, "wb") as f:
        pickle.dump(v5, f)
    if verbose:
        print(f"  deger onbellegi {before:,} -> {len(cache):,} mac; impact v5 {len(v5):,} mac", flush=True)


# ---------------------------------------------------------------------------
#  The unplayed games
# ---------------------------------------------------------------------------
def current_schedule():
    """(season, schedule) from the newest game_ids/schedule_<season>.csv."""
    paths = sorted(glob.glob(os.path.join(GAME_IDS_DIR, "schedule_*.csv")))
    if not paths:
        raise FileNotFoundError("game_ids/schedule_*.csv yok - once data_retriaval/fetch_schedule.py")
    path = paths[-1]
    season = os.path.basename(path)[len("schedule_"):-len(".csv")]
    return season, pd.read_csv(path, dtype={"game_id": str})


class StaleHistory(RuntimeError):
    """A game that is over is missing from nba_data/."""


def pending_games(schedule, played_ids):
    """Regular-season games with both teams known that nba_data/ does not have."""
    return schedule[(schedule["season_type"] == "regular_season") & schedule["home"].notna()
                    & ~schedule["game_id"].isin(played_ids)]


def check_history(pending, now):
    """Raise StaleHistory when a pending game is already over."""
    tipped_off = pd.to_datetime(pending["tipoff_utc"], utc=True) < now - OVER_AFTER
    over = pending[(pending["status"] == "final") | ((pending["postponed"] != "Y") & tipped_off)]
    if len(over):
        listed = ", ".join(f"{g.game_id} {g.game_date} {g.away}@{g.home}" for g in over.head(5).itertuples())
        raise StaleHistory(f"{len(over)} mac bitti ama nba_data/'da yok ({listed}) - once fikstur ve "
                           "veri yenilemesi (DURUM.md, gunluk komutlar)")


def next_dates(pending, days):
    """The first `days` dates of the pending games that are not postponed, and those games."""
    upcoming = pending[pending["postponed"] != "Y"]
    return sorted(upcoming["game_date"].unique())[:days], upcoming


def unplayed_frame(upcoming, season, date):
    games = upcoming[upcoming["game_date"] == date]
    return pd.DataFrame({"game_id": games["game_id"].str.zfill(10), "season": season,
                         "game_date": pd.to_datetime(games["game_date"]),
                         "home_team": games["home"], "away_team": games["away"],
                         "home_team_id": games["home_id"].astype("int64"),
                         "away_team_id": games["away_id"].astype("int64")})


# ---------------------------------------------------------------------------
#  3. One pass: played games plus one date's unplayed games
# ---------------------------------------------------------------------------
def build_pass(unplayed, verbose=True):
    """The dataset for the played games and `unplayed`, roster family pre-game."""
    builder = _load_sibling("build_dataset_db")
    pregame = _load_sibling("pregame_roster")
    dataset, base_features, groups, _ = builder.build(verbose=verbose, with_clutch=False,
                                                       source="csv", unplayed=unplayed)
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    listings = pregame.load_listings()
    played = dataset[dataset["is_played"]]
    upcoming = dataset[~dataset["is_played"]]
    if len(upcoming):
        listed = played[played["game_id"].isin(set(listings["game_id"]))]
        sheets = pregame.carried_forward_listings(listings, listed, upcoming)
        listings = pd.concat([listings, sheets], ignore_index=True)
    impacts = pregame.load_impacts()
    def_totals, off_totals = pregame.load_matchup_totals()
    roster = pregame.build_pregame_features(dataset, listings, impacts, def_totals, off_totals, verbose)
    dataset = pregame.replace_roster_features(dataset, roster)
    groups = dict(groups)
    groups["pregame_extra"] = pregame.EXTRA_COLUMNS
    return dataset, base_features, groups


def build(days=1, now=None, verbose=True, refresh=True):
    """Played games plus the next `days` dates of unplayed games; returns the blob that is saved."""
    t0 = time.time()
    if refresh:
        csv_tables.update(verbose=verbose)
    season, schedule = current_schedule()
    pending = pending_games(schedule, set(csv_tables.load("games")["game_id"]))
    check_history(pending, now or pd.Timestamp.now(tz="UTC"))
    if refresh:
        refresh_impact(verbose)
    dates, upcoming = next_dates(pending, days)
    if verbose:
        print(f"  oynanmamis tarihler: {dates or 'yok'}", flush=True)

    passes = dates or [None]                  # with no date left, one pass over the played games
    dataset, base_features, groups = None, None, None
    unplayed_parts = []
    for date in passes:
        frame = unplayed_frame(upcoming, season, date) if date else None
        built, base_features, groups = build_pass(frame, verbose)
        if dataset is None:
            dataset = built
        if frame is not None:
            unplayed_parts.append(built[~built["is_played"]])
    played = dataset[dataset["is_played"]]
    dataset = pd.concat([played] + unplayed_parts, ignore_index=True)
    features = json.load(open(FEATURE_LIST_PATH, encoding="utf-8"))["features"]
    missing = [f for f in features if f not in dataset.columns]
    if missing:
        raise RuntimeError(f"uretim feature'lari veri setinde yok: {missing[:5]}")
    blob = {"dataset": dataset, "features": base_features, "feature_groups": groups,
            "rest_features": groups.get("rest", []), "production_features": features,
            "seasons": sorted(dataset["season"].unique().tolist()), "roster": "pregame",
            "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "unplayed_dates": dates}
    if verbose:
        print(f"  veri seti: {int(dataset['is_played'].sum()):,} oynanmis + "
              f"{int((~dataset['is_played']).sum())} oynanmamis mac ({time.time() - t0:.0f} sn)", flush=True)
    return blob


# ---------------------------------------------------------------------------
#  4. Parity with the production dataset
# ---------------------------------------------------------------------------
def without_late_matchups(played):
    """The played rows with their matchup columns swept again, LATE_MATCHUPS left out."""
    pregame = _load_sibling("pregame_roster")
    def_totals, off_totals = pregame.load_matchup_totals()
    for gid in LATE_MATCHUPS:
        def_totals.pop(gid, None)
        off_totals.pop(gid, None)
    roster = pregame.build_pregame_features(played, pregame.load_listings(), pregame.load_impacts(),
                                            def_totals, off_totals, verbose=False)
    columns = [c for c in roster.columns if "matchup" in c]
    swept = roster.set_index("game_id")[columns].fillna(0.0)    # as replace_roster_features fills
    out = played.set_index("game_id")
    out[columns] = swept.reindex(out.index)
    return out.reset_index()


def parity(blob, production_path=PRODUCTION_PATH):
    """Compare the rebuilt played rows with the production dataset, column by column."""
    production = pd.read_pickle(production_path)["dataset"].copy()
    production["game_id"] = production["game_id"].astype(str).str.zfill(10)
    rebuilt = blob["dataset"]
    rebuilt = rebuilt[rebuilt["is_played"] & rebuilt["game_id"].isin(set(production["game_id"]))]
    if LATE_MATCHUPS:
        rebuilt = without_late_matchups(rebuilt)
    columns = blob["production_features"] + ["home_score", "away_score", "point_diff", "total_score", "home_win"]
    left = production.set_index("game_id")
    right = rebuilt.set_index("game_id").reindex(left.index)
    report = {"games_production": len(left), "games_rebuilt": int(right["season"].notna().sum()),
              "tolerance": PARITY_TOLERANCE, "late_matchups_left_out": LATE_MATCHUPS, "columns": {}}
    for column in columns:
        a = pd.to_numeric(left[column], errors="coerce").astype(float)
        b = pd.to_numeric(right[column], errors="coerce").astype(float)
        gap = (a - b).abs()
        differ = int(((gap > PARITY_TOLERANCE) | (a.isna() != b.isna())).sum())
        if differ:
            worst = gap.idxmax() if gap.notna().any() else None
            report["columns"][column] = {"games_differing": differ, "max_gap": float(gap.max()),
                                         "worst_game": worst}
    for column in ID_COLUMNS[1:]:
        a, b = left[column], right[column]
        if column == "game_date":
            a, b = pd.to_datetime(a).dt.date, pd.to_datetime(b).dt.date
        differ = int((a.astype(str) != b.astype(str)).sum())
        if differ:
            report["columns"][column] = {"games_differing": differ}
    report["equal"] = (report["games_rebuilt"] == report["games_production"]) and not report["columns"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1, help="kac oynanmamis tarih eklensin")
    parser.add_argument("--check", action="store_true", help="uretim veri setiyle esitlik testi")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    blob = build(days=args.days)
    pd.to_pickle(blob, args.out)
    print(f"Yazildi: {args.out}")
    if not args.check:
        return 0
    report = parity(blob)
    with open(PARITY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"  esitlik: {report['games_rebuilt']:,}/{report['games_production']:,} mac, "
          f"farkli sutun {len(report['columns'])} -> {'ESIT' if report['equal'] else 'FARK VAR'}")
    for column, detail in list(report["columns"].items())[:15]:
        print(f"      {column}: {detail}")
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
