"""
Build the engineered dataset from MariaDB instead of the CSV tree.

WHAT THIS REUSES AND WHAT IT REPLACES
    Only the SOURCE of the per-game rows changes. Roster-impact features, the
    matchup module, and FeatureEngineer are the same objects the CSV pipeline
    uses, called in the same order, so the two datasets are comparable feature
    for feature. predict_2025_2026.py is not modified at all â€” that keeps the
    csv-pipeline and db-pipeline branches cleanly separable, and means a bug
    found here cannot be a bug introduced into the CSV path.

    Replaced: GameDataLoader's per-game CSV reads, which take ~100 minutes for
    10,749 games. db_source.load_master_frame returns the same rows in 1.7
    seconds.

    Rebuilt rather than reused: NBAPredictor._player_records. The CSV path
    fills it as a side effect of walking every game; here it is reconstructed
    from the impact cache, which holds the same per-player values keyed by
    game_id. _build_player_history and _add_roster_features then run untouched.

WHAT IS NEW IN THE OUTPUT
    Five columns the CSV path never produced: home_rest, away_rest, rest_diff,
    home_b2b, away_b2b. _select_features works from a prefix whitelist that
    does not know about them, so they are returned as a SEPARATE list rather
    than smuggled into the base feature set â€” a caller that wants them has to
    ask, and an arm that does not want them is unaffected.

Run:  py prediction_engines/build_dataset_db.py [--seasons 2019_2020 ...]
Output: output/engineered_dataset_db.pkl   {dataset, features, rest_features, seasons}
"""
import os
import sys
import time
import pickle
import argparse
import importlib.util

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PROJECT_ROOT, "prediction_engines")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v5.pkl")


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _player_impact_history(master, cache_path=IMPACT_CACHE):
    """(person_id, game_date, impact) for every player-game in the cache.

    The availability features need to value players who did NOT appear in a
    game, so the history has to be addressable by date rather than by game.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    dates = master.set_index("game_id")["game_date"].to_dict()
    rows = []
    for gid, entry in cache.items():
        gdate = dates.get(gid)
        if gdate is None or not isinstance(entry, dict):
            continue
        for value in (entry.get("players") or {}).values():
            if isinstance(value, dict) and value.get("person_id") is not None:
                rows.append((int(value["person_id"]), gdate,
                             float(value.get("impact", 0.0))))
    return pd.DataFrame(rows, columns=["person_id", "game_date", "impact"])


def player_records_from_cache(master, cache_path=IMPACT_CACHE):
    """Rebuild NBAPredictor._player_records from the impact cache.

    One record per (player, game) carrying the player's impact and the team
    they played for, which is what _build_player_history needs to compute
    trailing form. A player whose team is missing or does not match either side
    of the game is dropped: the trailing windows are per-player, but the roster
    aggregation joins on (game_id, team), so a stray tricode would attach a
    player to a team that did not field them.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    meta = master[["game_id", "game_date", "season", "home_team", "away_team"]]
    records = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        entry = cache.get(gid)
        if not isinstance(entry, dict):
            continue
        for key, value in (entry.get("players") or {}).items():
            if not isinstance(value, dict):
                continue
            team = value.get("team")
            if team not in (home, away):
                continue
            # Identity is the person id, not the surname. _build_player_history
            # groups by this column to build trailing form, and surnames are not
            # unique - two players sharing one would have their histories merged.
            person_id = value.get("person_id")
            records.append({"player": person_id if person_id is not None else str(key),
                            "player_name": value.get("name") or str(key),
                            "team": team, "game_id": gid,
                            "game_date": gdate, "season": season,
                            "impact": float(value.get("impact", 0.0))})
    return records


def _load_master(source, seasons, verbose):
    """The one game-level frame everything downstream is built from.

    Two sources produce it identically - verified column by column over all
    10,749 games, 79 of 80 matching exactly and the 80th, rest_diff, differing
    only where the database's stored value contradicted its own home_rest and
    away_rest on 16 season-opening rows.

    "auto" prefers the database and falls back to CSV, because the database is
    a few seconds against a few minutes when it is up - and it was not up on
    2026-09-03, when the phone hosting it died. A pipeline that stops because
    its cache is offline is a pipeline with a single point of failure it did
    not need.
    """
    if source in ("auto", "db"):
        try:
            db = _load_sibling("db_source")
            reachable, detail = db.probe()
            if reachable:
                return db.load_master_frame(seasons=seasons, verbose=verbose)
            if source == "db":
                raise RuntimeError(f"veritabani cevap vermiyor: {detail}")
            if verbose:
                print(f"  veritabani yok ({detail}) - CSV'ye dusuluyor")
        except RuntimeError:
            raise
        except Exception as exc:                      # noqa: BLE001
            if source == "db":
                raise
            if verbose:
                print(f"  veritabani kullanilamadi ({exc}) - CSV'ye dusuluyor")

    return _load_sibling("csv_master").load_master_frame(
        seasons=seasons, verbose=verbose)


LABEL_COLUMNS = ["home_score", "away_score", "point_diff", "total_score", "home_win"]


def _require_results(master):
    """A played game without a final score stops the build.

    The CSV master derives home_win from point_diff > 0, which turns a missing
    score into a home loss, and the builder then fills the gaps with zeros - a
    0-0 game - without a word. A game the list calls played has a score.
    """
    missing = master.loc[master["home_score"].isna() | master["away_score"].isna(), "game_id"]
    if len(missing):
        raise RuntimeError(f"{len(missing)} oynanmis macin skoru yok (ornek {list(missing[:5])}); "
                           f"kutu skorlari eksik mi?")


def _append_unplayed(master, unplayed):
    """Games not played yet, after the played ones: no result, and `is_played` False.

    `unplayed` gives game_id, season, game_date, home_team, away_team,
    home_team_id and away_team_id. Everything a game's result would supply is
    missing. The rest columns are recomputed over the combined schedule,
    because an upcoming game's rest is the days since that team's last game.
    """
    csv_master = _load_sibling("csv_master")
    played_ids = set(master["game_id"])
    overlap = sorted(set(unplayed["game_id"]) & played_ids)
    if overlap:
        raise RuntimeError(f"oynanmamis denilen maclar oynanmis listesinde: {overlap[:5]}")
    upcoming = unplayed.assign(game_date=pd.to_datetime(unplayed["game_date"]), is_played=False)
    combined = pd.concat([master.assign(is_played=True), upcoming], ignore_index=True)
    combined = combined.drop(columns=[c for c in csv_master.REST_COLUMNS if c in combined.columns])
    combined = csv_master.add_rest_columns(combined)
    return combined.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build(seasons=None, verbose=True, with_clutch=True, source="auto", unplayed=None):
    """Return (dataset, base_features, extra_feature_groups, predictor).

    extra_feature_groups is a dict of {name: [columns]} for feature families
    that _select_features cannot see, because it works from a prefix whitelist
    that predates them. Keeping them grouped and separate is what lets an arm
    ask for one family without dragging in the others.

    `unplayed` (see _append_unplayed) adds games not played yet. FeatureEngineer
    gives them features from the games before them, as it does every game, and
    their labels come out missing, never zero. The dataset says which rows are
    which in `is_played`.
    """
    predictor_module = _load_sibling("predict_2025_2026")

    t0 = time.time()
    master = _load_master(source, seasons, verbose)
    if master.empty:
        raise RuntimeError("veritabanindan hic mac gelmedi")
    _require_results(master)
    if unplayed is not None and len(unplayed):
        master = _append_unplayed(master, unplayed)
    else:
        master = master.assign(is_played=True)
    master = _load_sibling("db_source").attach_impact(master, cache_path=IMPACT_CACHE)
    if verbose:
        print(f"  master: {master.shape} ({time.time() - t0:.1f} sn)")

    predictor = predictor_module.NBAPredictor()
    predictor._player_records = player_records_from_cache(master)
    if verbose:
        print(f"  oyuncu-mac kaydi: {len(predictor._player_records):,}")

    master = predictor._add_roster_features(master)

    add_matchup_features = getattr(predictor_module, "add_matchup_features", None)
    if add_matchup_features is not None:
        try:
            master = add_matchup_features(master, predictor_module.BASE_DATA_DIR)
        except Exception as exc:  # noqa: BLE001 - matchups are optional, never fatal
            print(f"  [warn] matchup feature'lari atlandi: {exc}")

    # Clutch and availability read the play-by-play and the per-player comment
    # field straight from the database; neither has a CSV path yet. When the
    # database is down they are skipped rather than fatal, which costs nothing
    # measurable - both families were screened at high power and neither beat
    # the base feature set (clutch and availability each landed inside their
    # own error bars over 315 seed x fold cells).
    clutch_cols, avail_cols = [], []
    db = _load_sibling("db_source")
    reachable, detail = db.probe() if with_clutch else (False, "istenmedi")
    if with_clutch and not reachable:
        print(f"  [warn] clutch/availability atlandi - veritabani yok ({detail})")
    if with_clutch and reachable:
        clutch_module = _load_sibling("clutch_features")
        avail_module = _load_sibling("availability_features")
        conn = db.connect()
        try:
            master = clutch_module.add_clutch_features(master, conn, verbose=verbose)
            clutch_cols = [c for c in clutch_module.CLUTCH_FEATURE_COLS
                           if c in master.columns]
        except Exception as exc:  # noqa: BLE001 - additive, never fatal
            print(f"  [warn] clutch feature'lari atlandi: {exc}")
        try:
            impact_history = _player_impact_history(master)
            master = avail_module.add_availability_features(
                master, conn, impact_history, verbose=verbose)
            avail_cols = [c for c in avail_module.AVAIL_FEATURE_COLS
                          if c in master.columns]
        except Exception as exc:  # noqa: BLE001 - additive, never fatal
            print(f"  [warn] availability feature'lari atlandi: {exc}")
        finally:
            conn.close()

    dataset = predictor.fe.engineer(master.copy()).fillna(0)
    unplayed_rows = ~dataset["is_played"].astype(bool)
    dataset.loc[unplayed_rows, LABEL_COLUMNS] = np.nan
    predictor.master_df = master
    predictor.dataset = dataset

    base_features = predictor._select_features()
    groups = {
        "rest": [c for c in db.REST_COLUMNS if c in dataset.columns],
        "clutch": [c for c in clutch_cols if c in dataset.columns],
        "avail": [c for c in avail_cols if c in dataset.columns],
    }
    if verbose:
        matchup = [f for f in base_features if "matchup_" in f]
        print(f"  dataset: {dataset.shape[0]} mac x {dataset.shape[1]} sutun")
        print(f"  temel feature: {len(base_features)} ({len(matchup)} matchup)")
        for name, cols in groups.items():
            print(f"  ek '{name}' feature: {len(cols)}")
        print(f"  toplam sure: {time.time() - t0:.1f} sn")
    return dataset, base_features, groups, predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--out", default=DATASET_PATH)
    parser.add_argument("--source", choices=["auto", "db", "csv"], default="auto",
                        help="master frame nereden gelsin (varsayilan: db varsa db)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset, features, groups, _ = build(args.seasons, source=args.source)

    pd.to_pickle({"dataset": dataset, "features": features,
                  "feature_groups": groups,
                  "rest_features": groups.get("rest", []),   # eski okuyucular icin
                  "seasons": sorted(dataset["season"].unique().tolist())},
                 args.out)
    print(f"\nYazildi: {args.out}")
    print(dataset.groupby("season").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
