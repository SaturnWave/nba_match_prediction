"""
Fill game_impact_cache_v4.pkl for every game that has no cached impact yet.

(The name says from_db because that is where it first read. It now reads from
the database OR from the CSVs on disk; the database is no longer required.)

WHY
    The impact cache holds what compute_game_impact produced per game, keyed by
    game_id. The model's impact features are built from it, and so are the
    dashboard's power rankings and roster panel (pregame_roster.load_impacts,
    app.py _build_player_games). A game absent from the cache has no impact at
    all: the model is fed NaN and the dashboard shows no players for it.

    It first existed because the database carried 2,460 games the CSV pipeline
    never had - all of 2017-18 and 2018-19. It is needed again every time the
    season moves on: the cache does not grow by itself, so games fetched after
    it was last written are missing from it, silently.

TWO SOURCES
    db   play-by-play from the phone's MariaDB: five games arrive in 0.14 s and
         compute_game_impact takes about 0.01 s per game.
    csv  play-by-play from nba_data/<season>/<gid>/play_by_play/, read exactly
         as predict_2025_2026.load_game_data reads it - roughly 0.7 s a game,
         which is nothing for one season and hours for nine.
    --source auto (the default) takes the database when it answers a probe
    within five seconds and the CSVs when it does not. The database lives on a
    phone and is not always up; the CSVs always are.

CONSISTENCY
    Spot-checking five already-cached 2025-26 games through the database, three
    reproduced exactly and two moved by about 0.4 on aggregates near 200
    (0.2%). The scoring-run term is order-sensitive and the database orders
    strictly by action_number, which the CSV files do not: a 2025-26 file and a
    2026-27 file were both measured out of order in places (4 and 13 descending
    steps). So the CSV source reads each file in the order it is written, which
    is the order this cache's existing entries were made in. Sorting it to
    match the database would make the new entries disagree with the old ones
    inside the same cache, which is the disagreement that actually matters.

    Existing entries are left alone by either source, so neither rewrites
    history - only missing games are added.

SAFETY
    The cache is written atomically (temp file, then replace) and checkpointed
    every 250 games. A crash halfway through cannot truncate a 10 MB cache that
    took hours to build.

Run:  py prediction_engines/backfill_impact_from_db.py
      py prediction_engines/backfill_impact_from_db.py --seasons 2026_2027
      py prediction_engines/backfill_impact_from_db.py --source csv --dry-run
"""
import argparse
import datetime
import importlib.util
import os
import pickle
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PROJECT_ROOT, "prediction_engines")
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
CACHE_PATH = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
CHUNK = 200
MIN_BYTES = 65          # a file this small is a header and nothing else
CHECKPOINT_EVERY = 250


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cache(path=CACHE_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        return cache if isinstance(cache, dict) else {}
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        print(f"[warn] cache okunamadi ({exc}); bostan baslaniyor", file=sys.stderr)
        return {}


def write_cache_atomic(cache, path=CACHE_PATH):
    """Temp file then replace, so an interrupted run leaves the old cache intact."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def resolve_source(requested, db):
    """'db' or 'csv'. 'auto' asks the database for its greeting banner first.

    A TCP connect to the phone succeeds whether or not MariaDB is behind the
    port - Tailscale accepts it either way - so the probe reads the server's
    unprompted greeting instead of trusting the open port.
    """
    if requested in ("db", "csv"):
        return requested
    alive, detail = db.probe()
    print(f"veritabani: {'acik' if alive else 'kapali'} ({detail})")
    return "db" if alive else "csv"


def csv_pbp(game_ids, seasons):
    """(game_id, play-by-play) per game from the per-game CSVs.

    Read with a plain read_csv and in file order, which is how
    predict_2025_2026.load_game_data read the games already in the cache. A
    game whose file is missing or is only a header yields an empty frame, and
    the caller counts that as a failure rather than a zero-impact game.
    """
    for gid in game_ids:
        season = seasons.get(gid)
        if season is None:
            yield gid, pd.DataFrame()
            continue
        path = os.path.join(DATA_DIR, season, gid, "play_by_play", f"{gid}pbp.csv")
        if not (os.path.isfile(path) and os.path.getsize(path) >= MIN_BYTES):
            yield gid, pd.DataFrame()
            continue
        try:
            yield gid, pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"    {gid} okunamadi: {type(exc).__name__}: {exc}", flush=True)
            yield gid, pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="yalnizca bu sezonlar (varsayilan: hepsi)")
    parser.add_argument("--source", choices=("auto", "db", "csv"), default="auto",
                        help="play-by-play nereden okunacak (varsayilan: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="hesaplama yapma, yalnizca eksikleri raporla")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    db = _load_sibling("db_source")
    predictor = _load_sibling("predict_2025_2026")
    source = resolve_source(args.source, db)

    if source == "db":
        master = db.load_master_frame(seasons=args.seasons)
    else:
        master = _load_sibling("csv_master").load_master_frame(seasons=args.seasons)
    master["game_id"] = master["game_id"].astype(str).str.zfill(10)
    # A game not played yet has no play-by-play and must not be counted as a
    # failure: the CSV schedule lists the fixture before the result exists.
    if "home_score" in master.columns:
        unplayed = int(master["home_score"].isna().sum())
        if unplayed:
            print(f"{unplayed} mac henuz oynanmamis, disarida birakildi")
            master = master[master["home_score"].notna()]
    cache = read_cache()
    print(f"cache: {len(cache)} kayit, kaynak: {source}")

    pending = [g for g in master["game_id"] if not isinstance(cache.get(g), dict)]
    if args.limit:
        pending = pending[:args.limit]
    by_season = master[master["game_id"].isin(pending)].groupby("season").size()
    print(f"eksik: {len(pending)} mac")
    if not by_season.empty:
        print(by_season.to_string())
    if args.dry_run or not pending:
        print("yapilacak is yok" if not pending else "dry-run, cikiliyor")
        return 0

    meta = master.set_index("game_id")[
        ["home_team", "away_team", "home_team_id", "away_team_id"]].to_dict("index")

    start = datetime.datetime.now()
    conn = db.connect() if source == "db" else None
    games = (db.load_pbp(pending, conn=conn, chunk_size=CHUNK) if source == "db"
             else csv_pbp(pending, dict(zip(master["game_id"], master["season"]))))
    done, failed = 0, []
    try:
        for gid, pbp in games:
            info = meta.get(gid)
            if info is None or pbp.empty:
                failed.append(gid)
                continue
            try:
                cache[gid] = predictor.compute_game_impact(
                    pbp, info["home_team"], info["away_team"],
                    int(info["home_team_id"]), int(info["away_team_id"]), {})
            except (KeyError, ValueError, TypeError) as exc:
                # A malformed play-by-play frame should cost one game, not the run.
                print(f"    {gid} atlandi: {type(exc).__name__}: {exc}", flush=True)
                failed.append(gid)
                continue
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                elapsed = (datetime.datetime.now() - start).total_seconds()
                rate = done / elapsed if elapsed else 0
                eta = (len(pending) - done) / rate if rate else 0
                write_cache_atomic(cache)   # checkpoint
                print(f"  [{done}/{len(pending)}] {elapsed/60:.1f} dk | "
                      f"~{eta/60:.1f} dk kaldi | {len(failed)} hata", flush=True)
    finally:
        if conn is not None:
            conn.close()
        write_cache_atomic(cache)

    elapsed = (datetime.datetime.now() - start).total_seconds()
    print(f"\nbitti: {done} mac islendi, {len(failed)} hata, {elapsed/60:.1f} dk")
    print(f"cache artik {len(cache)} kayit")
    if failed:
        print("hatali game_id'ler: " + ", ".join(failed[:20])
              + (" ..." if len(failed) > 20 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
