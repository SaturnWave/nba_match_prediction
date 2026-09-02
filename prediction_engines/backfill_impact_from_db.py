"""
Fill game_impact_cache_v4.pkl for games the CSV pipeline never covered.

WHY
    The impact cache holds what compute_game_impact produced per game, keyed by
    game_id, and the model's impact features are built from it. It covers the
    8,289 games the CSV pipeline knew about. The database carries 10,749 â€”
    2,460 more, all of 2017-18 and 2018-19 â€” and those games have no cached
    impact, so training on nine seasons would silently feed them NaN.

    Reading play-by-play from the database makes this cheap: 5 games of
    play-by-play arrive in 0.14 seconds and compute_game_impact takes about
    0.01 seconds per game, against roughly 0.7 seconds per game through the
    CSV path.

CONSISTENCY
    Spot-checking five already-cached 2025-26 games, three reproduced exactly
    and two moved by about 0.4 on aggregates near 200 (0.2%). The scoring-run
    term is order-sensitive and the database orders strictly by action_number,
    which the CSV export did not always preserve. Existing entries are left
    alone, so that difference never rewrites history â€” only new games are
    added.

SAFETY
    The cache is written atomically (temp file, then replace). A crash halfway
    through cannot truncate a 6 MB cache that took hours to build.

Run:  py prediction_engines/backfill_impact_from_db.py [--seasons 2017_2018 2018_2019]
      py prediction_engines/backfill_impact_from_db.py --dry-run
"""
import os
import sys
import time
import pickle
import argparse
import importlib.util
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PROJECT_ROOT, "prediction_engines")
CACHE_PATH = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
CHUNK = 200


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="yalnizca bu sezonlar (varsayilan: hepsi)")
    parser.add_argument("--dry-run", action="store_true",
                        help="hesaplama yapma, yalnizca eksikleri raporla")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    db = _load_sibling("db_source")
    predictor = _load_sibling("predict_2025_2026")

    master = db.load_master_frame(seasons=args.seasons)
    cache = read_cache()
    print(f"cache: {len(cache)} kayit")

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
    conn = db.connect()
    done, failed = 0, []
    try:
        for gid, pbp in db.load_pbp(pending, conn=conn, chunk_size=CHUNK):
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
            if done % 250 == 0:
                elapsed = (datetime.datetime.now() - start).total_seconds()
                rate = done / elapsed if elapsed else 0
                eta = (len(pending) - done) / rate if rate else 0
                write_cache_atomic(cache)   # checkpoint
                print(f"  [{done}/{len(pending)}] {elapsed/60:.1f} dk | "
                      f"~{eta/60:.1f} dk kaldi | {len(failed)} hata", flush=True)
    finally:
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
