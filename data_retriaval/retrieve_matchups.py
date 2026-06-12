"""
Resumable retrieval of BoxScoreMatchupsV3 (defender-vs-offensive-player partials)
for whole seasons. Verified working for 2025-26 (V3 endpoint; the V2 box-score
endpoints are dead for 2025-26).

Each game gets one CSV of ~150 matchup rows (offPlayer x defPlayer):
  nba_data/<season>/<gid>/box_scores/<gid>box_score_matchups.csv

Columns are kept in the V3 camelCase schema (there is no legacy V2 matchup file
on disk to stay compatible with): personIdOff/personIdDef, partialPossessions,
playerPoints, matchupFieldGoals*, matchupTurnovers, matchupBlocks, helpBlocks,
shootingFouls, matchupMinutes, percentageDefenderTotalTime, ...

Usage:
  py data_retriaval/retrieve_matchups.py 2025_2026
  py data_retriaval/retrieve_matchups.py 2024_2025 100   # cap to first 100 games
"""
import os
import sys
import time
import datetime
import traceback
import pandas as pd
from nba_api.stats.endpoints import boxscorematchupsv3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")

API_DELAY = 0.6
RETRY_DELAYS = [1, 2, 4, 8, 12]
MAX_RETRIES = 4
TIMEOUT = 30


def fmt_gid(game_id):
    g = str(game_id)
    return g if g.startswith("00") else g.zfill(10)


def matchup_path(season, gid):
    return os.path.join(BASE_DIR, season, gid, "box_scores", f"{gid}box_score_matchups.csv")


def _retry(callable_, label, gid):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return callable_()
        except Exception as e:  # noqa: BLE001
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"    retry {attempt + 1}/{MAX_RETRIES} {label} {gid} ({e}); wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"    FAILED {label} {gid}: {e}", flush=True)
                return None


def process_game(season, gid):
    path = matchup_path(season, gid)
    if os.path.exists(path) and os.path.getsize(path) > 64:
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = _retry(lambda: boxscorematchupsv3.BoxScoreMatchupsV3(
        game_id=gid, timeout=TIMEOUT).get_data_frames(), "MatchupsV3", gid)
    time.sleep(API_DELAY)
    if frames and len(frames) and not frames[0].empty:
        frames[0].to_csv(path, index=False)
        return True
    return False


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2025_2026"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    csv_path = os.path.join(GAME_IDS_DIR, f"game_id_{season}.csv")
    games = pd.read_csv(csv_path).drop_duplicates(subset=["GAME_ID"]).sort_values("GAME_DATE")
    gids = [fmt_gid(g) for g in games["GAME_ID"].tolist()]
    if limit:
        gids = gids[:limit]

    pending = [g for g in gids
               if not (os.path.exists(matchup_path(season, g))
                       and os.path.getsize(matchup_path(season, g)) > 64)]
    start = datetime.datetime.now()
    print(f"[{start}] {season} matchup retrieval: {len(gids)} total, {len(pending)} pending", flush=True)

    failures = 0
    for i, gid in enumerate(pending, 1):
        if not process_game(season, gid):
            failures += 1
        if i % 25 == 0 or i == len(pending):
            elapsed = (datetime.datetime.now() - start).total_seconds()
            rate = i / elapsed if elapsed else 0
            eta = (len(pending) - i) / rate if rate else 0
            print(f"  [{i}/{len(pending)}] {gid} | {elapsed/60:.1f} min | ~{eta/60:.1f} min ETA "
                  f"| {failures} failures", flush=True)

    print(f"[{datetime.datetime.now()}] DONE {season}. processed: {len(pending)}, failures: {failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Fatal error: {e}")
        traceback.print_exc()
