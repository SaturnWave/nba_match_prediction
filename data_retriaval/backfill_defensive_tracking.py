"""
Resumable backfill of the two data types the 2025-26 retrieval never fetched:
defensive box scores and player tracking.

Why this exists: retrieve_2025_2026_data.py only pulls play-by-play + traditional
+ advanced. But ImpactScoreCalculator (prediction_engines/2023_2024.py) also
consumes `box_defensive` and `player_tracking`, and every season from 2019-20
through 2024-25 has both on disk. Without them the 2025-26 impact scores are
computed from a strictly smaller input set than the training seasons, which
silently shifts the feature distribution the models are trained on.

Schema notes (verified against the on-disk files of earlier seasons):
  - defensive: BoxScoreDefensiveV2 still serves 2025-26 and its camelCase
    columns are EXACTLY what earlier seasons store. Written through unchanged.
  - tracking:  BoxScorePlayerTrackV2 no longer exists in nba_api >= 1.11, and V3
    returns camelCase. Earlier seasons store UPPER_SNAKE, so V3 is renamed back
    to that schema (same trick retrieve_2025_2026_data.py uses for trad/adv).

Already-downloaded games are skipped. Run:
    py data_retriaval/backfill_defensive_tracking.py [SEASON] [LIMIT]
e.g. py data_retriaval/backfill_defensive_tracking.py 2025_2026
"""
import os
import sys
import time
import datetime
import traceback
import pandas as pd
from nba_api.stats.endpoints import BoxScoreDefensiveV2, BoxScorePlayerTrackV3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")

API_DELAY = 0.6
RETRY_DELAYS = [1, 2, 4, 8, 12]
MAX_RETRIES = 5
TIMEOUT = 30

# --- PlayerTrackV3 (camelCase) -> on-disk V2 schema (UPPER_SNAKE) ------------
TRACK_MAP = {
    "gameId": "GAME_ID", "teamId": "TEAM_ID", "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY", "teamName": "TEAM_NAME", "personId": "PLAYER_ID",
    "position": "START_POSITION", "comment": "COMMENT", "minutes": "MIN",
    "speed": "SPD", "distance": "DIST",
    "reboundChancesOffensive": "ORBC", "reboundChancesDefensive": "DRBC",
    "reboundChancesTotal": "RBC", "touches": "TCHS",
    "secondaryAssists": "SAST", "freeThrowAssists": "FTAST",
    "passes": "PASS", "assists": "AST",
    "contestedFieldGoalsMade": "CFGM", "contestedFieldGoalsAttempted": "CFGA",
    "contestedFieldGoalPercentage": "CFG_PCT",
    "uncontestedFieldGoalsMade": "UFGM", "uncontestedFieldGoalsAttempted": "UFGA",
    "uncontestedFieldGoalsPercentage": "UFG_PCT", "fieldGoalPercentage": "FG_PCT",
    "defendedAtRimFieldGoalsMade": "DFGM",
    "defendedAtRimFieldGoalsAttempted": "DFGA",
    "defendedAtRimFieldGoalPercentage": "DFG_PCT",
}

# Exact column order of the earlier seasons' files, so downstream readers that
# index positionally or diff headers see no change.
PLAYER_TRACK_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_ID",
    "PLAYER_NAME", "START_POSITION", "COMMENT", "MIN", "SPD", "DIST",
    "ORBC", "DRBC", "RBC", "TCHS", "SAST", "FTAST", "PASS", "AST",
    "CFGM", "CFGA", "CFG_PCT", "UFGM", "UFGA", "UFG_PCT", "FG_PCT",
    "DFGM", "DFGA", "DFG_PCT", "GAME_ID_CLEAN",
]
TEAM_TRACK_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "TEAM_CITY",
    "MIN", "DIST", "ORBC", "DRBC", "RBC", "TCHS", "SAST", "FTAST", "PASS",
    "AST", "CFGM", "CFGA", "CFG_PCT", "UFGM", "UFGA", "UFG_PCT", "FG_PCT",
    "DFGM", "DFGA", "DFG_PCT", "GAME_ID_CLEAN",
]


def fmt_gid(game_id):
    g = str(game_id)
    return g if g.startswith("00") else g.zfill(10)


def _retry(callable_, label, gid):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return callable_()
        except Exception as e:  # noqa: BLE001 - nba_api raises bare Exception on HTTP/JSON faults
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"    retry {attempt + 1}/{MAX_RETRIES} {label} {gid} ({e}); wait {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"    FAILED {label} {gid}: {e}", flush=True)
                return None


def _track_frame(df, gid, columns, is_player):
    """Rename PlayerTrackV3 columns to the on-disk V2 schema and fix column order."""
    out = df.rename(columns=TRACK_MAP).copy()
    if is_player:
        out["PLAYER_NAME"] = (df["firstName"].fillna("").astype(str) + " "
                              + df["familyName"].fillna("").astype(str)).str.strip()
    out["GAME_ID_CLEAN"] = gid
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 64


def game_paths(season, gid):
    gdir = os.path.join(BASE_DIR, season, gid)
    box_dir = os.path.join(gdir, "box_scores")
    track_dir = os.path.join(gdir, "player_tracking")
    return {
        "box_dir": box_dir,
        "track_dir": track_dir,
        "def": os.path.join(box_dir, f"{gid}box_score_defensive.csv"),
        "def_team": os.path.join(box_dir, f"{gid}box_score_defensive_team.csv"),
        "track": os.path.join(track_dir, f"{gid}player_tracking.csv"),
        "track_team": os.path.join(track_dir, f"{gid}team_tracking.csv"),
    }


def already_complete(season, gid):
    p = game_paths(season, gid)
    return all(_exists(p[k]) for k in ("def", "def_team", "track", "track_team"))


def process_game(season, gid):
    p = game_paths(season, gid)
    os.makedirs(p["box_dir"], exist_ok=True)
    os.makedirs(p["track_dir"], exist_ok=True)
    ok = True

    if not (_exists(p["def"]) and _exists(p["def_team"])):
        frames = _retry(lambda: BoxScoreDefensiveV2(game_id=gid, timeout=TIMEOUT).get_data_frames(),
                        "DefensiveV2", gid)
        if frames and len(frames) >= 2 and not frames[0].empty:
            # V2 already emits the camelCase schema stored for earlier seasons.
            frames[0].to_csv(p["def"], index=False)
            frames[1].to_csv(p["def_team"], index=False)
        else:
            ok = False
        time.sleep(API_DELAY)

    if not (_exists(p["track"]) and _exists(p["track_team"])):
        frames = _retry(lambda: BoxScorePlayerTrackV3(game_id=gid, timeout=TIMEOUT).get_data_frames(),
                        "PlayerTrackV3", gid)
        if frames and len(frames) >= 2 and not frames[0].empty:
            _track_frame(frames[0], gid, PLAYER_TRACK_COLUMNS, True).to_csv(p["track"], index=False)
            _track_frame(frames[1], gid, TEAM_TRACK_COLUMNS, False).to_csv(p["track_team"], index=False)
        else:
            ok = False
        time.sleep(API_DELAY)

    return ok


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2025_2026"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    csv_path = os.path.join(GAME_IDS_DIR, f"game_id_{season}.csv")
    games = pd.read_csv(csv_path).drop_duplicates(subset=["GAME_ID"]).sort_values("GAME_DATE")
    gids = [fmt_gid(g) for g in games["GAME_ID"].tolist()]
    if limit:
        gids = gids[:limit]

    pending = [g for g in gids if not already_complete(season, g)]
    start = datetime.datetime.now()
    print(f"[{start}] {season} defensive+tracking backfill: {len(gids)} total, "
          f"{len(pending)} pending, {len(gids) - len(pending)} complete", flush=True)

    failures = []
    for i, gid in enumerate(pending, 1):
        if not process_game(season, gid):
            failures.append(gid)
        if i % 25 == 0 or i == len(pending):
            elapsed = (datetime.datetime.now() - start).total_seconds()
            rate = i / elapsed if elapsed else 0
            eta = (len(pending) - i) / rate if rate else 0
            print(f"  [{i}/{len(pending)}] {gid} | {elapsed/60:.1f} min | ~{eta/60:.1f} min ETA "
                  f"| {len(failures)} failures", flush=True)

    print(f"[{datetime.datetime.now()}] DONE. games this run: {len(pending)}, "
          f"failures: {len(failures)}", flush=True)
    if failures:
        # Surfaced, not swallowed: these games stay incomplete and a re-run retries them.
        print("failed game ids: " + ", ".join(failures), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - top-level guard so a partial run still reports
        print(f"Fatal error: {e}")
        traceback.print_exc()
