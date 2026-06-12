"""
Focused, resumable retrieval of the 2025-26 NBA season game data needed by the
prediction engine (prediction_engines/2023_2024.py -> GameDataLoader).

IMPORTANT: As of the 2025-26 season the NBA stopped publishing data on the V2
box-score endpoints (BoxScoreTraditionalV2 / BoxScoreAdvancedV2 return empty).
We therefore use the V3 endpoints and RENAME their camelCase columns back to the
UPPER_SNAKE schema used by every earlier season on disk, so the downstream
pipeline (which expects FGM, FGA, TEAM_ABBREVIATION, PACE, OFF_RATING, ...)
works unchanged.

Fetched per game:
  - play-by-play           PlayByPlayV3
  - traditional box (V3)   player + team -> box_score_traditional[/ _team].csv
  - advanced box   (V3)    player + team -> box_score_advanced[/ _team].csv

Already-downloaded games are skipped (resumable). Run with an integer arg to cap
the number of games (chronological), e.g. `py retrieve_2025_2026_data.py 50`.
"""
import os
import sys
import time
import datetime
import traceback
import pandas as pd
from nba_api.stats.endpoints import (
    PlayByPlayV3,
    BoxScoreTraditionalV3,
    BoxScoreAdvancedV3,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")

SEASON_FMT = "2025_2026"
API_DELAY = 0.6
RETRY_DELAYS = [1, 2, 4, 8, 12]
MAX_RETRIES = 5
TIMEOUT = 30

# --- V3 (camelCase) -> V2 (UPPER_SNAKE) column maps -------------------------
TRAD_MAP = {
    "gameId": "GAME_ID", "teamId": "TEAM_ID", "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY", "teamName": "TEAM_NAME", "personId": "PLAYER_ID",
    "nameI": "NICKNAME", "position": "START_POSITION", "comment": "COMMENT",
    "minutes": "MIN", "fieldGoalsMade": "FGM", "fieldGoalsAttempted": "FGA",
    "fieldGoalsPercentage": "FG_PCT", "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A", "threePointersPercentage": "FG3_PCT",
    "freeThrowsMade": "FTM", "freeThrowsAttempted": "FTA", "freeThrowsPercentage": "FT_PCT",
    "reboundsOffensive": "OREB", "reboundsDefensive": "DREB", "reboundsTotal": "REB",
    "assists": "AST", "steals": "STL", "blocks": "BLK", "turnovers": "TO",
    "foulsPersonal": "PF", "points": "PTS", "plusMinusPoints": "PLUS_MINUS",
}
ADV_MAP = {
    "gameId": "GAME_ID", "teamId": "TEAM_ID", "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY", "teamName": "TEAM_NAME", "personId": "PLAYER_ID",
    "nameI": "NICKNAME", "minutes": "MIN",
    "estimatedOffensiveRating": "E_OFF_RATING", "offensiveRating": "OFF_RATING",
    "estimatedDefensiveRating": "E_DEF_RATING", "defensiveRating": "DEF_RATING",
    "estimatedNetRating": "E_NET_RATING", "netRating": "NET_RATING",
    "assistPercentage": "AST_PCT", "assistToTurnover": "AST_TOV", "assistRatio": "AST_RATIO",
    "offensiveReboundPercentage": "OREB_PCT", "defensiveReboundPercentage": "DREB_PCT",
    "reboundPercentage": "REB_PCT", "estimatedTeamTurnoverPercentage": "E_TM_TOV_PCT",
    "turnoverRatio": "TM_TOV_PCT", "effectiveFieldGoalPercentage": "EFG_PCT",
    "trueShootingPercentage": "TS_PCT", "usagePercentage": "USG_PCT",
    "estimatedUsagePercentage": "E_USG_PCT", "estimatedPace": "E_PACE", "pace": "PACE",
    "pacePer40": "PACE_PER40", "possessions": "POSS", "PIE": "PIE",
}


def fmt_gid(game_id):
    g = str(game_id)
    return g if g.startswith("00") else g.zfill(10)


def _remap(df, col_map, is_player):
    """Rename V3 columns to V2 schema; add PLAYER_NAME for player frames."""
    out = df.rename(columns=col_map).copy()
    if is_player and "firstName" in df.columns and "familyName" in df.columns:
        out["PLAYER_NAME"] = (df["firstName"].fillna("").astype(str) + " "
                              + df["familyName"].fillna("").astype(str)).str.strip()
    return out


def _team_frame(frames):
    """Pick the 2-row team-totals frame (no personId)."""
    for fr in frames:
        if len(fr) == 2 and "personId" not in fr.columns:
            return fr
    return None


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


def game_dirs(gid):
    gdir = os.path.join(BASE_DIR, SEASON_FMT, gid)
    return gdir, os.path.join(gdir, "play_by_play"), os.path.join(gdir, "box_scores")


def already_complete(gid):
    _, pbp_dir, box_dir = game_dirs(gid)
    paths = [
        os.path.join(pbp_dir, f"{gid}pbp.csv"),
        os.path.join(box_dir, f"{gid}box_score_traditional.csv"),
        os.path.join(box_dir, f"{gid}box_score_traditional_team.csv"),
        os.path.join(box_dir, f"{gid}box_score_advanced.csv"),
        os.path.join(box_dir, f"{gid}box_score_advanced_team.csv"),
    ]
    # require existence AND non-empty (guards against earlier empty-V2 writes)
    return all(os.path.exists(p) and os.path.getsize(p) > 64 for p in paths)


def process_game(gid):
    _, pbp_dir, box_dir = game_dirs(gid)
    os.makedirs(pbp_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    ok = True

    pbp_path = os.path.join(pbp_dir, f"{gid}pbp.csv")
    if not (os.path.exists(pbp_path) and os.path.getsize(pbp_path) > 64):
        frames = _retry(lambda: PlayByPlayV3(game_id=gid, start_period=1, end_period=10,
                                             timeout=TIMEOUT).get_data_frames(), "PBP", gid)
        if frames and len(frames):
            frames[0].to_csv(pbp_path, index=False)
        else:
            ok = False
        time.sleep(API_DELAY)

    trad_path = os.path.join(box_dir, f"{gid}box_score_traditional.csv")
    if not (os.path.exists(trad_path) and os.path.getsize(trad_path) > 64):
        frames = _retry(lambda: BoxScoreTraditionalV3(game_id=gid, timeout=TIMEOUT).get_data_frames(),
                        "TraditionalV3", gid)
        if frames:
            _remap(frames[0], TRAD_MAP, True).to_csv(trad_path, index=False)
            team = _team_frame(frames)
            if team is not None:
                _remap(team, TRAD_MAP, False).to_csv(
                    os.path.join(box_dir, f"{gid}box_score_traditional_team.csv"), index=False)
            else:
                ok = False
        else:
            ok = False
        time.sleep(API_DELAY)

    adv_path = os.path.join(box_dir, f"{gid}box_score_advanced.csv")
    if not (os.path.exists(adv_path) and os.path.getsize(adv_path) > 64):
        frames = _retry(lambda: BoxScoreAdvancedV3(game_id=gid, timeout=TIMEOUT).get_data_frames(),
                        "AdvancedV3", gid)
        if frames:
            _remap(frames[0], ADV_MAP, True).to_csv(adv_path, index=False)
            team = _team_frame(frames)
            if team is not None:
                _remap(team, ADV_MAP, False).to_csv(
                    os.path.join(box_dir, f"{gid}box_score_advanced_team.csv"), index=False)
            else:
                ok = False
        else:
            ok = False
        time.sleep(API_DELAY)

    return ok


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    csv_path = os.path.join(GAME_IDS_DIR, f"game_id_{SEASON_FMT}.csv")
    games = pd.read_csv(csv_path).drop_duplicates(subset=["GAME_ID"]).sort_values("GAME_DATE")
    gids = [fmt_gid(g) for g in games["GAME_ID"].tolist()]
    if limit:
        gids = gids[:limit]

    pending = [g for g in gids if not already_complete(g)]
    start = datetime.datetime.now()
    print(f"[{start}] 2025-26 retrieval: {len(gids)} total, {len(pending)} pending, "
          f"{len(gids) - len(pending)} complete", flush=True)

    failures = 0
    for i, gid in enumerate(pending, 1):
        if not process_game(gid):
            failures += 1
        if i % 10 == 0 or i == len(pending):
            elapsed = (datetime.datetime.now() - start).total_seconds()
            rate = i / elapsed if elapsed else 0
            eta = (len(pending) - i) / rate if rate else 0
            print(f"  [{i}/{len(pending)}] {gid} | {elapsed/60:.1f} min | ~{eta/60:.1f} min ETA "
                  f"| {failures} failures", flush=True)

    print(f"[{datetime.datetime.now()}] DONE. games this run: {len(pending)}, failures: {failures}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Fatal error: {e}")
        traceback.print_exc()
