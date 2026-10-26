"""Which 2026-27 games already have tracking-based box scores on stats.nba.com? Read-only probe."""
import datetime
import time
from nba_api.stats.endpoints import (boxscoredefensivev2, boxscorehustlev2, boxscorematchupsv3,
                                     boxscoreplayertrackv3, leaguegamelog)

print("probe time UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"), flush=True)
log = leaguegamelog.LeagueGameLog(season="2026-27", season_type_all_star="Regular Season",
                                  player_or_team_abbreviation="T", timeout=90).get_data_frames()[0]
games = log.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE"]].sort_values("GAME_ID")
sample = list(games.head(2).itertuples(index=False)) + list(games.tail(6).itertuples(index=False))
ENDPOINTS = {"PlayerTrackV3": boxscoreplayertrackv3.BoxScorePlayerTrackV3,
             "MatchupsV3": boxscorematchupsv3.BoxScoreMatchupsV3,
             "DefensiveV2": boxscoredefensivev2.BoxScoreDefensiveV2,
             "HustleV2": boxscorehustlev2.BoxScoreHustleV2}
for game_id, game_date in sample:
    row = {}
    for name, endpoint in ENDPOINTS.items():
        try:
            frames = endpoint(game_id=game_id, timeout=60).get_data_frames()
            row[name] = len(frames[0]) if frames else 0
        except Exception as error:
            row[name] = type(error).__name__
        time.sleep(1.2)
    print(game_id, game_date, row, flush=True)
