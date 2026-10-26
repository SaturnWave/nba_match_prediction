"""Tracking box score availability by game date (one or two games per date). Read-only probe."""
import datetime, time
from nba_api.stats.endpoints import boxscoredefensivev2, boxscorematchupsv3, boxscoreplayertrackv3, leaguegamelog
print("probe time UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"), flush=True)
log = leaguegamelog.LeagueGameLog(season="2026-27", season_type_all_star="Regular Season",
                                  player_or_team_abbreviation="T", timeout=90).get_data_frames()[0]
games = log.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE"]]
print("games per date:", games.groupby("GAME_DATE").size().to_dict(), flush=True)
for game_date in ("2026-10-21", "2026-10-22", "2026-10-23"):
    for game_id in games[games.GAME_DATE == game_date].GAME_ID.head(2):
        row = {}
        for name, ep in (("PlayerTrackV3", boxscoreplayertrackv3.BoxScorePlayerTrackV3),
                         ("MatchupsV3", boxscorematchupsv3.BoxScoreMatchupsV3),
                         ("DefensiveV2", boxscoredefensivev2.BoxScoreDefensiveV2)):
            try:
                row[name] = len(ep(game_id=game_id, timeout=60).get_data_frames()[0])
            except Exception as error:
                row[name] = type(error).__name__
            time.sleep(1.2)
        print(game_id, game_date, row, flush=True)
