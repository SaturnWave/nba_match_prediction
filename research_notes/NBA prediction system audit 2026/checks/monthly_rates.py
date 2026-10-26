"""2025-26 and 2024-25 league rates by month (NBA.com Month param: 1 = October)."""
import inspect
import time
from nba_api.stats.endpoints import leaguedashteamstats

params = inspect.signature(leaguedashteamstats.LeagueDashTeamStats.__init__).parameters
month_param = next(name for name in params if name.startswith("month"))
print("month parameter:", month_param, flush=True)
for season in ("2025-26", "2024-25"):
    for month_number, month_name in enumerate(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"], start=1):
        frame = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star="Regular Season", per_mode_detailed="Totals",
            measure_type_detailed_defense="Base", timeout=90, **{month_param: month_number},
        ).get_data_frames()[0]
        games = int(frame["GP"].sum())
        s = lambda f: float(frame[f].sum())
        print(season, month_name, {"FTA": round(s("FTA") / games, 2), "PF": round(s("PF") / games, 2),
              "TOV": round(s("TOV") / games, 2), "FG3_PCT": round(100 * s("FG3M") / s("FG3A"), 2),
              "team_games": games}, flush=True)
        time.sleep(1.5)
