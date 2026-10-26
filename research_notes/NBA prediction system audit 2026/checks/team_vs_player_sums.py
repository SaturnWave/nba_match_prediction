"""Team totals vs sum of player totals (NBA.com): exposes team-only turnovers and team-only shot
attempts such as heaves recorded as team attempts from 2025-26. Read-only probe."""
import time
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
for season in ("2024-25", "2025-26", "2026-27"):
    common = dict(season=season, season_type_all_star="Regular Season", per_mode_detailed="Totals",
                  measure_type_detailed_defense="Base", timeout=120)
    team = leaguedashteamstats.LeagueDashTeamStats(**common).get_data_frames()[0]
    time.sleep(1.5)
    player = leaguedashplayerstats.LeagueDashPlayerStats(**common).get_data_frames()[0]
    time.sleep(1.5)
    games = int(team["GP"].sum())
    row = {"team_games": games}
    for field in ("TOV", "PF", "FGA", "FG3A", "FG3M", "FTA"):
        t, p = float(team[field].sum()), float(player[field].sum())
        row[field] = f"team {t/games:.2f} | players {p/games:.2f} | team-only {t-p:.0f}"
    row["FG3_PCT"] = (f"team {100*team.FG3M.sum()/team.FG3A.sum():.2f} | players "
                      f"{100*player.FG3M.sum()/player.FG3A.sum():.2f}")
    print(season, row, flush=True)
