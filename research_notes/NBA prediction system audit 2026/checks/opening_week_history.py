"""Opening-week vs full-season league rates for 2017-18 .. 2025-26, plus 2025-26 by month,
plus per-team-game spread for 2026-27 to date (for a standard error). Read-only probe."""
import json
import statistics
import time

from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog

OPENERS = {
    "2017-18": ("10/17/2017", "10/23/2017"),
    "2018-19": ("10/16/2018", "10/22/2018"),
    "2019-20": ("10/22/2019", "10/28/2019"),
    "2020-21": ("12/22/2020", "12/28/2020"),
    "2021-22": ("10/19/2021", "10/25/2021"),
    "2022-23": ("10/18/2022", "10/24/2022"),
}
MONTHS = {1: "Oct", 2: "Nov", 3: "Dec", 4: "Jan", 5: "Feb", 6: "Mar", 7: "Apr"}


def rates(**filters):
    frame = leaguedashteamstats.LeagueDashTeamStats(
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
        timeout=90,
        **filters,
    ).get_data_frames()[0]
    team_games = int(frame["GP"].sum())
    total = lambda field: float(frame[field].sum())
    return {
        "FTA": round(total("FTA") / team_games, 2),
        "PF": round(total("PF") / team_games, 2),
        "TOV": round(total("TOV") / team_games, 2),
        "FG3_PCT": round(100 * total("FG3M") / total("FG3A"), 2),
        "team_games": team_games,
    }


def probe(label, **filters):
    try:
        result = rates(**filters)
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    print(label, result, flush=True)
    time.sleep(1.5)


for season, (first_day, last_day) in OPENERS.items():
    probe(f"{season} opening week", season=season, date_from_nullable=first_day, date_to_nullable=last_day)
    probe(f"{season} full season", season=season)

for month_number, month_name in MONTHS.items():
    probe(f"2025-26 {month_name}", season="2025-26", month_nullable=month_number)

log = leaguegamelog.LeagueGameLog(
    season="2026-27", season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=90
).get_data_frames()[0]
spread = {}
for field in ("FTA", "PF", "TOV"):
    values = log[field].astype(float).tolist()
    spread[field] = {
        "mean": round(statistics.mean(values), 2),
        "sd": round(statistics.stdev(values), 2),
        "se": round(statistics.stdev(values) / len(values) ** 0.5, 2),
    }
spread["team_games"] = len(log)
spread["dates"] = [str(log["GAME_DATE"].min()), str(log["GAME_DATE"].max())]
print("2026-27 team-game spread", json.dumps(spread), flush=True)
