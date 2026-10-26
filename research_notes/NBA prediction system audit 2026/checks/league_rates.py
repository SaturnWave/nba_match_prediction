"""Compare league-wide per-team-game rates: opening week vs full season, from NBA.com.

Read-only check for the research brief: is the 2026-27 opening-week jump in fouls,
free throws and turnovers (and the drop in 3P%) visible in NBA.com's own team totals,
and how large are opening-week effects in earlier seasons?
"""
import json
import sys
import time

from nba_api.stats.endpoints import leaguedashteamstats

WINDOWS = [
    # label, season, date_from, date_to (None = whole regular season)
    ("2026-27 to date", "2026-27", None, None),
    ("2025-26 opening week", "2025-26", "10/21/2025", "10/27/2025"),
    ("2025-26 full season", "2025-26", None, None),
    ("2024-25 opening week", "2024-25", "10/22/2024", "10/28/2024"),
    ("2024-25 full season", "2024-25", None, None),
    ("2023-24 opening week", "2023-24", "10/24/2023", "10/30/2023"),
    ("2023-24 full season", "2023-24", None, None),
]

FIELDS = ("FTA", "PF", "TOV", "FG3M", "FG3A", "FGA", "PTS")


def league_rates(season, date_from, date_to):
    frame = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
        date_from_nullable=date_from or "",
        date_to_nullable=date_to or "",
        timeout=90,
    ).get_data_frames()[0]
    team_games = int(frame["GP"].sum())
    totals = {field: float(frame[field].sum()) for field in FIELDS}
    rates = {field: round(totals[field] / team_games, 2) for field in FIELDS if field not in ("FG3M",)}
    rates["FG3_PCT"] = round(100 * totals["FG3M"] / totals["FG3A"], 2)
    rates["team_games"] = team_games
    return rates


def main():
    results = {}
    for label, season, date_from, date_to in WINDOWS:
        try:
            results[label] = league_rates(season, date_from, date_to)
        except Exception as error:  # report and continue; this is a one-off probe
            results[label] = {"error": f"{type(error).__name__}: {error}"}
        print(label, results[label], flush=True)
        time.sleep(2)
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
