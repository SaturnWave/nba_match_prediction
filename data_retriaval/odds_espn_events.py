"""
Step 1 of the ESPN odds: which ESPN event is which NBA game.

ESPN numbers its games differently from the NBA, so first each game date's
scoreboard is read (one request per date) and every regular-season event is
listed with its date and teams in NBA tricodes. The dates are the regular-season
dates the project already knows (phonedb_cache/game_summary.pkl, plus the
played 2026-27 games in game_ids/schedule_2026_2027.csv) from FIRST_DATE on.
FIRST_DATE overlaps the Sportsbook Reviews archive by two weeks (it ends
2023-01-16), so the two sources can be checked against each other.

Writes odds_data/espn_events.csv. Scoreboards are cached under
odds_data/raw/espn/scoreboard/; a re-run only fetches dates not yet cached.

Run:  py data_retriaval/odds_espn_events.py
"""
import os

import pandas as pd

from odds_common import ODDS_DIR, PROJECT_ROOT, RAW_DIR, Refused, cached_json, nba_code

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
FIRST_DATE = "2023-01-01"
REGULAR_SEASON = 2
OUT_PATH = os.path.join(ODDS_DIR, "espn_events.csv")


def game_dates():
    """Regular-season dates from FIRST_DATE up to the last played game."""
    games = pd.read_pickle(os.path.join(PROJECT_ROOT, "phonedb_cache", "game_summary.pkl"))
    dates = set(pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d"))
    schedule_path = os.path.join(PROJECT_ROOT, "game_ids", "schedule_2026_2027.csv")
    if os.path.exists(schedule_path):
        schedule = pd.read_csv(schedule_path)
        played = schedule[(schedule["season_type"] == "regular_season") & (schedule["status"] == "final")]
        dates |= set(played["game_date"])
    return sorted(d for d in dates if d >= FIRST_DATE)


def events_on(date):
    board = cached_json(SCOREBOARD_URL, os.path.join(RAW_DIR, "espn", "scoreboard", f"{date}.json"),
                        params={"dates": date.replace("-", "")})
    rows = []
    for event in board.get("events", []):
        if event.get("season", {}).get("type") != REGULAR_SEASON:
            continue
        sides = {c["homeAway"]: c for c in event["competitions"][0]["competitors"]}
        rows.append({"event_id": event["id"], "date": date,
                     "home": nba_code(sides["home"]["team"]["abbreviation"]),
                     "away": nba_code(sides["away"]["team"]["abbreviation"]),
                     "home_score": sides["home"].get("score"), "away_score": sides["away"].get("score")})
    return rows


def main():
    dates = game_dates()
    print(f"{len(dates)} mac gunu ({dates[0]} .. {dates[-1]})", flush=True)
    rows = []
    try:
        for n, date in enumerate(dates, 1):
            rows.extend(events_on(date))
            if n % 50 == 0:
                print(f"  {n}/{len(dates)} gun, {len(rows)} mac", flush=True)
    except Refused as refusal:
        print(f"DURDU: {refusal}")
    events = pd.DataFrame(rows).drop_duplicates("event_id")
    events.to_csv(OUT_PATH, index=False)
    print(f"{len(events)} ESPN maci -> {OUT_PATH}")


if __name__ == "__main__":
    main()
