"""
Step 2 of the ESPN odds: each event's lines, one median line per game.

For every event in odds_data/espn_events.csv the odds endpoint lists each
sportsbook ESPN carries. From the 2023-24 season each book has its own opening
and closing line; earlier entries ("consensus", "teamrankings") carry one line,
used as the close. Single books are sometimes off - a stale total, an alternate
spread - so the game's line is the median over books, and in-game "Live" feeds
are left out. ESPN's `spread` is the home team's: negative means the home team
was favoured.

Writes odds_data/espn_lines.csv. Responses are cached under
odds_data/raw/espn/odds/; a re-run only fetches what is missing.

Run:  py data_retriaval/odds_espn_lines.py
"""
import os
import statistics

import pandas as pd

from odds_common import ODDS_DIR, RAW_DIR, Refused, cached_json

ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event}"
            "/competitions/{event}/odds")
EVENTS_PATH = os.path.join(ODDS_DIR, "espn_events.csv")
OUT_PATH = os.path.join(ODDS_DIR, "espn_lines.csv")
FIELDS = ("spread_open", "spread_close", "total_open", "total_close", "p_home_open", "p_home_close")


def number(value):
    """'+6.5', '-110', 219.5, {'american': '+6.5', ...}, None -> float or None.

    Inside the open/close blocks every value is an object carrying the same
    number in several notations; the American one is the line as quoted.
    """
    if isinstance(value, dict):
        value = value.get("american", value.get("alternateDisplayValue"))
    if value is None:
        return None
    try:
        return float(str(value).replace("+", ""))
    except ValueError:
        return None


def total_of(block):
    """The total inside an item's open/close block, e.g. {'total': {'american': '219.5'}}."""
    return number((block or {}).get("total"))


def no_vig_home(home_american, away_american):
    """The home side's win probability implied by two moneylines, the book's margin removed."""
    if not home_american or not away_american:
        return None
    implied = [1 / (1 + (a / 100 if a > 0 else 100 / -a)) for a in (home_american, away_american)]
    return implied[0] / sum(implied)


def book_line(item):
    """One sportsbook's lines; opening values are None when the book gives no open.

    Moneylines become a probability per book before any median is taken: a
    median of American odds straddling -100/+100 (say -105 and +105) is not a price.
    """
    home, away = item.get("homeTeamOdds", {}), item.get("awayTeamOdds", {})
    home_open, home_close = home.get("open") or {}, home.get("close") or {}
    away_open, away_close = away.get("open") or {}, away.get("close") or {}
    close_home = number(home_close.get("moneyLine")) if home_close else number(home.get("moneyLine"))
    close_away = number(away_close.get("moneyLine")) if away_close else number(away.get("moneyLine"))
    return {
        "spread_open": number(home_open.get("pointSpread")),
        "spread_close": number(home_close.get("pointSpread")) if home_close else number(item.get("spread")),
        "total_open": total_of(item.get("open")),
        "total_close": total_of(item.get("close")) or number(item.get("overUnder")),
        "p_home_open": no_vig_home(number(home_open.get("moneyLine")), number(away_open.get("moneyLine"))),
        "p_home_close": no_vig_home(close_home, close_away),
    }


def game_line(event_id):
    payload = cached_json(ODDS_URL.format(event=event_id), os.path.join(RAW_DIR, "espn", "odds", f"{event_id}.json"))
    books = [(item.get("provider", {}).get("name", "?"), book_line(item)) for item in payload.get("items", [])
             if "live" not in item.get("provider", {}).get("name", "").lower()]
    row = {"event_id": event_id, "books": len(books), "book_names": "|".join(sorted(name for name, _ in books))}
    for field in FIELDS:
        values = [line[field] for _, line in books if line[field] is not None]
        row[field] = statistics.median(values) if values else None
    return row


def main():
    events = pd.read_csv(EVENTS_PATH, dtype={"event_id": str})
    print(f"{len(events)} ESPN maci icin oranlar", flush=True)
    rows = []
    try:
        for n, event_id in enumerate(events["event_id"], 1):
            rows.append(game_line(event_id))
            if n % 200 == 0:
                print(f"  {n}/{len(events)}", flush=True)
    except Refused as refusal:
        print(f"DURDU: {refusal}")
    lines = events.merge(pd.DataFrame(rows), on="event_id")
    lines.to_csv(OUT_PATH, index=False)
    no_line = int(lines["spread_close"].isna().sum())
    print(f"{len(lines)} mac -> {OUT_PATH} (kapanis handikabi olmayan {no_line})")


if __name__ == "__main__":
    main()
