"""
Opening and closing lines for 2017-18 to 2022-23 from the Sportsbook Reviews odds archive.

The archive has one page per season: every game is two rows, the visitor's and
then the home team's (two "N" rows at a neutral site such as the 2020 bubble).
Each row has the final score, the opening and closing line and the team's
moneyline. In the Open and Close columns the favourite's row carries the point
spread and the other row the total, so which number is which is read from its
size - a total is over 100 and no spread is. The archive ends on 2023-01-16;
the rest of 2022-23 and later seasons come from ESPN (odds_espn.py).

Pages are cached under odds_data/raw/sbr/ (ignored by git) and parsed into
odds_data/sbr_lines.csv, one row per game with lines from the first-listed
team's side: a negative spread means that team was favoured. Matching to NBA
game ids, and turning everything to the home team's side, is odds_build.py's job.
The site's robots.txt allows the archive; it is fetched one page per season.

Run:  py data_retriaval/odds_sbr.py
"""
import io
import math
import os
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "odds_data", "raw", "sbr")
OUT_PATH = os.path.join(PROJECT_ROOT, "odds_data", "sbr_lines.csv")
URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}"
SEASONS = ("2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba_match_prediction personal research)"}
TIMEOUT = 60
DELAY = 3
TOTAL_FLOOR = 100          # every total is above this, every spread below it
TEAMS = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA", "Chicago": "CHI",
    "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "GoldenState": "GSW",
    "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND", "LAClippers": "LAC", "LALakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN", "NewOrleans": "NOP",
    "NewYork": "NYK", "OklahomaCity": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "SanAntonio": "SAS", "Toronto": "TOR", "Utah": "UTA",
    "Washington": "WAS",
}


def season_page(season):
    """The season's archive page, from the local cache when it is there."""
    path = os.path.join(RAW_DIR, f"nba-odds-{season}.html")
    if not os.path.exists(path):
        response = requests.get(URL.format(season=season), headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        os.makedirs(RAW_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
        time.sleep(DELAY)
    with open(path, encoding="utf-8") as f:
        return f.read()


def number(value):
    """A line as a float: 'pk' (pick'em) is 0, anything unreadable is missing."""
    text = str(value).strip().lower()
    if text == "pk":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return math.nan


def split_line(value_first, value_second):
    """(spread from the first team's side, total) out of one column's two rows."""
    first, second = number(value_first), number(value_second)
    if first >= TOTAL_FLOOR > second:          # second team's row has the spread: it was favoured
        return second, first
    if second >= TOTAL_FLOOR > first:          # first team was favoured
        return -first, second
    return math.nan, math.nan


def full_dates(mmdd_values, start_year):
    """'1017', '105', ... -> ISO dates; the year turns when the month goes back (December to January)."""
    year, previous_month, dates = start_year, None, []
    for value in mmdd_values:
        text = str(value).zfill(4)
        month, day = int(text[:2]), int(text[2:])
        if previous_month is not None and month < previous_month:
            year += 1
        previous_month = month
        dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates


def parse(html, season):
    table = pd.read_html(io.StringIO(html))[0]
    table.columns = table.iloc[0]
    table = table.iloc[1:].reset_index(drop=True)
    if len(table) % 2:
        raise ValueError(f"{season}: tek sayida satir ({len(table)}), maclar ikiser satir olmali")
    unknown = sorted(set(table["Team"].astype(str)) - set(TEAMS))
    if unknown:
        raise ValueError(f"{season}: tanimadigim takim adlari {unknown}")
    table["date"] = full_dates(table["Date"], int(season[:4]))
    first, second = table.iloc[0::2].reset_index(drop=True), table.iloc[1::2].reset_index(drop=True)
    if (first["date"] != second["date"]).any():
        raise ValueError(f"{season}: ayni macin iki satirinin tarihi farkli")
    rows = []
    for a, b in zip(first.itertuples(index=False), second.itertuples(index=False)):
        spread_open, total_open = split_line(a.Open, b.Open)
        spread_close, total_close = split_line(a.Close, b.Close)
        rows.append({"season": season, "date": a.date, "team_a": TEAMS[a.Team], "team_b": TEAMS[b.Team],
                     "neutral": a.VH == "N", "score_a": int(a.Final), "score_b": int(b.Final),
                     "spread_open_a": spread_open, "total_open": total_open,
                     "spread_close_a": spread_close, "total_close": total_close,
                     "ml_a": number(a.ML), "ml_b": number(b.ML)})
    return pd.DataFrame(rows)


def main():
    frames = []
    for season in SEASONS:
        games = parse(season_page(season), season)
        unreadable = int(games["spread_close_a"].isna().sum())
        print(f"{season}: {len(games)} mac, {games['date'].min()} .. {games['date'].max()}, "
              f"kapanis cizgisi okunamayan {unreadable}")
        frames.append(games)
    lines = pd.concat(frames, ignore_index=True)
    lines.to_csv(OUT_PATH, index=False)
    print(f"{len(lines)} mac -> {OUT_PATH}")


if __name__ == "__main__":
    main()
