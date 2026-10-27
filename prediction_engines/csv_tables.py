"""
The tables the pipeline reads, built from the CSVs in nba_data/: the phone database made optional.

WHY
    value_engine, impact_v5 and pregame_roster read their game list, player
    box scores and tracking from phonedb_cache/, a copy of the phone database
    pulled on 2026-09-12. That copy has no 2026-27 game and the database is
    offline, so nothing after 2025-26 could reach the impact cache or the
    roster features. The database was only ever these same CSVs loaded into
    tables (DB_KURTARMA.md), so the tables are rebuilt from them here, in the
    database's own column names, and the consumers read them through load().

TABLES
    games                   game_id, season: the played regular-season games
    game_summary            one row per game: sides, team ids, final score
    box_player_traditional  one row per listed player-game, database schema
    player_tracking         one row per listed player-game, database schema
    team_traditional_raw    the traditional team box, CSV schema (csv_master)
    team_advanced_raw       the advanced team box, CSV schema (csv_master)

    The game list is game_ids/game_id_<season>.csv. The per-game files come
    in two layouts - the V2 endpoints' columns, and the V3 endpoints' renamed
    to V2 names with five extra columns - and both map onto one schema here.

CONTRACTS
    A rebuild that breaks one of these raises TableContractError and writes
    nothing: an id or a key column is missing, a (game, player) pair is
    listed twice, a game does not have exactly two teams, a played game has no
    final score, or a team's player points do not add up to its team points.

CACHE
    table_cache/csv_tables.pkl (ignored by git), with the size of every file
    each game was read from. An update re-reads only games that are new or
    whose files changed size, so the tracking the NBA publishes a day or two
    after a game is picked up on the next run. A game with all its files and
    older than SETTLE_DAYS is settled and not looked at, which spares the slow
    disk minutes of file checks; --full looks at everything.

Run:  py prediction_engines/csv_tables.py            (bring the tables up to date)
      py prediction_engines/csv_tables.py --full     (look at every game's files again)
      py prediction_engines/csv_tables.py --check    (and compare with phonedb_cache/ where both exist)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
CACHE_PATH = os.path.join(PROJECT_ROOT, "table_cache", "csv_tables.pkl")
PHONEDB_DIR = os.path.join(PROJECT_ROOT, "phonedb_cache")
MIN_BYTES = 65                     # an endpoint with nothing to return still writes a header
SETTLE_DAYS = 14                   # after this, a game with all its files is not looked at again
TABLES = ("games", "game_summary", "box_player_traditional", "player_tracking",
          "team_traditional_raw", "team_advanced_raw")

# Per-game files, relative to nba_data/<season>/<game_id>/
FILES = {
    "player_traditional": ("box_scores", "box_score_traditional.csv"),
    "team_traditional": ("box_scores", "box_score_traditional_team.csv"),
    "team_advanced": ("box_scores", "box_score_advanced_team.csv"),
    "player_tracking": ("player_tracking", "player_tracking.csv"),
}

# CSV column -> database column. V2 files carry the first block only; V3 files
# carry all of it. A column a file lacks becomes missing, as in the database.
PLAYER_TRADITIONAL = {
    "GAME_ID": "game_id", "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team_abbreviation",
    "TEAM_CITY": "team_city", "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
    "NICKNAME": "nickname", "START_POSITION": "start_position", "COMMENT": "comment", "MIN": "min",
    "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct", "FG3M": "fg3_m", "FG3A": "fg3_a",
    "FG3_PCT": "fg3_pct", "FTM": "ftm", "FTA": "fta", "FT_PCT": "ft_pct", "OREB": "oreb",
    "DREB": "dreb", "REB": "reb", "AST": "ast", "STL": "stl", "BLK": "blk", "TO": "to", "PF": "pf",
    "PTS": "pts", "PLUS_MINUS": "plus_minus",
    "TEAM_NAME": "team_name", "teamSlug": "team_slug", "firstName": "first_name",
    "familyName": "family_name", "playerSlug": "player_slug", "jerseyNum": "jersey_num",
}
PLAYER_TRACKING = {
    "GAME_ID": "game_id", "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team_abbreviation",
    "TEAM_CITY": "team_city", "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
    "START_POSITION": "start_position", "COMMENT": "comment", "MIN": "min", "SPD": "spd",
    "DIST": "dist", "ORBC": "orbc", "DRBC": "drbc", "RBC": "rbc", "TCHS": "tchs", "SAST": "sast",
    "FTAST": "ftast", "PASS": "pass", "AST": "ast", "CFGM": "cfgm", "CFGA": "cfga",
    "CFG_PCT": "cfg_pct", "UFGM": "ufgm", "UFGA": "ufga", "UFG_PCT": "ufg_pct", "FG_PCT": "fg_pct",
    "DFGM": "dfgm", "DFGA": "dfga", "DFG_PCT": "dfg_pct", "GAME_ID_CLEAN": "game_id_clean",
}
TEXT_COLUMNS = {"game_id", "team_abbreviation", "team_city", "player_name", "nickname",
                "start_position", "comment", "min", "team_name", "team_slug", "first_name",
                "family_name", "player_slug", "jersey_num", "game_id_clean"}
ID_COLUMNS = ("team_id", "player_id")


class TableContractError(ValueError):
    """A rebuilt table broke one of the contracts in the module docstring."""


# ---------------------------------------------------------------------------
#  Reading one game
# ---------------------------------------------------------------------------
def game_files(season, game_id):
    """{kind: path} for one game's files, whether or not they exist."""
    game_dir = os.path.join(DATA_DIR, season, game_id)
    return {kind: os.path.join(game_dir, sub, game_id + suffix) for kind, (sub, suffix) in FILES.items()}


def fingerprint(paths):
    """The size of each file, or 0 when it is absent or only a header."""
    sizes = []
    for path in paths.values():
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        sizes.append(size if size >= MIN_BYTES else 0)
    return tuple(sizes)


def _read(path, columns):
    """A per-game CSV renamed to the database schema; None when the file is absent or a header."""
    if not (os.path.isfile(path) and os.path.getsize(path) >= MIN_BYTES):
        return None
    frame = pd.read_csv(path, dtype=str).rename(columns=columns).reindex(columns=list(columns.values()))
    for column in frame.columns:
        if column in TEXT_COLUMNS:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["game_id"] = frame["game_id"].str.zfill(10)
    return frame


def _read_team_raw(path, season):
    """A team box in its own CSV schema, as csv_master reads it."""
    if not (os.path.isfile(path) and os.path.getsize(path) >= MIN_BYTES):
        return None
    frame = pd.read_csv(path, dtype={"GAME_ID": str})
    frame["season"] = season
    frame["game_id"] = frame["GAME_ID"].str.zfill(10)
    return frame


def read_game(season, game_id):
    """Every table's rows for one game; a kind whose file is missing is None."""
    paths = game_files(season, game_id)
    return {
        "box_player_traditional": _read(paths["player_traditional"], PLAYER_TRADITIONAL),
        "player_tracking": _read(paths["player_tracking"], PLAYER_TRACKING),
        "team_traditional_raw": _read_team_raw(paths["team_traditional"], season),
        "team_advanced_raw": _read_team_raw(paths["team_advanced"], season),
    }


# ---------------------------------------------------------------------------
#  The game list and the per-game summary
# ---------------------------------------------------------------------------
def game_list():
    """Played regular-season games with their sides, from game_ids/ (csv_master's reader)."""
    sys.path.insert(0, HERE)
    import csv_master                                  # noqa: E402 - sibling module
    schedule = csv_master.load_schedule()
    return schedule.drop_duplicates("game_id").sort_values(["game_date", "game_id"]).reset_index(drop=True)


def summarise(schedule, team_traditional):
    """game_summary in the database's layout, from the schedule and the traditional team box."""
    box = team_traditional[["game_id", "TEAM_ABBREVIATION", "TEAM_ID", "PTS"]].copy()
    box["PTS"] = pd.to_numeric(box["PTS"], errors="coerce")
    out = schedule[["game_id", "season", "game_date", "home_team", "away_team"]].copy()
    for side in ("home", "away"):
        side_box = box.rename(columns={"TEAM_ABBREVIATION": f"{side}_team", "TEAM_ID": f"{side}_team_id",
                                       "PTS": f"{side}_pts"})
        out = out.merge(side_box, on=["game_id", f"{side}_team"], how="left")
    out = out.rename(columns={"home_team": "home_abbr", "away_team": "away_abbr"})
    out["margin"] = out["home_pts"] - out["away_pts"]
    out["home_win"] = np.where(out["margin"].isna(), np.nan, (out["margin"] > 0).astype(float))
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    return out[["game_id", "season", "home_team_id", "away_team_id", "home_abbr", "away_abbr",
                "home_pts", "away_pts", "margin", "home_win", "game_date"]]


# ---------------------------------------------------------------------------
#  Contracts
# ---------------------------------------------------------------------------
def check_contracts(tables):
    """Raise TableContractError listing every broken contract; return the tables otherwise."""
    problems = []
    players = tables["box_player_traditional"]
    for name in ("box_player_traditional", "player_tracking"):
        frame = tables[name]
        for column in ("game_id",) + ID_COLUMNS:
            missing = int(frame[column].isna().sum())
            if missing:
                problems.append(f"{name}: {missing} satirda {column} yok")
        doubled = int(frame.duplicated(["game_id", "player_id"]).sum())
        if doubled:
            problems.append(f"{name}: {doubled} (mac, oyuncu) cifti iki kez listelenmis")
    teams_per_game = players.groupby("game_id")["team_id"].nunique()
    odd = teams_per_game[teams_per_game != 2]
    if len(odd):
        problems.append(f"box_player_traditional: {len(odd)} macta takim sayisi 2 degil, ornek {list(odd.index[:3])}")
    summary = tables["game_summary"]
    no_score = summary[summary["home_pts"].isna() | summary["away_pts"].isna()]
    if len(no_score):
        problems.append(f"game_summary: {len(no_score)} oynanmis macin skoru yok, ornek {list(no_score['game_id'][:3])}")
    team_pts = (tables["team_traditional_raw"].assign(PTS=lambda t: pd.to_numeric(t["PTS"], errors="coerce"))
                .set_index(["game_id", "TEAM_ID"])["PTS"])
    player_pts = players.groupby(["game_id", "team_id"])["pts"].sum(min_count=1)
    both = pd.concat([player_pts.rename("players"), team_pts.rename("team")], axis=1, join="inner")
    off = both[(both["players"] - both["team"]).abs() > 0.5]
    if len(off):
        problems.append(f"box_player_traditional: {len(off)} takim-macta oyuncu sayilari takim sayisini "
                        f"tutmuyor, ornek {[tuple(i) for i in off.index[:3]]}")
    if problems:
        raise TableContractError("; ".join(problems))
    return tables


# ---------------------------------------------------------------------------
#  Building and caching
# ---------------------------------------------------------------------------
_LOADED = {}                        # the tables, once read in this process


def _empty_cache():
    return {"fingerprints": {}, "rows": {}}


def _read_cache():
    if not os.path.exists(CACHE_PATH):
        return _empty_cache()
    return pd.read_pickle(CACHE_PATH)


def settled(cached_print, game_date, today):
    """A game whose files are all there and which is older than SETTLE_DAYS is not looked at again.

    Looking at every file costs minutes on the hard disk nba_data/ lives on;
    a finished game with all four files does not change. A game still missing
    a file (tracking published late) or played recently is checked each time.
    """
    return (cached_print is not None and all(cached_print)
            and (today - game_date).days > SETTLE_DAYS)


def update(verbose=True, full=False):
    """Bring every table up to date with nba_data/; return {name: DataFrame}.

    `full` looks at every game's files, settled or not.
    """
    t0 = time.time()
    schedule = game_list()
    cache = _read_cache()
    wanted = dict(zip(schedule["game_id"], schedule["season"]))
    dates = dict(zip(schedule["game_id"], pd.to_datetime(schedule["game_date"])))
    today = pd.Timestamp.today().normalize()
    stale = [gid for gid in cache["rows"] if gid not in wanted]
    for gid in stale:                                   # a game that left the game list
        cache["rows"].pop(gid, None)
        cache["fingerprints"].pop(gid, None)
    todo, skipped = [], 0
    for gid, season in wanted.items():
        cached_print = cache["fingerprints"].get(gid)
        if not full and settled(cached_print, dates[gid], today):
            skipped += 1
            continue
        print_now = fingerprint(game_files(season, gid))
        if cache["fingerprints"].get(gid) != print_now:
            todo.append((gid, season, print_now))
    if verbose:
        print(f"  {len(wanted):,} mac ({skipped:,} kapanmis, bakilmadi), {len(todo):,} tanesi yeni ya da "
              f"degismis ({time.time() - t0:.0f} sn)", flush=True)
    for n, (gid, season, print_now) in enumerate(todo, 1):
        cache["rows"][gid] = read_game(season, gid)
        cache["fingerprints"][gid] = print_now
        if verbose and n % 500 == 0:
            print(f"    {n:,}/{len(todo):,} ({time.time() - t0:.0f} sn)", flush=True)

    tables = assemble(schedule, cache["rows"])
    check_contracts(tables)
    if todo or stale or not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        partial = f"{CACHE_PATH}.{os.getpid()}.part"     # the dashboard may be updating it too
        pd.to_pickle(cache, partial)
        os.replace(partial, CACHE_PATH)
    _LOADED.clear()
    _LOADED.update(tables)
    if verbose:
        sizes = ", ".join(f"{name} {len(frame):,}" for name, frame in tables.items())
        print(f"  tablolar: {sizes} ({time.time() - t0:.0f} sn)")
    return tables


def assemble(schedule, rows_by_game):
    """The tables from the cached per-game rows, in the game list's order."""
    parts = {name: [] for name in ("box_player_traditional", "player_tracking",
                                   "team_traditional_raw", "team_advanced_raw")}
    for gid in schedule["game_id"]:
        for name, frame in (rows_by_game.get(gid) or {}).items():
            if frame is not None:
                parts[name].append(frame)
    tables = {name: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
              for name, frames in parts.items()}
    with_box = schedule[schedule["game_id"].isin(set(tables["team_traditional_raw"]["game_id"]))]
    tables["games"] = with_box[["game_id", "season"]].reset_index(drop=True)
    tables["game_summary"] = summarise(with_box, tables["team_traditional_raw"])
    for name in ("box_player_traditional", "player_tracking"):
        frame = tables[name]
        for column in ID_COLUMNS:
            frame[column] = frame[column].astype("int64")
    return {name: tables[name] for name in TABLES}


def load(name):
    """One table, brought up to date once per process."""
    if name not in TABLES:
        raise KeyError(f"bilinmeyen tablo {name!r}; mevcut: {TABLES}")
    if not _LOADED:
        _LOADED.update(update(verbose=False))
    return _LOADED[name]


# ---------------------------------------------------------------------------
#  Parity with the phone database's copy
# ---------------------------------------------------------------------------
COMPARED = {
    "games": ("game_id", ["season"]),
    "game_summary": ("game_id", ["season", "home_team_id", "away_team_id", "home_abbr", "away_abbr",
                                 "home_pts", "away_pts", "margin", "home_win", "game_date"]),
    "box_player_traditional": (["game_id", "player_id"],
                               ["team_id", "team_abbreviation", "player_name", "comment", "min", "fgm",
                                "fga", "fg3_m", "fg3_a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk",
                                "to", "pf", "pts", "plus_minus"]),
    "player_tracking": (["game_id", "player_id"], ["team_id", "team_abbreviation", "player_name", "sast",
                                                   "ftast", "min"]),
}


def _same(a, b):
    """Element-wise equality that counts two missing values as equal and compares numbers numerically."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return (np.isclose(a.astype(float), b.astype(float), equal_nan=True))
    a, b = a.astype("string").fillna("<NA>").str.strip(), b.astype("string").fillna("<NA>").str.strip()
    return (a == b).to_numpy()


def compare_with_phonedb(tables):
    """{table: report} where both sources have the table, over the database's games."""
    report = {}
    for name, (key, columns) in COMPARED.items():
        path = os.path.join(PHONEDB_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            continue
        db = pd.read_pickle(path)
        db["game_id"] = db["game_id"].astype(str).str.zfill(10)
        mine = tables[name]
        mine = mine[mine["game_id"].isin(set(db["game_id"]))]
        keys = [key] if isinstance(key, str) else key
        both = db.merge(mine, on=keys, how="outer", suffixes=("_db", "_csv"), indicator=True)
        entry = {"rows_db": len(db), "rows_csv": len(mine),
                 "only_db": int((both["_merge"] == "left_only").sum()),
                 "only_csv": int((both["_merge"] == "right_only").sum()), "columns": {}}
        matched = both[both["_merge"] == "both"]
        for column in columns:
            if f"{column}_db" not in matched or f"{column}_csv" not in matched:
                continue
            left, right = matched[f"{column}_db"], matched[f"{column}_csv"]
            if column == "game_date":
                left, right = pd.to_datetime(left).dt.date.astype(str), pd.to_datetime(right).dt.date.astype(str)
            differ = int((~_same(left, right)).sum())
            if differ:
                example = matched.loc[~_same(left, right), keys + [f"{column}_db", f"{column}_csv"]].head(2)
                entry["columns"][column] = {"differ": differ, "example": example.to_dict(orient="records")}
        report[name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="phonedb_cache/ ile karsilastir")
    parser.add_argument("--full", action="store_true", help="kapanmis maclar dahil her dosyaya bak")
    args = parser.parse_args()
    tables = update(full=args.full)
    if not args.check:
        return 0
    report = compare_with_phonedb(tables)
    clean = True
    for name, entry in report.items():
        print(f"  {name}: veritabani {entry['rows_db']:,}, CSV {entry['rows_csv']:,}, yalniz veritabaninda "
              f"{entry['only_db']}, yalniz CSV'de {entry['only_csv']}")
        for column, detail in entry["columns"].items():
            clean = False
            print(f"      {column}: {detail['differ']:,} fark, ornek {detail['example']}")
        clean &= entry["only_db"] == 0 and entry["only_csv"] == 0
    print("ESIT" if clean else "FARK VAR")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
