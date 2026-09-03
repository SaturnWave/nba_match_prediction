"""load_master_frame() without a database, straight from the CSVs on disk.

WHY
    The MariaDB instance lived on a phone, and on 2026-09-03 the phone's Linux
    install failed and took the database with it. Nothing in the repo could
    rebuild it: db_build_derived.py only builds derived tables from base tables
    that already exist, and whatever loaded the base tables lived on the phone.

    No data was lost. The database was a speed layer over nba_data/, which holds
    all 10,749 games as CSV and is committed to this repository. What was lost
    is the 60x speedup - and, it turned out, less than that, because
    build_dataset_db.build() calls the database exactly once:

        master = db.load_master_frame(...)

    Everything after that line is pandas over local files. So the whole
    dependency is one function returning one frame, and reproducing that frame
    from CSV removes the database from the critical path rather than recreating
    it. This module is that reproduction, with the same signature and the same
    columns, so build_dataset_db can take either source.

WHAT FEEDS WHAT
    game_ids/game_id_{season}.csv         game_id, date, and which side is home
    {game}/box_scores/*_traditional_team  the 20 traditional team columns
    {game}/box_scores/*_advanced_team     the 13 advanced team columns
    derived here                          home_rest, away_rest, rest_diff,
                                          home_b2b, away_b2b

    The rest columns came from the database's training_games table and are the
    one thing with no CSV behind it. They are pure schedule arithmetic, so they
    are recomputed here instead: rest is the number of calendar days since that
    team's previous game in the same season, 0 for its first, and a back-to-back
    is a rest of exactly 1. That reproduces the stored columns on 99.85% of rows
    - the 16 that differ are ones where the stored rest_diff disagreed with its
    own home_rest and away_rest, so the disagreement is in the old data.

SPEED
    Two small CSVs per game rather than the six the full pipeline reads, cached
    to a pickle afterwards. The first pass is minutes; every pass after it is
    seconds. The cache is keyed by season and is safe to delete.
"""
import glob
import os
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_ID_DIR = os.path.join(PROJECT_ROOT, "game_ids")
CACHE_PATH = os.path.join(PROJECT_ROOT, "team_box_cache_v1.pkl")

# The CSV headers are already the UPPER_SNAKE names downstream code expects, so
# these are the columns to keep rather than a rename. They are deliberately the
# VALUES of db_source's TRADITIONAL_MAP and ADVANCED_MAP: same columns, sourced
# differently. _check_against_db_source() below asserts they have not drifted.
TRADITIONAL_COLUMNS = [
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS",
]
ADVANCED_COLUMNS = [
    "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING", "TS_PCT", "EFG_PCT",
    "POSS", "PIE", "AST_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT",
]
REST_COLUMNS = ["home_rest", "away_rest", "rest_diff", "home_b2b", "away_b2b"]


def available_seasons():
    """Seasons with both a schedule file and a game directory."""
    seasons = []
    for path in sorted(glob.glob(os.path.join(GAME_ID_DIR, "game_id_*.csv"))):
        season = os.path.basename(path)[len("game_id_"):-len(".csv")]
        if os.path.isdir(os.path.join(DATA_DIR, season)):
            seasons.append(season)
    return seasons


def load_schedule(seasons=None):
    """game_id, season, date and the home/away sides, one row per game.

    MATCHUP is written from one team's point of view: "CLE vs. BOS" means CLE
    hosted, "BOS @ CLE" means CLE hosted. Both forms appear in every season
    (642 and 588 of them in 2025-26), so reading the separator is the only way
    to get the sides right - taking the first team as home would call 48% of
    games backwards.
    """
    seasons = seasons or available_seasons()
    frames = []
    for season in seasons:
        path = os.path.join(GAME_ID_DIR, f"game_id_{season}.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path, dtype={"GAME_ID": str})
        s["season"] = season
        frames.append(s)
    if not frames:
        return pd.DataFrame()

    sched = pd.concat(frames, ignore_index=True)
    sched["game_id"] = sched["GAME_ID"].str.zfill(10)
    sched["game_date"] = pd.to_datetime(sched["GAME_DATE"])

    matchup = sched["MATCHUP"].astype(str)
    hosted = matchup.str.contains(" vs. ", regex=False)
    parts = matchup.str.split(r" vs\. | @ ", n=1, expand=True, regex=True)
    first, second = parts[0].str.strip(), parts[1].str.strip()
    sched["home_team"] = np.where(hosted, first, second)
    sched["away_team"] = np.where(hosted, second, first)

    return sched[["game_id", "season", "game_date", "home_team", "away_team"]]


def _read_team_box(season, game_id, kind):
    """One game's team box score, or None when the file is absent or empty."""
    path = os.path.join(DATA_DIR, season, game_id, "box_scores",
                        f"{game_id}box_score_{kind}_team.csv")
    if not (os.path.exists(path) and os.path.getsize(path) > 64):
        return None
    try:
        return pd.read_csv(path, dtype={"GAME_ID": str})
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def load_team_box(seasons=None, use_cache=True, verbose=True):
    """Traditional and advanced team box scores, two rows per game.

    Cached, because this is the only slow part: 2 files per game across 10,749
    games on OneDrive-backed storage. A cache hit for the requested seasons
    skips the walk entirely.
    """
    seasons = seasons or available_seasons()
    if use_cache and os.path.exists(CACHE_PATH):
        cached = pd.read_pickle(CACHE_PATH)
        if set(seasons).issubset(set(cached["seasons"])):
            trad = cached["traditional"]
            adv = cached["advanced"]
            trad = trad[trad["season"].isin(seasons)]
            adv = adv[adv["season"].isin(seasons)]
            if verbose:
                print(f"  onbellek: {len(trad):,} takim-mac (traditional), "
                      f"{len(adv):,} (advanced)")
            return trad.reset_index(drop=True), adv.reset_index(drop=True)

    t0 = time.time()
    trad_rows, adv_rows, missing = [], [], {"traditional": 0, "advanced": 0}
    for season in seasons:
        season_dir = os.path.join(DATA_DIR, season)
        game_ids = sorted(os.listdir(season_dir)) if os.path.isdir(season_dir) else []
        for game_id in game_ids:
            for kind, sink in (("traditional", trad_rows), ("advanced", adv_rows)):
                frame = _read_team_box(season, game_id, kind)
                if frame is None:
                    missing[kind] += 1
                    continue
                frame["season"] = season
                sink.append(frame)
        if verbose:
            print(f"    {season}: {len(game_ids)} mac okundu "
                  f"({time.time() - t0:.0f} sn)", flush=True)

    trad = pd.concat(trad_rows, ignore_index=True) if trad_rows else pd.DataFrame()
    adv = pd.concat(adv_rows, ignore_index=True) if adv_rows else pd.DataFrame()
    for frame in (trad, adv):
        if not frame.empty:
            frame["game_id"] = frame["GAME_ID"].str.zfill(10)

    if verbose:
        print(f"  {len(trad):,} + {len(adv):,} takim-mac, eksik dosya "
              f"traditional {missing['traditional']}, advanced {missing['advanced']} "
              f"({time.time() - t0:.0f} sn)")
    if use_cache:
        pd.to_pickle({"seasons": list(seasons), "traditional": trad,
                      "advanced": adv}, CACHE_PATH)
    return trad, adv


def add_rest_columns(master):
    """Days since each team's previous game this season, and back-to-backs.

    Computed from the schedule rather than stored, because the schedule is the
    only thing it ever depended on. A team's first game of a season has no
    previous game; that is recorded as 0 rest and not a back-to-back, which is
    what the database's training_games table held.
    """
    sides = pd.concat([
        master[["game_id", "season", "game_date", "home_team"]]
            .rename(columns={"home_team": "team"}).assign(side="home"),
        master[["game_id", "season", "game_date", "away_team"]]
            .rename(columns={"away_team": "team"}).assign(side="away"),
    ], ignore_index=True).sort_values(["season", "team", "game_date", "game_id"])

    previous = sides.groupby(["season", "team"], sort=False)["game_date"].shift(1)
    gap = (sides["game_date"] - previous).dt.days
    sides["rest"] = gap.fillna(0.0)
    sides["b2b"] = (gap == 1).astype(float)

    wide = sides.pivot_table(index="game_id", columns="side",
                             values=["rest", "b2b"], aggfunc="first")
    wide.columns = [f"{side}_{stat}" for stat, side in wide.columns]
    wide = wide.reset_index()

    out = master.merge(wide, on="game_id", how="left")
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out


def load_master_frame(seasons=None, conn=None, verbose=True, use_cache=True):
    """One row per game, ready for FeatureEngineer - same contract as db_source.

    `conn` is accepted and ignored so this is a drop-in replacement at the one
    call site that matters, build_dataset_db.build().
    """
    seasons = seasons or available_seasons()
    master = load_schedule(seasons)
    if master.empty:
        return master

    trad_raw, adv_raw = load_team_box(seasons, use_cache=use_cache, verbose=verbose)
    if trad_raw.empty:
        raise RuntimeError("hic takim box score bulunamadi - nba_data/ eksik mi?")

    master = _attach_scores(master, trad_raw)
    master["point_diff"] = master["home_score"] - master["away_score"]
    master["total_score"] = master["home_score"] + master["away_score"]
    master["home_win"] = (master["point_diff"] > 0).astype(int)

    for raw, columns in ((trad_raw, TRADITIONAL_COLUMNS),
                         (adv_raw, ADVANCED_COLUMNS)):
        if raw.empty:
            continue
        side_frame, stats = _side_frame(raw, columns)
        for side in ("home", "away"):
            master = _attach_side(master, side_frame, stats, side)

    master = add_rest_columns(master)
    master = master.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    if verbose:
        covered = int(master["home_PTS"].notna().sum()) if "home_PTS" in master else 0
        print(f"  csv: {len(master)} mac, {master['season'].nunique()} sezon, "
              f"takim box score {covered}/{len(master)}, "
              f"rest {int(master['home_rest'].notna().sum())}/{len(master)}")
    return master


def _attach_scores(master, trad_raw):
    """Add home/away team ids and final scores from the traditional box score.

    The schedule file names the sides but carries no score; the box score
    carries the score but not which side hosted. One row per game needs both,
    joined on abbreviation.

    The score comes from the box score deliberately. The CSV pipeline once took
    it from the play-by-play's last row, which trails the true final by 2-3
    points on games whose closing plays are missing from the feed, and produced
    a fabricated 0-0 where the feed was empty - on 0021900106 that flipped the
    label, since Toronto actually won 124-120.
    """
    box = trad_raw[["game_id", "TEAM_ABBREVIATION", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "abbr"})
    box["PTS"] = pd.to_numeric(box["PTS"], errors="coerce")

    for side in ("home", "away"):
        side_box = box.rename(columns={"abbr": f"{side}_team",
                                       "TEAM_ID": f"{side}_team_id",
                                       "PTS": f"{side}_score"})
        master = master.merge(side_box, on=["game_id", f"{side}_team"], how="left")
    return master


def _side_frame(raw, columns):
    """Per-team stats keyed by (game_id, abbr), numeric and nothing else."""
    present = [c for c in columns if c in raw.columns]
    out = raw[["game_id", "TEAM_ABBREVIATION"] + present].rename(
        columns={"TEAM_ABBREVIATION": "abbr"})
    out[present] = out[present].apply(pd.to_numeric, errors="coerce")
    return out, present


def _attach_side(master, side_frame, stats, side):
    """Merge one team's stats onto the game row as home_* / away_* columns."""
    renamed = side_frame.rename(columns={c: f"{side}_{c}" for c in stats})
    renamed = renamed.rename(columns={"abbr": f"{side}_team"})
    return master.merge(renamed, on=["game_id", f"{side}_team"], how="left")
