"""
Team-level features from the two data types the 2025-26 engine never consumed:
defensive box scores and player tracking.

WHY THIS MODULE EXISTS
    predict_2025_2026.py loads <gid>box_score_defensive.csv into its game bundle
    but never turns it into a feature, and never loads player tracking at all.
    Its impact score comes from play-by-play alone (compute_game_impact takes
    only pbp_df). These two files therefore carry signal the model has never
    seen: who the defence actually stopped, and how the offence moved.

SCHEMA ORIENTATION (verified empirically, game 0022400001, 2026-08-28)
    box_score_defensive.csv: teamTricode belongs to the DEFENDER — all 19 rows
    matched the defender's own team in the same game's traditional box score.
    So playerPoints summed per teamTricode is the points that team's defenders
    ALLOWED (ATL 126 allowed vs BOS's 116 actual; matchup attribution
    over-counts slightly because a defender is credited for everything scored
    while assigned, which is why these are used as rates, never as raw totals).

    team_tracking.csv: one row per team, MIN = "240:00", DIST in miles,
    DFGM/DFGA = shots defended at the rim made/attempted AGAINST that team.

    switchesOn is 0 in every game checked (25/25 in 2023-24) — a column the API
    never populates — so it is deliberately not a feature.

LEAKAGE
    Every column this module adds is a TRAILING value: for team T and game g it
    is the mean over T's games strictly before g (expanding mean, min_periods=1,
    shift(1), sorted by date). No same-game value ever reaches the model. This
    is the same discipline matchup_impact.py uses.

SCOPE vs matchup_impact.py
    That module resolves per-player histories and weights them by the game
    roster, so it survives trades and injuries. This one aggregates to the team
    first and trails the team. It is the cheaper construction, chosen because
    its purpose is a clean A/B against the current feature set rather than a
    forward-looking roster query. A per-player version is the obvious next step
    if the ablation says these features earn their place.

Missing files are tolerated everywhere: a game with no defensive or tracking
CSV simply contributes no row, and teams with no prior history get NaN (the
caller fills with 0, as it does for matchup features).
"""
import os
import re
import glob
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
DT_CACHE = os.path.join(PROJECT_ROOT, "defensive_tracking_cache_v1.pkl")

SEASON_DIR_RE = re.compile(r"^\d{4}_\d{4}$")

# The eight per-team, per-game rates. Kept deliberately small: n_train is ~2152
# games, so every added column costs more than it looks like it does.
DEF_METRICS = ["def_pts_allowed_per100pp", "def_fg_pct_allowed",
               "def_3p_pct_allowed", "def_stops_per100pp"]
TRK_METRICS = ["trk_dist_per_min", "trk_passes_per_min",
               "trk_uncontested_share", "trk_rim_fg_pct_allowed"]
DT_METRICS = DEF_METRICS + TRK_METRICS

DT_FEATURE_COLS = ([f"home_dt_{m}" for m in DT_METRICS]
                   + [f"away_dt_{m}" for m in DT_METRICS]
                   + [f"diff_dt_{m}" for m in DT_METRICS])


# ===========================================================================
#  Parsing helpers
# ===========================================================================
def parse_minutes(value):
    """'240:00' / '9:11' -> float minutes. Returns 0.0 for blanks and DNP rows."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0.0
    if ":" in text:
        mins, _, secs = text.partition(":")
        try:
            return float(mins) + float(secs) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _num(df, col):
    """Numeric column as float, zeros where the column is absent or unparseable."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _rate(numerator, denominator, scale=1.0):
    """Guarded division: NaN when the denominator has no support, so a team with
    zero attempts is 'unknown', never a misleading 0.0."""
    return scale * numerator / denominator if denominator > 0 else np.nan


def _detect_seasons(data_dir):
    if not os.path.isdir(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir)
                  if SEASON_DIR_RE.match(d) and os.path.isdir(os.path.join(data_dir, d)))


def _load_date_map(game_ids_dir, seasons):
    """game_id -> (game_date, season) from the per-season game-ID CSVs."""
    dates = {}
    for season in seasons:
        fp = os.path.join(game_ids_dir, f"game_id_{season}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        for gid, gdate in zip(df["GAME_ID"].astype(str).str.zfill(10),
                              pd.to_datetime(df["GAME_DATE"], errors="coerce")):
            dates[gid] = (gdate, season)
    return dates


# ===========================================================================
#  Per-game -> per-team metric rows
# ===========================================================================
DEF_USECOLS = {"teamTricode", "playerPoints", "partialPossessions",
               "matchupFieldGoalsMade", "matchupFieldGoalsAttempted",
               "matchupThreePointersMade", "matchupThreePointersAttempted",
               "steals", "blocks", "matchupTurnovers"}
TRK_USECOLS = {"TEAM_ABBREVIATION", "MIN", "DIST", "PASS", "CFGA", "UFGA", "DFGM", "DFGA"}


def _read_subset(path, wanted):
    """Read only the columns we use — this runs over ~17k files, and parsing the
    other 20+ columns per file is the single largest cost in a full build."""
    try:
        return pd.read_csv(path, usecols=lambda c: c in wanted)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _defensive_rows(path):
    """Per-team defensive rates for one game, or [] when the file is unusable."""
    df = _read_subset(path, DEF_USECOLS)
    if df is None or df.empty or "teamTricode" not in df.columns:
        return []

    df = df.assign(
        _pts=_num(df, "playerPoints"),
        _poss=_num(df, "partialPossessions"),
        _fgm=_num(df, "matchupFieldGoalsMade"),
        _fga=_num(df, "matchupFieldGoalsAttempted"),
        _fg3m=_num(df, "matchupThreePointersMade"),
        _fg3a=_num(df, "matchupThreePointersAttempted"),
        _stl=_num(df, "steals"),
        _blk=_num(df, "blocks"),
        _tov=_num(df, "matchupTurnovers"),
    )
    rows = []
    for team, g in df.groupby("teamTricode"):
        poss = g["_poss"].sum()
        rows.append({
            "team": team,
            "def_pts_allowed_per100pp": _rate(g["_pts"].sum(), poss, 100.0),
            "def_fg_pct_allowed": _rate(g["_fgm"].sum(), g["_fga"].sum()),
            "def_3p_pct_allowed": _rate(g["_fg3m"].sum(), g["_fg3a"].sum()),
            "def_stops_per100pp": _rate(g["_stl"].sum() + g["_blk"].sum() + g["_tov"].sum(),
                                        poss, 100.0),
        })
    return rows


def _tracking_rows(path):
    """Per-team tracking rates for one game, or [] when the file is unusable."""
    df = _read_subset(path, TRK_USECOLS)
    if df is None or df.empty or "TEAM_ABBREVIATION" not in df.columns:
        return []

    numeric = {c: _num(df, c) for c in ("DIST", "PASS", "CFGA", "UFGA", "DFGM", "DFGA")}
    rows = []
    for i, team in enumerate(df["TEAM_ABBREVIATION"]):
        minutes = parse_minutes(df["MIN"].iloc[i] if "MIN" in df.columns else None)
        cfga = float(numeric["CFGA"].iloc[i])
        ufga = float(numeric["UFGA"].iloc[i])
        rows.append({
            "team": team,
            "trk_dist_per_min": _rate(float(numeric["DIST"].iloc[i]), minutes),
            "trk_passes_per_min": _rate(float(numeric["PASS"].iloc[i]), minutes),
            "trk_uncontested_share": _rate(ufga, ufga + cfga),
            "trk_rim_fg_pct_allowed": _rate(float(numeric["DFGM"].iloc[i]),
                                            float(numeric["DFGA"].iloc[i])),
        })
    return rows


def load_team_game_metrics(data_dir=BASE_DATA_DIR, seasons=None,
                           game_ids_dir=GAME_IDS_DIR, cache_path=DT_CACHE,
                           use_cache=True):
    """One row per (game_id, team) with the eight raw same-game rates.

    Raw values only — trailing versions are derived in build_trailing_table.
    """
    seasons = seasons or _detect_seasons(data_dir)
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("seasons") == list(seasons):
                return cached["table"]
        except (OSError, pickle.UnpicklingError, KeyError, AttributeError):
            pass  # a stale or truncated cache is rebuilt, never trusted

    dates = _load_date_map(game_ids_dir, seasons)
    per_game = {}
    for season in seasons:
        pattern = os.path.join(data_dir, season, "*", "box_scores", "*box_score_defensive.csv")
        for fp in glob.glob(pattern):
            gid = os.path.basename(fp).replace("box_score_defensive.csv", "")
            for row in _defensive_rows(fp):
                per_game.setdefault((gid, row["team"]), {"game_id": gid, "season": season,
                                                         "team": row["team"]}).update(row)
        pattern = os.path.join(data_dir, season, "*", "player_tracking", "*team_tracking.csv")
        for fp in glob.glob(pattern):
            gid = os.path.basename(fp).replace("team_tracking.csv", "")
            for row in _tracking_rows(fp):
                per_game.setdefault((gid, row["team"]), {"game_id": gid, "season": season,
                                                         "team": row["team"]}).update(row)

    if not per_game:
        return pd.DataFrame(columns=["game_id", "season", "team", "game_date"] + DT_METRICS)

    table = pd.DataFrame(list(per_game.values()))
    table["game_date"] = table["game_id"].map(lambda g: dates.get(g, (pd.NaT, None))[0])
    table = table.dropna(subset=["game_date"])
    for m in DT_METRICS:
        if m not in table.columns:
            table[m] = np.nan
    table = table[["game_id", "season", "team", "game_date"] + DT_METRICS]

    # use_cache=False means "do not touch the cache at all" — reading and writing
    # both stay off, so a probe run can never leave a stale table behind.
    if use_cache and cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"seasons": list(seasons), "table": table}, f)
        except OSError as exc:
            print(f"  [warn] could not write {os.path.basename(cache_path)}: {exc}")
    return table


# ===========================================================================
#  Trailing (leakage-free) team form
# ===========================================================================
def build_trailing_table(table):
    """Add trail_<metric>: the mean over each team's PRIOR games only.

    expanding(min_periods=1).mean().shift(1) per team in date order — the same
    construction FeatureEngineer._rolling uses, so a team's first game of the
    corpus has NaN and every later game sees only its own past.
    """
    if table is None or table.empty:
        return pd.DataFrame()
    out = table.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    grp = out.groupby("team", sort=False)
    for m in DT_METRICS:
        out[f"trail_{m}"] = grp[m].transform(
            lambda s: s.expanding(min_periods=1).mean().shift(1))
    return out


def add_defensive_tracking_features(master_df, data_dir=BASE_DATA_DIR,
                                    game_ids_dir=GAME_IDS_DIR, use_cache=True):
    """Add home/away/diff trailing defensive + tracking features per game.

    Adds the 24 columns in DT_FEATURE_COLS, NaN where a team has no prior
    history (callers fillna(0), matching how matchup features are handled).
    Returns master_df unchanged in shape when no source files exist.
    """
    df = master_df.copy()
    df = df.drop(columns=[c for c in DT_FEATURE_COLS if c in df.columns], errors="ignore")
    if not {"game_id", "home_team", "away_team"}.issubset(df.columns):
        for c in DT_FEATURE_COLS:
            df[c] = np.nan
        return df

    trailing = build_trailing_table(
        load_team_game_metrics(data_dir, game_ids_dir=game_ids_dir, use_cache=use_cache))
    if trailing.empty:
        for c in DT_FEATURE_COLS:
            df[c] = np.nan
        return df

    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    trail_cols = ["game_id", "team"] + [f"trail_{m}" for m in DT_METRICS]
    side_table = trailing[trail_cols]

    for side in ("home", "away"):
        renamed = side_table.rename(
            columns={"team": f"{side}_team",
                     **{f"trail_{m}": f"{side}_dt_{m}" for m in DT_METRICS}})
        df = df.merge(renamed, on=["game_id", f"{side}_team"], how="left")

    for m in DT_METRICS:
        df[f"diff_dt_{m}"] = df[f"home_dt_{m}"] - df[f"away_dt_{m}"]

    covered = int(df[f"home_dt_{DT_METRICS[0]}"].notna().sum())
    print(f"  defensive+tracking features: {covered}/{len(df)} games covered "
          f"({trailing['game_id'].nunique()} games with source files on disk)")
    return df
