"""
MariaDB-backed replacement for the per-game CSV reads.

WHY
    Building the engineered dataset from CSVs takes about 100 minutes, and
    almost none of that is computation: it opens roughly 64,000 files across
    10,749 games, at about 1.4 games per second on OneDrive-backed storage.
    That cost is what has kept the walk-forward harness at 14 folds and 3 seeds
    instead of the 28 folds and 5 seeds needed to settle whether the ratings
    and calibration gains are real. The same data lives in phonedb as indexed
    InnoDB tables; one season of team box scores comes back in 0.1 seconds.

    It is also more correct. Comparing the two sources over the 8,289 games
    they share turned up eight games where the CSV-built dataset is wrong and
    the database is right:
      * five games (0021900106, 0022000520, 0022000794, 0022400322, 0022400648)
        carry a fabricated 0-0 result even though a valid team box score sits
        on disk - and for 0021900106 that flips the label, since Toronto
        actually won 124-120;
      * three games (0022100028, 0022100298, 0022301202) took their score from
        the play-by-play feed's last row, which trails the box score by 2-3
        points because the final scoring plays are missing from the feed.
    Reading the team box score from the database avoids both failure modes.

WHAT THIS RETURNS
    load_master_frame() produces one row per game using the SAME column names
    the CSV pipeline produces, so FeatureEngineer, _select_features,
    team_ratings and defensive_tracking_features all consume it unchanged. It
    adds five columns the CSV path never had: home_rest, away_rest, rest_diff,
    home_b2b and away_b2b, precomputed in the database's training_games table
    and verified against an independent measurement (17.6% back-to-backs here
    against the 17.9% the improvement report derived from the schedule).

    Impact scores are NOT recomputed here. They come from the existing
    game_impact_cache_v4.pkl, keyed by game_id and derived from play-by-play,
    which this module does not change.

CREDENTIALS
    Read from environment variables, falling back to a .env file in the repo
    root. That file is gitignored: this repo auto-pushes to GitHub every five
    minutes, so a hardcoded password would be public within the hour.
"""
import os

import numpy as np
import pandas as pd
import pymysql

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Database column -> the UPPER_SNAKE name every downstream consumer expects.
TRADITIONAL_MAP = {
    "fgm": "FGM", "fga": "FGA", "fg_pct": "FG_PCT",
    "fg3_m": "FG3M", "fg3_a": "FG3A", "fg3_pct": "FG3_PCT",
    "ftm": "FTM", "fta": "FTA", "ft_pct": "FT_PCT",
    "oreb": "OREB", "dreb": "DREB", "reb": "REB", "ast": "AST",
    "stl": "STL", "blk": "BLK", "to": "TO", "pf": "PF", "pts": "PTS",
    "plus_minus": "PLUS_MINUS",
}
ADVANCED_MAP = {
    "pace": "PACE", "off_rating": "OFF_RATING", "def_rating": "DEF_RATING",
    "net_rating": "NET_RATING", "ts_pct": "TS_PCT", "efg_pct": "EFG_PCT",
    "poss": "POSS", "pie": "PIE", "ast_pct": "AST_PCT",
    "oreb_pct": "OREB_PCT", "dreb_pct": "DREB_PCT", "reb_pct": "REB_PCT",
    "tm_tov_pct": "TM_TOV_PCT",
}
REST_COLUMNS = ["home_rest", "away_rest", "rest_diff", "home_b2b", "away_b2b"]

# play_by_play is stored snake_case; compute_game_impact and every other
# consumer were written against the camelCase CSV schema, so frames handed out
# by load_pbp are renamed back to it rather than changing the callers.
PBP_MAP = {
    "action_number": "actionNumber", "clock": "clock", "period": "period",
    "team_id": "teamId", "team_tricode": "teamTricode",
    "person_id": "personId", "player_name": "playerName",
    "player_name_i": "playerNameI", "x_legacy": "xLegacy", "y_legacy": "yLegacy",
    "shot_distance": "shotDistance", "shot_result": "shotResult",
    "is_field_goal": "isFieldGoal", "score_home": "scoreHome",
    "score_away": "scoreAway", "points_total": "pointsTotal",
    "location": "location", "description": "description",
    "action_type": "actionType", "sub_type": "subType",
    "shot_value": "shotValue", "action_id": "actionId",
}


def load_config(env_path=ENV_PATH):
    """Environment first, then .env â€” so a deployment can override the file."""
    cfg = {}
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    cfg[key.strip()] = value.strip()
    for key in ("NBA_DB_HOST", "NBA_DB_PORT", "NBA_DB_NAME",
                "NBA_DB_USER", "NBA_DB_PASSWORD"):
        if os.environ.get(key):
            cfg[key] = os.environ[key]
    missing = [k for k in ("NBA_DB_HOST", "NBA_DB_NAME", "NBA_DB_USER",
                           "NBA_DB_PASSWORD") if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            f"veritabani ayarlari eksik: {missing}. .env dosyasina ekleyin "
            f"ya da ortam degiskeni olarak verin.")
    cfg.setdefault("NBA_DB_PORT", "3306")
    return cfg


def connect(cfg=None, timeout=30):
    """A plain connection â€” callers close it, or use `with closing(...)`."""
    cfg = cfg or load_config()
    return pymysql.connect(host=cfg["NBA_DB_HOST"], port=int(cfg["NBA_DB_PORT"]),
                           user=cfg["NBA_DB_USER"], password=cfg["NBA_DB_PASSWORD"],
                           database=cfg["NBA_DB_NAME"], connect_timeout=timeout,
                           read_timeout=600)


def _season_filter(seasons):
    """('2019_2020', ...) -> ('AND season IN (%s,%s)', params). Empty = all."""
    if not seasons:
        return "", []
    placeholders = ",".join(["%s"] * len(seasons))
    return f" AND g.season IN ({placeholders})", list(seasons)


def _team_side_frame(raw, column_map, prefix):
    """Rename a per-team table to UPPER_SNAKE and key it by (game_id, abbr)."""
    keep = ["game_id", "team_abbreviation"] + [c for c in column_map if c in raw.columns]
    out = raw[keep].rename(columns=column_map)
    out = out.rename(columns={"team_abbreviation": "abbr"})
    stats = [c for c in out.columns if c not in ("game_id", "abbr")]
    out[stats] = out[stats].apply(pd.to_numeric, errors="coerce")
    return out, stats


def _attach_side(master, side_frame, stats, side):
    """Merge one team's stats onto the game row as home_* / away_* columns."""
    renamed = side_frame.rename(columns={c: f"{side}_{c}" for c in stats})
    renamed = renamed.rename(columns={"abbr": f"{side}_team"})
    return master.merge(renamed, on=["game_id", f"{side}_team"], how="left")


def load_master_frame(seasons=None, conn=None, verbose=True):
    """One row per game, ready for FeatureEngineer.

    Four bulk queries rather than four reads per game: game metadata, team
    traditional box scores, team advanced box scores, and the precomputed
    rest/back-to-back columns.
    """
    own_connection = conn is None
    conn = conn or connect()
    try:
        where, params = _season_filter(seasons)

        games = pd.read_sql(
            "SELECT gs.game_id, gs.season, gs.game_date, gs.home_abbr, gs.away_abbr, "
            "       gs.home_team_id, gs.away_team_id, gs.home_pts, gs.away_pts "
            "FROM game_summary gs JOIN games g ON g.game_id = gs.game_id "
            f"WHERE 1=1{where} ORDER BY gs.game_date, gs.game_id",
            conn, params=params)
        if games.empty:
            return games

        master = games.rename(columns={
            "home_abbr": "home_team", "away_abbr": "away_team",
            "home_pts": "home_score", "away_pts": "away_score"})
        master["game_date"] = pd.to_datetime(master["game_date"])
        master["home_score"] = pd.to_numeric(master["home_score"], errors="coerce")
        master["away_score"] = pd.to_numeric(master["away_score"], errors="coerce")
        master["point_diff"] = master["home_score"] - master["away_score"]
        master["total_score"] = master["home_score"] + master["away_score"]
        master["home_win"] = (master["point_diff"] > 0).astype(int)

        ids = tuple(master["game_id"])
        placeholders = ",".join(["%s"] * len(ids))

        # `to` is a reserved word, hence the backticks around every column.
        trad_cols = ", ".join(f"`{c}`" for c in TRADITIONAL_MAP)
        trad_raw = pd.read_sql(
            f"SELECT game_id, team_abbreviation, {trad_cols} FROM box_team_traditional "
            f"WHERE game_id IN ({placeholders})", conn, params=list(ids))
        adv_cols = ", ".join(f"`{c}`" for c in ADVANCED_MAP)
        adv_raw = pd.read_sql(
            f"SELECT game_id, team_abbreviation, {adv_cols} FROM box_team_advanced "
            f"WHERE game_id IN ({placeholders})", conn, params=list(ids))

        trad, trad_stats = _team_side_frame(trad_raw, TRADITIONAL_MAP, "")
        adv, adv_stats = _team_side_frame(adv_raw, ADVANCED_MAP, "")
        for side in ("home", "away"):
            master = _attach_side(master, trad, trad_stats, side)
            master = _attach_side(master, adv, adv_stats, side)

        rest = pd.read_sql(
            "SELECT game_id, home_rest, away_rest, rest_diff, home_b2b, away_b2b "
            f"FROM training_games WHERE game_id IN ({placeholders})",
            conn, params=list(ids))
        master = master.merge(rest, on="game_id", how="left")
        for col in REST_COLUMNS:
            if col in master.columns:
                master[col] = pd.to_numeric(master[col], errors="coerce")

        master = master.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        if verbose:
            covered = int(master["home_PTS"].notna().sum()) if "home_PTS" in master else 0
            print(f"  db: {len(master)} mac, {master['season'].nunique()} sezon, "
                  f"takim box score {covered}/{len(master)}, "
                  f"rest {int(master['home_rest'].notna().sum())}/{len(master)}")
        return master
    finally:
        if own_connection:
            conn.close()


def attach_impact(master, cache_path=None):
    """Add home/away impact aggregates from the existing play-by-play cache.

    The cache is keyed by game_id and holds what compute_game_impact produced;
    a game missing from it gets NaN rather than a silent zero, because a zero
    impact is a real value and would be indistinguishable from "not computed".
    """
    import pickle

    cache_path = cache_path or os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            print(f"  [warn] impact cache okunamadi: {exc}")

    out = master.copy()
    home, away = [], []
    for gid in out["game_id"]:
        entry = cache.get(gid)
        if isinstance(entry, dict):
            home.append(entry.get("home_impact", np.nan))
            away.append(entry.get("away_impact", np.nan))
        else:
            home.append(np.nan)
            away.append(np.nan)
    out["home_impact_score_agg"] = home
    out["away_impact_score_agg"] = away
    out["impact_score_diff"] = out["home_impact_score_agg"] - out["away_impact_score_agg"]
    hit = int(out["home_impact_score_agg"].notna().sum())
    print(f"  impact cache: {hit}/{len(out)} mac ({hit/max(len(out),1):.1%})")
    return out


def load_pbp(game_ids, conn=None, chunk_size=200):
    """Play-by-play for many games at once, keyed by game_id.

    Yields {game_id: DataFrame} in the camelCase CSV schema. Fetched in chunks
    because the table holds 5.3M rows: pulling a whole season in one statement
    materialises hundreds of megabytes client-side for no benefit, while the
    (game_id, action_number) index makes chunked reads cheap.
    """
    own_connection = conn is None
    conn = conn or connect()
    try:
        ids = [str(g).zfill(10) for g in game_ids]
        columns = ", ".join(f"`{c}`" for c in PBP_MAP)
        for start in range(0, len(ids), chunk_size):
            batch = ids[start:start + chunk_size]
            placeholders = ",".join(["%s"] * len(batch))
            raw = pd.read_sql(
                f"SELECT game_id, {columns} FROM play_by_play "
                f"WHERE game_id IN ({placeholders}) ORDER BY game_id, action_number",
                conn, params=batch)
            if raw.empty:
                continue
            raw = raw.rename(columns=PBP_MAP)
            for gid, frame in raw.groupby("game_id", sort=False):
                yield gid, frame.drop(columns=["game_id"]).reset_index(drop=True)
    finally:
        if own_connection:
            conn.close()


def games_missing_impact(master, cache_path=None):
    """Game ids present in the frame but absent from the impact cache."""
    import pickle

    cache_path = cache_path or os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            cache = {}
    return [g for g in master["game_id"] if not isinstance(cache.get(g), dict)]


def available_seasons(conn=None):
    own = conn is None
    conn = conn or connect()
    try:
        return pd.read_sql(
            "SELECT season, COUNT(*) AS n FROM games GROUP BY season ORDER BY season",
            conn)
    finally:
        if own:
            conn.close()
