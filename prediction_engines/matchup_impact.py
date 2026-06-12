"""
Matchup-level impact module (player-vs-player / player-vs-team).

Built on the BoxScoreMatchupsV3 CSVs that the background retriever writes to
nba_data/<season>/<gid>/box_scores/<gid>box_score_matchups.csv (camelCase
schema, ~155-230 rows per game, one row per offensive-player / defender pair).

SCHEMA ORIENTATION (verified empirically, 2026-06-12):
    teamId / teamTricode on a matchup row belong to the OFFENSIVE player
    (personIdOff / firstNameOff / familyNameOff).  Cross-checking against the
    same games' box_score_traditional.csv (PLAYER_NAME -> TEAM_ABBREVIATION):
        game 0022500001: off-player team matched teamTricode 166/166 rows,
                         def-player team matched 0/166;
        game 0022500089: off 193/193, def 0/193;
        game 0022500140: off 232/232, def 0/232.
    The defender (personIdDef / firstNameDef / familyNameDef) therefore plays
    for the OPPOSING team, which this module derives as the other tricode that
    appears in the same game's matchup file.

SCALE NOTE: per-matchup partial possessions are counted per defender pair, so
an individual defender's points allowed per 100 partial possessions sits
around 20-26 (team-level ~110-120 per 100 possessions divided across the five
defenders on the floor).  Offensive per-100pp rates live on the same scale, so
edges (offense rate minus defense allowed rate) are directly comparable.

Everything that becomes a model feature is leakage-free: trailing values are
the mean over a player's PRIOR games only (sorted by date, expanding mean,
shift(1), min_periods=1), and team aggregates are computed as-of game date.

Only some games have a matchup CSV at any moment (the download is in flight);
all entry points are graceful when files are missing or zero files exist.
"""
import os
import glob
import pickle
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
MATCHUP_CACHE = os.path.join(PROJECT_ROOT, "matchup_cache_v1.pkl")

SEASON_DIR_RE = re.compile(r"^\d{4}_\d{4}$")

LONG_COLUMNS = ["game_id", "season", "game_date", "off_team", "def_team",
                "off_player", "def_player", "person_id_off", "person_id_def",
                "partial_poss", "player_points", "fgm", "fga",
                "turnovers", "blocks", "shooting_fouls", "ftm", "fta"]

MATCHUP_FEATURE_COLS = ["home_matchup_def_quality", "away_matchup_def_quality",
                        "home_matchup_edge", "away_matchup_edge",
                        "diff_matchup_def_quality", "diff_matchup_edge"]


# ===========================================================================
#  Loading the long (game, offPlayer, defPlayer) DataFrame
# ===========================================================================
def _detect_seasons(data_dir):
    if not os.path.isdir(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir)
                  if SEASON_DIR_RE.match(d) and os.path.isdir(os.path.join(data_dir, d)))


def _list_matchup_files(data_dir, seasons):
    files = []
    for season in seasons:
        pattern = os.path.join(data_dir, season, "*", "box_scores", "*box_score_matchups.csv")
        for fp in sorted(glob.glob(pattern)):
            files.append((season, fp))
    return files


def _load_date_map(game_ids_dir, seasons):
    """dict: game_id (zfill 10) -> game_date (Timestamp), from game_ids CSVs."""
    date_map = {}
    for season in seasons:
        fp = os.path.join(game_ids_dir, f"game_id_{season}.csv")
        if not os.path.exists(fp):
            continue
        try:
            ids = pd.read_csv(fp)
        except Exception:
            continue
        if not {"GAME_ID", "GAME_DATE"}.issubset(ids.columns):
            continue
        gids = ids["GAME_ID"].astype(str).str.zfill(10)
        dates = pd.to_datetime(ids["GAME_DATE"], errors="coerce")
        date_map.update(dict(zip(gids, dates)))
    return date_map


def _full_name(df, first_col, last_col):
    first = df[first_col].fillna("").astype(str).str.strip()
    last = df[last_col].fillna("").astype(str).str.strip()
    return (first + " " + last).str.strip()


def _gid_from_path(fp):
    return os.path.basename(fp).split("box_score")[0].zfill(10)


def _resolve_cache_path(cache_path, data_dir):
    """Use the shared cache file only for the default data dir; a custom
    data_dir gets no cache unless an explicit path is passed, so tests or
    alternate dirs can never clobber the main cache."""
    if cache_path != "auto":
        return cache_path
    if os.path.abspath(data_dir) == os.path.abspath(BASE_DATA_DIR):
        return MATCHUP_CACHE
    return None


def _read_one_matchup_csv(fp, season):
    """One matchup CSV -> normalised long rows, or None if unreadable/partial."""
    try:
        m = pd.read_csv(fp)
    except Exception:
        return None
    required = {"teamTricode", "firstNameOff", "familyNameOff",
                "firstNameDef", "familyNameDef", "partialPossessions", "playerPoints"}
    if m.empty or not required.issubset(m.columns):
        return None
    gid = os.path.basename(fp).split("box_score")[0].zfill(10)
    tris = [t for t in m["teamTricode"].dropna().unique() if isinstance(t, str)]
    other = {tris[0]: tris[1], tris[1]: tris[0]} if len(tris) == 2 else {}

    def num(col):
        if col in m.columns:
            return pd.to_numeric(m[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=m.index)

    out = pd.DataFrame({
        "game_id": gid,
        "season": season,
        "off_team": m["teamTricode"],
        "def_team": m["teamTricode"].map(other),     # defender is on the OTHER team
        "off_player": _full_name(m, "firstNameOff", "familyNameOff"),
        "def_player": _full_name(m, "firstNameDef", "familyNameDef"),
        "person_id_off": num("personIdOff"),
        "person_id_def": num("personIdDef"),
        "partial_poss": num("partialPossessions"),
        "player_points": num("playerPoints"),
        "fgm": num("matchupFieldGoalsMade"),
        "fga": num("matchupFieldGoalsAttempted"),
        "turnovers": num("matchupTurnovers"),
        "blocks": num("matchupBlocks"),
        "shooting_fouls": num("shootingFouls"),
        "ftm": num("matchupFreeThrowsMade"),
        "fta": num("matchupFreeThrowsAttempted"),
    })
    return out[(out["off_player"] != "") & (out["def_player"] != "")]


def load_matchup_data(data_dir=BASE_DATA_DIR, seasons=None, game_ids_dir=GAME_IDS_DIR,
                      cache_path="auto", use_cache=True):
    """Long DataFrame over all games that have a matchup CSV on disk.

    One row per (game, offensive player, defender) with game_date merged from
    game_ids/game_id_<season>.csv. Cached to `cache_path` keyed by a
    {file path -> size} index, so while the background download keeps adding
    files a re-run only parses the new/changed CSVs (a partially-written file
    is re-parsed once its size changes). cache_path=None disables caching;
    the default "auto" uses the shared cache only for the default data_dir.
    """
    cache_path = _resolve_cache_path(cache_path, data_dir)
    if seasons is None:
        seasons = _detect_seasons(data_dir)
    files = _list_matchup_files(data_dir, seasons)
    index = {}
    for _season, fp in files:
        try:
            index[fp] = os.path.getsize(fp)
        except OSError:
            pass

    cached_index, cached_df = {}, None
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            cached_index = cached.get("file_index") or {}
            cached_df = cached.get("long_df")
        except Exception:
            cached_index, cached_df = {}, None
    if cached_df is not None and cached_index == index:
        return cached_df

    # reuse cached rows for files whose size is unchanged; (re)parse the rest
    frames = []
    reuse_fps = {fp for fp, size in index.items() if cached_index.get(fp) == size}
    if cached_df is not None and reuse_fps and not cached_df.empty:
        reuse_gids = {_gid_from_path(fp) for fp in reuse_fps}
        base = cached_df[cached_df["game_id"].isin(reuse_gids)]
        if not base.empty:
            frames.append(base[LONG_COLUMNS])
    else:
        reuse_fps = set()

    new_frames = []
    for season, fp in files:
        if fp in reuse_fps or fp not in index:
            continue
        rows = _read_one_matchup_csv(fp, season)
        if rows is not None and not rows.empty:
            new_frames.append(rows)
    if new_frames:
        new_df = pd.concat(new_frames, ignore_index=True)
        date_map = _load_date_map(game_ids_dir, seasons)
        new_df["game_date"] = new_df["game_id"].map(date_map)
        n_undated = int(new_df["game_date"].isna().sum())
        if n_undated:
            print(f"  [warn] dropping {n_undated} matchup rows with no game date")
            new_df = new_df.dropna(subset=["game_date"])
        frames.append(new_df[LONG_COLUMNS])

    if not frames:
        long_df = pd.DataFrame(columns=LONG_COLUMNS)
    else:
        long_df = (pd.concat(frames, ignore_index=True)
                   .sort_values(["game_date", "game_id"]).reset_index(drop=True))

    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"file_index": index, "seasons": list(seasons),
                             "long_df": long_df}, f)
        except Exception as exc:
            print(f"  [warn] could not write matchup cache: {exc}")
    return long_df


# ===========================================================================
#  Per-player per-game aggregation + leakage-free trailing values
# ===========================================================================
def _add_trailing(table, player_col, rate_col, poss_col="partial_poss"):
    """Leakage-free trailing columns: for each (player, game) the mean of the
    rate over that player's PRIOR games only (expanding mean, shift(1)), plus
    the cumulative prior partial possessions (used as a reliability weight)."""
    table = table.sort_values([player_col, "game_date", "game_id"]).reset_index(drop=True)
    grp = table.groupby(player_col, sort=False)
    table[f"trail_{rate_col}"] = grp[rate_col].transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1))
    table["trail_partial_poss"] = grp[poss_col].transform(lambda s: s.cumsum().shift(1))
    return table


def build_defender_game_table(long_df):
    """Per-DEFENDER per-game aggregates and trailing defender quality.

    pts_allowed_per100pp = 100 * points allowed to direct assignments /
    partial possessions defended (~20-26 for a typical NBA defender; lower is
    better defense). trail_pts_allowed_per100pp is the leakage-free version.
    """
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    g = (long_df.groupby(["def_player", "def_team", "game_id", "game_date", "season"],
                         dropna=False)
         .agg(points_allowed=("player_points", "sum"),
              partial_poss=("partial_poss", "sum"),
              fgm_allowed=("fgm", "sum"),
              fga_allowed=("fga", "sum"),
              turnovers_forced=("turnovers", "sum"),
              shooting_fouls=("shooting_fouls", "sum"))
         .reset_index())
    g["pts_allowed_per100pp"] = 100.0 * g["points_allowed"] / np.maximum(g["partial_poss"], 1.0)
    return _add_trailing(g, "def_player", "pts_allowed_per100pp")


def build_offense_game_table(long_df):
    """Per-OFFENSIVE-player per-game scoring rate and its trailing version.

    pts_per100pp = 100 * points scored / partial possessions faced.
    """
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    g = (long_df.groupby(["off_player", "off_team", "game_id", "game_date", "season"],
                         dropna=False)
         .agg(points=("player_points", "sum"),
              partial_poss=("partial_poss", "sum"),
              fgm=("fgm", "sum"),
              fga=("fga", "sum"))
         .reset_index())
    g["pts_per100pp"] = 100.0 * g["points"] / np.maximum(g["partial_poss"], 1.0)
    return _add_trailing(g, "off_player", "pts_per100pp")


def _weighted_game_team_quality(table, team_col, trail_col, out_col):
    """Per (game_id, team) partial-possession-weighted mean of a trailing metric.

    Weights are each player's PRIOR cumulative partial possessions, so a
    high-minute veteran counts more than a debutant. Falls back to the plain
    mean when no player has prior possessions; NaN when nobody has history.
    """
    t = table.copy()
    valid = t[trail_col].notna()
    w = t["trail_partial_poss"].where(valid, 0.0).fillna(0.0)
    t["_w"] = w
    t["_wx"] = w * t[trail_col].fillna(0.0)
    g = (t.groupby(["game_id", team_col])
         .agg(_w=("_w", "sum"), _wx=("_wx", "sum"), _m=(trail_col, "mean"))
         .reset_index())
    g[out_col] = np.where(g["_w"] > 0, g["_wx"] / np.where(g["_w"] > 0, g["_w"], 1.0), g["_m"])
    return g[["game_id", team_col, out_col]]


# ===========================================================================
#  MatchupImpact — forward-looking matchup edge + training features
# ===========================================================================
class MatchupImpact:
    """Holds the long matchup DataFrame plus precomputed per-player tables and
    serves both the forward-looking query (get_current_matchup_impact) and the
    fast vectorised training-feature builder (add_matchup_features)."""

    def __init__(self, data_dir=BASE_DATA_DIR, game_ids_dir=GAME_IDS_DIR,
                 seasons=None, cache_path="auto"):
        self.data_dir = data_dir
        self.game_ids_dir = game_ids_dir
        self.seasons = seasons
        self.cache_path = cache_path
        self.long_df = None
        self.defender_games = None
        self.offense_games = None

    def load(self, use_cache=True):
        self.long_df = load_matchup_data(self.data_dir, self.seasons,
                                         game_ids_dir=self.game_ids_dir,
                                         cache_path=self.cache_path, use_cache=use_cache)
        self.defender_games = build_defender_game_table(self.long_df)
        self.offense_games = build_offense_game_table(self.long_df)
        return self

    def _ensure_loaded(self):
        if self.long_df is None:
            self.load()

    @staticmethod
    def _pooled_rate(points_sum, poss_sum):
        if poss_sum > 0:
            return 100.0 * points_sum / poss_sum
        return None

    def get_current_matchup_impact(self, off_players, def_players, as_of_date, def_team=None):
        """Forward-looking expected scoring edge for an offense vs a defense.

        Uses only matchup history strictly before `as_of_date`. Per offensive
        player the expected per-100pp scoring rate is taken from, in order of
        preference:
          1. 'direct'   — history vs these exact defenders (pooled, i.e.
                          weighted by historical partialPossessions);
          2. 'vs_team'  — history vs the defending team (`def_team`, inferred
                          from the defenders' history when not given);
          3. 'trailing' — the player's overall trailing scoring rate.
        The defense baseline is the defenders' pooled pts_allowed_per100pp
        (league average before `as_of_date` if they have no history).

        Returns dict: edge_per100pp (expected offense rate minus defense
        baseline), expected_off_per100pp, def_quality_per100pp, def_team,
        coverage counts per source, and per_player detail.
        """
        self._ensure_loaded()
        off_players = list(off_players)
        def_players = list(def_players)
        out = {"edge_per100pp": 0.0, "expected_off_per100pp": None,
               "def_quality_per100pp": None, "def_team": def_team,
               "n_off_players": len(off_players),
               "coverage": {"direct": 0, "vs_team": 0, "trailing": 0, "none": 0},
               "per_player": {}}
        if self.long_df is None or self.long_df.empty:
            out["coverage"]["none"] = len(off_players)
            return out
        as_of = pd.to_datetime(as_of_date)
        ld = self.long_df[self.long_df["game_date"] < as_of]
        dg = self.defender_games[self.defender_games["game_date"] < as_of]
        og = self.offense_games[self.offense_games["game_date"] < as_of]

        # defense baseline: pooled pts allowed per 100pp of these defenders
        dhist = dg[dg["def_player"].isin(def_players)]
        def_quality = self._pooled_rate(dhist["points_allowed"].sum(),
                                        dhist["partial_poss"].sum())
        if def_quality is None and not dg.empty:   # league average fallback
            def_quality = self._pooled_rate(dg["points_allowed"].sum(),
                                            dg["partial_poss"].sum())
        if def_team is None and not dhist.empty:
            modes = dhist["def_team"].dropna().mode()
            if not modes.empty:
                def_team = modes.iloc[0]
        out["def_team"] = def_team

        for p in off_players:
            rate, source = None, "none"
            h = ld[(ld["off_player"] == p) & ld["def_player"].isin(def_players)]
            rate = self._pooled_rate(h["player_points"].sum(), h["partial_poss"].sum())
            if rate is not None:
                source = "direct"
            elif def_team is not None:
                h = ld[(ld["off_player"] == p) & (ld["def_team"] == def_team)]
                rate = self._pooled_rate(h["player_points"].sum(), h["partial_poss"].sum())
                if rate is not None:
                    source = "vs_team"
            if rate is None:
                h = og[og["off_player"] == p]
                rate = self._pooled_rate(h["points"].sum(), h["partial_poss"].sum())
                if rate is not None:
                    source = "trailing"
            out["coverage"][source] += 1
            out["per_player"][p] = {"expected_per100pp": rate, "source": source}

        rates = [d["expected_per100pp"] for d in out["per_player"].values()
                 if d["expected_per100pp"] is not None]
        if rates and def_quality is not None:
            out["expected_off_per100pp"] = float(np.mean(rates))
            out["def_quality_per100pp"] = float(def_quality)
            out["edge_per100pp"] = float(np.mean(rates) - def_quality)
        return out

    def add_matchup_features(self, master_df):
        """Add team-level matchup features per game, as-of game date.

        Adds (NaN where no matchup history exists — caller fillna(0)):
          home/away_matchup_def_quality — game-roster defenders' trailing
            pts_allowed_per100pp, weighted by prior partial possessions;
          home/away_matchup_edge — offense roster trailing scoring per100pp
            minus the opposing defense quality;
          diff_matchup_def_quality / diff_matchup_edge — home minus away.

        Fully vectorised: the per-player trailing tables are precomputed once
        and merged by (game_id, team) — no per-row loops.
        """
        self._ensure_loaded()
        df = master_df.copy()
        df = df.drop(columns=[c for c in MATCHUP_FEATURE_COLS if c in df.columns],
                     errors="ignore")
        empty = (self.defender_games is None or self.defender_games.empty
                 or not {"game_id", "home_team", "away_team"}.issubset(df.columns))
        if empty:
            for c in MATCHUP_FEATURE_COLS:
                df[c] = np.nan
            return df

        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
        defq = _weighted_game_team_quality(self.defender_games, "def_team",
                                           "trail_pts_allowed_per100pp", "quality")
        offq = _weighted_game_team_quality(self.offense_games, "off_team",
                                           "trail_pts_per100pp", "quality")
        df = df.merge(defq.rename(columns={"def_team": "home_team",
                                           "quality": "home_matchup_def_quality"}),
                      on=["game_id", "home_team"], how="left")
        df = df.merge(defq.rename(columns={"def_team": "away_team",
                                           "quality": "away_matchup_def_quality"}),
                      on=["game_id", "away_team"], how="left")
        df = df.merge(offq.rename(columns={"off_team": "home_team",
                                           "quality": "_home_matchup_off_quality"}),
                      on=["game_id", "home_team"], how="left")
        df = df.merge(offq.rename(columns={"off_team": "away_team",
                                           "quality": "_away_matchup_off_quality"}),
                      on=["game_id", "away_team"], how="left")
        df["home_matchup_edge"] = df["_home_matchup_off_quality"] - df["away_matchup_def_quality"]
        df["away_matchup_edge"] = df["_away_matchup_off_quality"] - df["home_matchup_def_quality"]
        df["diff_matchup_def_quality"] = (df["home_matchup_def_quality"]
                                          - df["away_matchup_def_quality"])
        df["diff_matchup_edge"] = df["home_matchup_edge"] - df["away_matchup_edge"]
        df = df.drop(columns=["_home_matchup_off_quality", "_away_matchup_off_quality"])
        n_cov = int(df["home_matchup_def_quality"].notna().sum())
        print(f"  matchup features: {n_cov}/{len(df)} games covered "
              f"({len(self.defender_games['game_id'].unique())} matchup games on disk)")
        return df


# ===========================================================================
#  Module-level convenience wrappers (shared default instance)
# ===========================================================================
_DEFAULT_INSTANCE = None


def _get_default_instance(data_dir=BASE_DATA_DIR):
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None or _DEFAULT_INSTANCE.data_dir != data_dir:
        _DEFAULT_INSTANCE = MatchupImpact(data_dir=data_dir).load()
    return _DEFAULT_INSTANCE


def get_current_matchup_impact(off_players, def_players, as_of_date,
                               def_team=None, data_dir=BASE_DATA_DIR):
    """Module-level wrapper; see MatchupImpact.get_current_matchup_impact."""
    return _get_default_instance(data_dir).get_current_matchup_impact(
        off_players, def_players, as_of_date, def_team=def_team)


def add_matchup_features(master_df, data_dir=BASE_DATA_DIR):
    """Module-level wrapper; see MatchupImpact.add_matchup_features."""
    return _get_default_instance(data_dir).add_matchup_features(master_df)
