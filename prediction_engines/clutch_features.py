"""
Clutch-time team features, computed from play-by-play in the database.

WHY THIS AND NOT SOMETHING ELSE
    The month-by-month diagnosis said the model's weak months are weak for one
    reason: the games are closer. December and January decide 33% of games by
    six points or fewer against April's 27%, and monthly accuracy tracks that
    almost perfectly - correlation -0.787 with the close-game share, +0.879
    with how strongly the features correlate with the outcome at all. In those
    months the existing features carry 0.21 correlation with the margin where
    April's carry 0.29.

    So the gap is in close games, and nothing in the 190 features describes how
    a team behaves in one. Rolling form, season averages and impact aggregates
    are all whole-game quantities: a team that wins by 20 and a team that wins
    three one-possession games look similar to them.

WHY FROM PLAY-BY-PLAY AND NOT LeagueDashTeamClutch
    The endpoint returns a season-to-date aggregate, so using it without leaking
    would mean fetching an as-of snapshot per team per date. Play-by-play is
    already in the database with the clock, the period and the running score, so
    the same statistics can be derived for all 10,749 games with no API calls,
    and derived as-of by construction.

DEFINITION
    Clutch is the NBA's: the last five minutes of the fourth quarter or any
    overtime, with the score within five points. A game that is never close in
    that window contributes no clutch plays, which is itself informative - the
    share of a team's games that reach clutch is one of the features.

LEAKAGE
    Every column is a trailing value: for team T and game g it is the mean over
    T's games strictly BEFORE g (expanding mean, shift(1)). The clutch record of
    the game being predicted never enters its own features.
"""
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLUTCH_METRICS = [
    "clutch_net_per_play",   # (scored - allowed) per clutch play
    "clutch_fg_pct",         # field-goal percentage in clutch
    "clutch_tov_rate",       # turnovers per clutch play
    "clutch_ft_pct",         # free-throw percentage in clutch
    "clutch_share",          # share of prior games that reached clutch
    "clutch_win_pct",        # win rate in prior games that reached clutch
]
CLUTCH_FEATURE_COLS = ([f"home_{m}" for m in CLUTCH_METRICS]
                       + [f"away_{m}" for m in CLUTCH_METRICS]
                       + [f"diff_{m}" for m in CLUTCH_METRICS])

CLUTCH_SECONDS = 300.0
CLUTCH_MARGIN = 5


def clock_to_seconds(value):
    """'PT11M23.00S' -> 683.0. Returns NaN for anything unparseable."""
    if not isinstance(value, str) or "PT" not in value:
        return np.nan
    try:
        minutes, _, rest = value.partition("PT")[2].partition("M")
        return float(minutes) * 60.0 + float(rest.rstrip("S"))
    except (ValueError, AttributeError):
        return np.nan


def load_clutch_plays(conn, seasons=None):
    """Every play in the last five minutes of Q4/OT, with the running score.

    Filtering on period and the clock in SQL keeps the transfer small - the
    database lives on a phone, and pulling 5.3M rows to find 400k is what makes
    aggregate queries there time out. The margin filter happens after the score
    is forward-filled, because only scoring rows carry it.

    The clock is zero-padded ("PT00M12.30S"), so the pattern needs the leading
    zero: '^PT[0-4]M' matches nothing at all, which is a silent empty result
    rather than an error.
    """
    where, params = "", []
    if seasons:
        where = f" AND g.season IN ({','.join(['%s'] * len(seasons))})"
        params = list(seasons)
    sql = f"""
        SELECT p.game_id, p.action_number, p.period, p.clock, p.team_tricode,
               p.action_type, p.sub_type, p.description,
               p.score_home, p.score_away, p.shot_value
        FROM play_by_play p
        JOIN games g ON g.game_id = p.game_id
        WHERE p.period >= 4 AND p.clock REGEXP '^PT0[0-4]M'{where}
        ORDER BY p.game_id, p.action_number
    """
    return pd.read_sql(sql, conn, params=params)


def per_game_clutch(plays, games):
    """Per (game, team) clutch counting stats.

    `games` supplies home_team/away_team so a play can be attributed to the
    right side, and the winner so clutch win rate can be built.
    """
    if plays.empty:
        return pd.DataFrame()

    df = plays.copy()
    df["clock_seconds"] = df["clock"].map(clock_to_seconds)
    df = df[df["clock_seconds"] <= CLUTCH_SECONDS]

    # Only scoring rows carry the score; carry it forward so every play knows
    # the state it happened in.
    for col in ("score_home", "score_away"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("game_id")[col].ffill().fillna(0.0)
    df["margin"] = (df["score_home"] - df["score_away"]).abs()
    df = df[df["margin"] <= CLUTCH_MARGIN]
    if df.empty:
        return pd.DataFrame()

    desc = df["description"].fillna("")
    at = df["action_type"].fillna("")
    df["is_made"] = (at == "Made Shot").astype(int)
    df["is_missed"] = (at == "Missed Shot").astype(int)
    df["is_tov"] = (at == "Turnover").astype(int)
    df["is_ft"] = (at == "Free Throw").astype(int)
    df["ft_made"] = (df["is_ft"] & ~desc.str.contains("MISS", na=False)).astype(int)
    df["points"] = np.where(at == "Made Shot",
                            pd.to_numeric(df["shot_value"], errors="coerce").fillna(2.0),
                            np.where(df["ft_made"] == 1, 1.0, 0.0))

    agg = (df[df["team_tricode"].notna()]
           .groupby(["game_id", "team_tricode"])
           .agg(plays=("action_number", "size"),
                pts=("points", "sum"),
                made=("is_made", "sum"),
                missed=("is_missed", "sum"),
                tov=("is_tov", "sum"),
                ft=("is_ft", "sum"),
                ft_made=("ft_made", "sum"))
           .reset_index()
           .rename(columns={"team_tricode": "team"}))

    # Points allowed = the other side's points in the same game.
    total = agg.groupby("game_id")[["pts", "plays"]].sum().rename(
        columns={"pts": "game_pts", "plays": "game_plays"})
    agg = agg.merge(total, on="game_id")
    agg["pts_allowed"] = agg["game_pts"] - agg["pts"]

    agg["clutch_net_per_play"] = (agg["pts"] - agg["pts_allowed"]) / agg["plays"].clip(lower=1)
    attempts = agg["made"] + agg["missed"]
    agg["clutch_fg_pct"] = np.where(attempts > 0, agg["made"] / attempts.clip(lower=1), np.nan)
    agg["clutch_tov_rate"] = agg["tov"] / agg["plays"].clip(lower=1)
    agg["clutch_ft_pct"] = np.where(agg["ft"] > 0,
                                    agg["ft_made"] / agg["ft"].clip(lower=1), np.nan)

    meta = pd.concat([
        games[["game_id", "home_team", "home_win"]].rename(
            columns={"home_team": "team"}).assign(won=lambda d: d["home_win"]),
        games[["game_id", "away_team", "home_win"]].rename(
            columns={"away_team": "team"}).assign(won=lambda d: 1 - d["home_win"]),
    ])[["game_id", "team", "won"]]
    return agg.merge(meta, on=["game_id", "team"], how="left")


def build_trailing(per_game, games):
    """Trailing clutch form per (game, team), over that team's PRIOR games.

    Every team-game appears here, not only the clutch ones: a team whose recent
    games were all blowouts still needs a row, carrying its clutch history from
    further back and a low clutch_share.
    """
    schedule = pd.concat([
        games[["game_id", "game_date", "season", "home_team"]].rename(
            columns={"home_team": "team"}),
        games[["game_id", "game_date", "season", "away_team"]].rename(
            columns={"away_team": "team"}),
    ]).sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    joined = schedule.merge(per_game, on=["game_id", "team"], how="left")
    joined["reached_clutch"] = joined["plays"].notna().astype(float)
    joined["clutch_won"] = np.where(joined["plays"].notna(), joined["won"], np.nan)

    grp = joined.groupby("team", sort=False)
    out = joined[["game_id", "team"]].copy()
    for metric in ("clutch_net_per_play", "clutch_fg_pct",
                   "clutch_tov_rate", "clutch_ft_pct"):
        out[metric] = grp[metric].transform(
            lambda s: s.expanding(min_periods=1).mean().shift(1))
    out["clutch_share"] = grp["reached_clutch"].transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1))
    out["clutch_win_pct"] = grp["clutch_won"].transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1))
    return out


def add_clutch_features(master_df, conn, seasons=None, verbose=True):
    """Attach the 18 trailing clutch columns to a per-game frame."""
    df = master_df.copy()
    df = df.drop(columns=[c for c in CLUTCH_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    needed = {"game_id", "game_date", "home_team", "away_team", "home_win"}
    if not needed.issubset(df.columns):
        raise ValueError(f"eksik sutunlar: {sorted(needed - set(df.columns))}")

    seasons = seasons or sorted(df["season"].dropna().unique()) if "season" in df else None
    plays = load_clutch_plays(conn, seasons)
    if verbose:
        print(f"  clutch: {len(plays):,} aday satir cekildi")

    per_game = per_game_clutch(plays, df)
    if per_game.empty:
        for c in CLUTCH_FEATURE_COLS:
            df[c] = np.nan
        return df
    if verbose:
        reached = per_game["game_id"].nunique()
        print(f"  clutch: {reached:,}/{len(df):,} mac clutch'a girdi "
              f"({reached/len(df):.1%}), {len(per_game):,} takim-mac kaydi")

    trailing = build_trailing(per_game, df)
    for side in ("home", "away"):
        renamed = trailing.rename(
            columns={"team": f"{side}_team",
                     **{m: f"{side}_{m}" for m in CLUTCH_METRICS}})
        df = df.merge(renamed, on=["game_id", f"{side}_team"], how="left")
    for m in CLUTCH_METRICS:
        df[f"diff_{m}"] = df[f"home_{m}"] - df[f"away_{m}"]

    if verbose:
        cov = df[f"home_{CLUTCH_METRICS[0]}"].notna().mean()
        print(f"  clutch feature kapsami: {cov:.1%}")
    return df
