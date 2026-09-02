"""
Who is missing, and how much they are worth.

WHY THIS IS DIFFERENT FROM THE FAMILIES THAT FAILED
    Opponent-adjusted ratings, rest/back-to-back, and clutch form were all
    SUMMARIES of data the model already had - re-expressions of results it could
    already see. Each one measured as new by a linear novelty test and each one
    added nothing, because a gradient-boosted model over 190 features can build
    those combinations itself.

    Availability is a different kind of thing: a fact about the game that is
    nowhere in the existing features. Nothing in rolling form, season averages,
    impact aggregates or matchup history says that a team's best player is not
    dressed tonight. box_player_traditional.comment records it - 17,721
    absences across nine seasons, with structured reasons ("DNP - Coach's
    Decision" 40,422, "DND - Injury/Illness" 4,549, "DND - Rest" 153).

THE LEAKAGE PROBLEM, AND HOW IT IS AVOIDED
    The comment column describes the game being predicted. Using it directly
    would be reading the team sheet after tip-off, and it would work
    spectacularly and mean nothing.

    So nothing here reads the current game's comments. Instead each feature
    describes what was true BEFORE it: which players the team has been without
    over its recent games, and what those players are worth. A star out for the
    last four games is very likely out for the fifth, and that inference uses
    only the past. The absence pattern is the signal; the current team sheet is
    never touched.

FEATURES (6 metrics x home/away/diff)
    avail_missing_impact   trailing impact of players absent in recent games
    avail_missing_share    share of the roster's impact that has been absent
    avail_star_out_rate    how often the team's top player has been out
    avail_injury_rate      injury/illness absences per game recently
    avail_churn            how much the available roster has been changing
    avail_depth_used       distinct players appearing per game recently
"""
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AVAIL_METRICS = [
    "avail_missing_impact",
    "avail_missing_share",
    "avail_star_out_rate",
    "avail_injury_rate",
    "avail_churn",
    "avail_depth_used",
]
AVAIL_FEATURE_COLS = ([f"home_{m}" for m in AVAIL_METRICS]
                      + [f"away_{m}" for m in AVAIL_METRICS]
                      + [f"diff_{m}" for m in AVAIL_METRICS])

LOOKBACK = 5          # games of absence history that predict tonight
INJURY_PATTERN = "Injur|Illness|Rest|Health"


def load_absences(conn, seasons=None):
    """One row per (game, player) with whether they were unavailable and why."""
    where, params = "", []
    if seasons:
        where = f" AND g.season IN ({','.join(['%s'] * len(seasons))})"
        params = list(seasons)
    sql = f"""
        SELECT b.game_id, b.player_id AS person_id, b.team_abbreviation AS team,
               b.comment
        FROM box_player_traditional b
        JOIN games g ON g.game_id = b.game_id
        WHERE 1=1{where}
    """
    df = pd.read_sql(sql, conn, params=params)
    df["comment"] = df["comment"].fillna("")
    df["absent"] = df["comment"].str.strip().ne("")
    df["injury"] = df["comment"].str.contains(INJURY_PATTERN, case=False, regex=True)
    return df


def player_value(player_impact):
    """Each player's career-to-date mean impact, as a time series per player.

    Expanding mean with shift(1), so the value at a given date reflects only
    that player's earlier games - the discipline every trailing feature here
    uses.

    Returned as (person_id, game_date, value) rather than keyed by game_id,
    because the point is to value players who are ABSENT. An absent player has
    no impact row for the game they missed, so joining on (game_id, person_id)
    finds nothing and every "missing impact" total silently comes out zero.
    The lookup has to be as-of the date instead.
    """
    pi = player_impact.sort_values(["person_id", "game_date"]).copy()
    pi["value"] = (pi.groupby("person_id")["impact"]
                   .transform(lambda s: s.expanding(min_periods=1).mean().shift(1)))
    return pi[["person_id", "game_date", "value"]].dropna(subset=["value"])


def value_as_of(valued, keys):
    """Look up each player's value as of a date, for players who may not have
    played that day. keys carries person_id and game_date."""
    left = keys.sort_values("game_date").reset_index(drop=True)
    right = valued.sort_values("game_date").reset_index(drop=True)
    merged = pd.merge_asof(left, right, on="game_date", by="person_id",
                           direction="backward", allow_exact_matches=True)
    return merged


def per_game_availability(absences, valued, games):
    """Per (game, team): who was missing and what they were worth.

    This describes each game as it happened. It becomes a leakage-free feature
    only after build_trailing shifts it, which is why nothing here is merged
    onto the frame directly.
    """
    dates = games[["game_id", "game_date"]].drop_duplicates()
    df = absences.merge(dates, on="game_id", how="left")
    df = df.dropna(subset=["game_date", "person_id"])
    df["person_id"] = df["person_id"].astype("int64")
    df = value_as_of(valued, df)

    grouped = df.groupby(["game_id", "team"], sort=False)
    out = grouped.apply(lambda g: pd.Series({
        "missing_impact": g.loc[g["absent"], "value"].fillna(0.0).sum(),
        "roster_impact": g["value"].fillna(0.0).sum(),
        "n_absent": float(g["absent"].sum()),
        "n_injury": float((g["absent"] & g["injury"]).sum()),
        "n_played": float((~g["absent"]).sum()),
        "star_out": float(
            g.loc[g["absent"], "value"].max() >= g["value"].max()
            if g["absent"].any() and g["value"].notna().any() else 0.0),
    }), include_groups=False).reset_index()

    out["missing_share"] = out["missing_impact"] / out["roster_impact"].clip(lower=1e-6)
    return out


def build_trailing(per_game, games):
    """Trailing absence pattern per (game, team), over that team's PRIOR games."""
    schedule = pd.concat([
        games[["game_id", "game_date", "home_team"]].rename(
            columns={"home_team": "team"}),
        games[["game_id", "game_date", "away_team"]].rename(
            columns={"away_team": "team"}),
    ]).sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    joined = schedule.merge(per_game, on=["game_id", "team"], how="left")
    grp = joined.groupby("team", sort=False)

    def trail(col):
        return grp[col].transform(
            lambda s: s.rolling(LOOKBACK, min_periods=1).mean().shift(1))

    out = joined[["game_id", "team"]].copy()
    out["avail_missing_impact"] = trail("missing_impact")
    out["avail_missing_share"] = trail("missing_share")
    out["avail_star_out_rate"] = trail("star_out")
    out["avail_injury_rate"] = trail("n_injury")
    out["avail_depth_used"] = trail("n_played")
    # Churn: how much the number of available players has been moving about.
    out["avail_churn"] = grp["n_played"].transform(
        lambda s: s.rolling(LOOKBACK, min_periods=2).std().shift(1))
    return out


def add_availability_features(master_df, conn, player_impact, seasons=None,
                              verbose=True):
    """Attach the 18 trailing availability columns.

    player_impact must carry person_id, game_date and impact - the local
    v4 cache flattened, or the player_game_impact table.
    """
    df = master_df.copy()
    df = df.drop(columns=[c for c in AVAIL_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    seasons = seasons or (sorted(df["season"].dropna().unique())
                          if "season" in df.columns else None)

    absences = load_absences(conn, seasons)
    if verbose:
        print(f"  availability: {len(absences):,} oyuncu-mac, "
              f"{int(absences.absent.sum()):,} yokluk "
              f"({absences.absent.mean():.1%}), "
              f"{int(absences.injury.sum()):,} sakat/dinlenme")
    if absences.empty:
        for c in AVAIL_FEATURE_COLS:
            df[c] = np.nan
        return df

    valued = player_value(player_impact)
    per_game = per_game_availability(absences, valued, df)
    trailing = build_trailing(per_game, df)

    for side in ("home", "away"):
        renamed = trailing.rename(
            columns={"team": f"{side}_team",
                     **{m: f"{side}_{m}" for m in AVAIL_METRICS}})
        df = df.merge(renamed, on=["game_id", f"{side}_team"], how="left")
    for m in AVAIL_METRICS:
        df[f"diff_{m}"] = df[f"home_{m}"] - df[f"away_{m}"]

    if verbose:
        cov = df[f"home_{AVAIL_METRICS[0]}"].notna().mean()
        print(f"  availability feature kapsami: {cov:.1%}")
    return df
