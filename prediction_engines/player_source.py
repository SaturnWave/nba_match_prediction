"""
Per-player game logs and the trailing features a player-level simulator needs.

WHY A PLAYER LAYER
    The existing simulator draws two team scores from Poisson rates and reads
    the winner, the margin and the total off the draws. That structure is
    right - one generative process, every quantity a statistic of the same
    draws - but it stops at the team.

    Awards are a player-season question: MVP and DPOY are read off what an
    individual did over a season, and the only honest way to project them is to
    simulate the remaining games and count. That needs player rates, not team
    rates. The same machinery then answers "what does this player's line look
    like tonight" as a by-product.

WHAT THIS PROVIDES
    load_player_games   one row per (game, player) with the box line, minutes,
                        impact, and whether they were available
    add_trailing        the same leakage discipline as everywhere else: every
                        feature is the mean over that player's PRIOR games,
                        expanding or rolling with shift(1)
    opponent_strength   the defence a player is about to face, as of the date

MINUTES ARE THE HINGE
    Everything a player accumulates scales with minutes, and minutes are the
    most volatile part - a rotation change moves a line more than a shooting
    slump does. So minutes are modelled as their own quantity and the rate
    features are per-36 where that makes sense, letting the simulator combine a
    minutes draw with a per-minute rate instead of predicting totals directly.
"""
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Box columns worth carrying. `to` is a reserved word in SQL, hence backticks
# at the call site.
BOX_COLUMNS = ["min", "fgm", "fga", "fg3_m", "fg3_a", "ftm", "fta",
               "oreb", "dreb", "reb", "ast", "stl", "blk", "to", "pf", "pts",
               "plus_minus"]

COUNTING_STATS = ["pts", "reb", "ast", "stl", "blk", "to", "fg3_m", "fga", "fta"]
WINDOWS = [(3, "l3"), (10, "l10"), (25, "l25")]


def parse_minutes(value):
    """'34:12' -> 34.2. Blank, NaN and DNP rows come back as 0."""
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


def _all_seasons(conn):
    return pd.read_sql("SELECT DISTINCT season FROM games ORDER BY season",
                       conn)["season"].tolist()


def load_player_games(conn, seasons=None, verbose=True):
    """One row per (game, player): box line, availability, impact, game meta.

    Pulled one season at a time. The whole thing is 277k rows across a
    four-table join, and the database runs on a phone: asking for it in one
    statement gets the connection dropped mid-result. Per season it is ~30k
    rows and comes back reliably.
    """
    seasons = seasons or _all_seasons(conn)
    cols = ", ".join(f"b.`{c}`" for c in BOX_COLUMNS)
    frames = []
    for season in seasons:
        frames.append(pd.read_sql(f"""
            SELECT b.game_id, b.player_id AS person_id, b.player_name,
                   b.team_abbreviation AS team, b.comment, {cols},
                   g.season, gd.game_date,
                   gs.home_abbr, gs.away_abbr, gs.home_pts, gs.away_pts
            FROM box_player_traditional b
            JOIN games g       ON g.game_id  = b.game_id
            JOIN game_dates gd ON gd.game_id = b.game_id
            JOIN game_summary gs ON gs.game_id = b.game_id
            WHERE g.season = %s
        """, conn, params=(season,)))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["comment"] = df["comment"].fillna("")
    df["available"] = df["comment"].str.strip().eq("")
    df["minutes"] = df["min"].map(parse_minutes)
    for c in COUNTING_STATS + ["fgm", "ftm", "oreb", "dreb", "pf", "plus_minus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["opponent"] = np.where(df["team"] == df["home_abbr"],
                              df["away_abbr"], df["home_abbr"])
    df["is_home"] = (df["team"] == df["home_abbr"]).astype(int)
    if verbose:
        print(f"  oyuncu-mac: {len(df):,}, oyuncu: {df.person_id.nunique():,}, "
              f"oynayan {df.available.mean():.1%}")
    return df.sort_values(["person_id", "game_date", "game_id"]).reset_index(drop=True)


def attach_impact(player_games, conn, seasons=None):
    """Join per-game impact, keyed on person id (surnames do not identify).

    Chunked by season for the same reason load_player_games is.
    """
    seasons = seasons or _all_seasons(conn)
    frames = []
    for season in seasons:
        frames.append(pd.read_sql(
            "SELECT i.game_id, i.person_id, i.impact FROM player_game_impact i "
            "JOIN games g ON g.game_id = i.game_id WHERE g.season = %s",
            conn, params=(season,)))
    imp = pd.concat(frames, ignore_index=True)
    return player_games.merge(imp, on=["game_id", "person_id"], how="left")


def add_trailing(player_games, verbose=True):
    """Trailing per-player features, all shifted so a game never sees itself.

    Rates are per-36 minutes rather than per-game: minutes swing far more than
    productivity does, and the simulator wants to draw minutes and rate
    separately rather than have one absorb the other.
    """
    df = player_games.sort_values(["person_id", "game_date", "game_id"]).copy()
    played = df[df["minutes"] > 0].copy()

    for stat in COUNTING_STATS:
        played[f"{stat}_per36"] = played[stat] * 36.0 / played["minutes"].clip(lower=1.0)

    grp = played.groupby("person_id", sort=False)
    out_cols = {}
    for window, tag in WINDOWS:
        out_cols[f"trail_min_{tag}"] = grp["minutes"].transform(
            lambda s: s.rolling(window, min_periods=1).mean().shift(1))
        for stat in COUNTING_STATS:
            out_cols[f"trail_{stat}36_{tag}"] = grp[f"{stat}_per36"].transform(
                lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    if "impact" in played.columns:
        out_cols["trail_impact_l10"] = grp["impact"].transform(
            lambda s: s.rolling(10, min_periods=1).mean().shift(1))
    out_cols["career_games"] = grp.cumcount()
    out_cols["trail_min_std_l10"] = grp["minutes"].transform(
        lambda s: s.rolling(10, min_periods=2).std().shift(1))

    trailing = pd.DataFrame(out_cols, index=played.index)
    played = pd.concat([played, trailing], axis=1)

    keep = ["game_id", "person_id"] + list(out_cols)
    merged = df.merge(played[keep], on=["game_id", "person_id"], how="left")
    if verbose:
        cov = merged["trail_min_l10"].notna().mean()
        print(f"  trailing kapsam: {cov:.1%} "
              f"({len(out_cols)} feature/oyuncu-mac)")
    return merged


def opponent_strength(player_games, verbose=True):
    """What the opponent has been allowing, as of the date - one row per game/team.

    Built from the same frame rather than a separate query: a team's allowed
    totals are just its opponents' scored totals, and the schedule is already
    here.
    """
    df = player_games[player_games["minutes"] > 0]
    team_game = (df.groupby(["game_id", "game_date", "team", "opponent"], sort=False)
                 [["pts", "reb", "ast", "stl", "blk"]].sum().reset_index())

    # Rename to "what the OPPONENT allowed in this game"
    allowed = team_game.rename(columns={
        "team": "scorer", "opponent": "defender",
        **{c: f"allowed_{c}" for c in ["pts", "reb", "ast", "stl", "blk"]}})
    allowed = allowed.sort_values(["defender", "game_date", "game_id"])
    grp = allowed.groupby("defender", sort=False)
    for c in ["pts", "reb", "ast", "stl", "blk"]:
        allowed[f"opp_allowed_{c}_l10"] = grp[f"allowed_{c}"].transform(
            lambda s: s.rolling(10, min_periods=1).mean().shift(1))
    cols = ["game_id", "defender"] + [f"opp_allowed_{c}_l10"
                                      for c in ["pts", "reb", "ast", "stl", "blk"]]
    out = allowed[cols].rename(columns={"defender": "opponent"})
    if verbose:
        print(f"  rakip gucu: {len(out):,} takim-mac")
    return out


def build(conn, seasons=None, verbose=True):
    """Everything a player-level model needs, in one frame."""
    pg = load_player_games(conn, seasons, verbose)
    pg = attach_impact(pg, conn, seasons)
    pg = add_trailing(pg, verbose)
    opp = opponent_strength(pg, verbose)
    return pg.merge(opp, on=["game_id", "opponent"], how="left")
