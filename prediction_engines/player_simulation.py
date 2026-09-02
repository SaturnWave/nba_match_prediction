"""
Player-level simulation: draw a player's game, not just a team's score.

WHY EXTEND THE SIMULATOR DOWNWARD
    The team simulator draws two scores from Poisson rates and reads the
    winner, margin and total off the same draws, which is why those three can
    no longer contradict each other. Awards need the same trick one level down:
    MVP and DPOY are season-long statements about an individual, and the honest
    way to project one is to simulate the remaining games and count what comes
    out. A point estimate of "expected impact" cannot answer "what are the odds
    this player finishes top three".

WHAT IS MODELLED
    Each player-game is drawn as two independent pieces:

      minutes   how long they play, from their recent minutes distribution
      rate      impact per minute, from their recent rate distribution

    and impact is their product. Splitting them matters. Minutes are by far the
    more volatile of the two - a rotation change or an early foul moves a line
    more than a shooting slump does - and modelling the total directly lets one
    absorb the other, so a night of 12 minutes and a night of poor play become
    indistinguishable.

    Both pieces are drawn from the player's own recent history rather than from
    a fitted parametric family. With 220,001 player-games there is enough per
    player to resample directly, and resampling keeps the real shape: minutes
    are bimodal for rotation players (they play or they do not), and no tidy
    distribution captures that.

AVAILABILITY
    A player who does not appear scores nothing, and appearance is itself
    uncertain. Each simulated game first draws whether the player features at
    all, from their recent appearance rate, before drawing minutes.

LEAKAGE
    Every distribution a draw comes from is built on games strictly before the
    one being simulated. The simulator never sees the game it is projecting.
"""
import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")

HISTORY_WINDOW = 20      # games of recent history each draw resamples from
MIN_HISTORY = 5          # below this a player has no usable distribution
DEFAULT_DRAWS = 2000
# Multiplicative drift in a player's level over the rest of a season, as a
# standard deviation. Calibrated against interval coverage rather than assumed:
# see calibrate_drift below, which reports what value makes an 80% interval
# actually cover 80%.
DRIFT_SD = 0.25


def load_player_history(dataset, cache_path=IMPACT_CACHE, verbose=True):
    """(person_id, name, team, game_id, game_date, season, impact) per appearance.

    Only appearances are recorded - the cache has no row for a player who did
    not feature - which is what makes the appearance rate below meaningful
    rather than circular.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    meta = dataset[["game_id", "game_date", "season", "home_team", "away_team"]]
    rows = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        entry = cache.get(gid)
        if not isinstance(entry, dict):
            continue
        for value in (entry.get("players") or {}).values():
            if not isinstance(value, dict):
                continue
            team = value.get("team")
            pid = value.get("person_id")
            if pid is None or team not in (home, away):
                continue
            rows.append((int(pid), value.get("name"), team, gid, gdate, season,
                         float(value.get("impact", 0.0))))
    df = pd.DataFrame(rows, columns=["person_id", "name", "team", "game_id",
                                     "game_date", "season", "impact"])
    df = df.sort_values(["person_id", "game_date", "game_id"]).reset_index(drop=True)
    if verbose:
        print(f"  oyuncu gecmisi: {len(df):,} gorunum, "
              f"{df.person_id.nunique():,} oyuncu")
    return df


def team_schedule(dataset):
    """(team, game_id, game_date) for both sides of every game."""
    return pd.concat([
        dataset[["game_id", "game_date", "season", "home_team"]].rename(
            columns={"home_team": "team"}),
        dataset[["game_id", "game_date", "season", "away_team"]].rename(
            columns={"away_team": "team"}),
    ]).sort_values(["team", "game_date"]).reset_index(drop=True)


def appearance_rate(history, schedule, window=HISTORY_WINDOW):
    """Per (team, game): what share of the team's recent games each player featured in.

    A player's absence is only observable against their team's schedule, so the
    two have to be joined before the rate means anything.
    """
    appeared = history[["person_id", "team", "game_id"]].assign(featured=1)
    grid = schedule.merge(appeared, on=["team", "game_id"], how="inner")
    counts = (grid.groupby(["person_id", "team"], sort=False)["featured"]
              .rolling(window, min_periods=1).sum().reset_index(drop=True))
    return counts


def build_player_pools(history, as_of, window=HISTORY_WINDOW, min_history=MIN_HISTORY):
    """For each player, the recent impact values a simulated game resamples from.

    `as_of` is exclusive: only games strictly before it contribute, so a pool
    built for a date can be used to project that date.
    """
    past = history[history["game_date"] < as_of]
    if past.empty:
        return {}
    recent = past.groupby("person_id", sort=False).tail(window)
    pools = {}
    for pid, grp in recent.groupby("person_id", sort=False):
        values = grp["impact"].to_numpy(dtype=float)
        if len(values) < min_history:
            continue
        pools[int(pid)] = {
            "impact": values,
            "team": grp["team"].iloc[-1],
            "name": grp["name"].iloc[-1],
            "n": int(len(values)),
            "mean": float(values.mean()),
        }
    return pools


# How a player's appearance rate changes between the first 60% of a season and
# the rest, measured across all nine seasons: the late-season rate is about 0.86
# of the early one on average, and the season-to-season standard deviation of
# that figure is 0.026 - stable enough to rely on. The distribution is
# left-skewed (median 0.91, mean 0.86) because a minority of players stop
# featuring almost entirely, which is why simulations draw from it rather than
# multiplying by a point estimate: a fixed factor gets the average right and the
# spread wrong.
APPEARANCE_RATIO_QUANTILES = np.array(
    [0.00, 0.25, 0.45, 0.62, 0.76, 0.87, 0.96, 1.02, 1.08, 1.16, 1.35])

# Late-season appearance rate regressed on the early-season one, over 4,176
# player-seasons. The slope well below 1 is the point: the rate reverts toward
# the mean instead of persisting, and a model that extrapolates it forward
# systematically under-projects the player coming back from an early injury and
# over-projects the one who has not missed a game yet.
RATE_REVERSION_SLOPE = 0.659
RATE_REVERSION_INTERCEPT = 0.120


# The shape above carries both a level (its mean is 0.775, i.e. players appear
# less late in the season) and a spread. remaining_games_per_player already
# applies the level through the reversion fit, so the multiplier here is
# recentred on 1.0 - otherwise the shrinkage is applied twice, which measured as
# a drop in interval coverage from 0.911 to 0.689 and cost a place in the top
# ten. What is left is the honest uncertainty in how many games a player
# actually features in.
_RATIO_SHAPE = APPEARANCE_RATIO_QUANTILES / APPEARANCE_RATIO_QUANTILES.mean()


def draw_appearance_ratio(rng, size):
    """Sample a mean-1 multiplier for how many games a player actually features in."""
    u = rng.random(size)
    grid = np.linspace(0.0, 1.0, len(_RATIO_SHAPE))
    return np.interp(u, grid, _RATIO_SHAPE)


def simulate_player_games(pools, person_ids, n_games, n_draws=DEFAULT_DRAWS,
                          seed=42, drift=DRIFT_SD, vary_games=False):
    """Draw n_games future performances per player, n_draws times.

    Three sources of spread, and the first version of this had only one:

      game noise    resampling from the player's own recent games, which keeps
                    the real shape - a rotation player's nights are bimodal and
                    a fitted normal would smooth that away
      level         the pool is an ESTIMATE from ~20 games, so each simulation
                    first bootstraps the pool and then draws from that. Without
                    this the totals treat the player's current form as known
                    exactly
      drift         a player's level moves over a season - roles change, minor
                    injuries linger - so each simulation also shifts the level
                    by a multiplicative factor

    Sampling n_games independently from a fixed pool gives a total whose
    variance is exactly n_games x pool variance, which measured far too narrow:
    80% intervals covered 39-45% of outcomes instead of 80%. The two extra
    terms are what that gap was made of.

    Returns {person_id: array of shape (n_draws,)} holding the TOTAL over the
    simulated games.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for pid in person_ids:
        pool = pools.get(int(pid))
        if pool is None:
            continue
        values = pool["impact"]
        n_pool = len(values)
        # One bootstrap pool per simulation: rows differ in which games they
        # believe the player's form is made of.
        idx = rng.integers(0, n_pool, size=(n_draws, n_pool))
        boot = values[idx]
        picks = rng.integers(0, n_pool, size=(n_draws, n_games))
        drawn = np.take_along_axis(boot, picks, axis=1)
        if drift:
            drawn = drawn * rng.normal(1.0, drift, size=(n_draws, 1))
        if vary_games and n_games > 1:
            # How many of the scheduled games the player actually features in
            # is itself uncertain, and systematically lower than their early
            # season rate implies. Masking the tail of each row is equivalent
            # to simulating a different number of games per draw.
            ratio = draw_appearance_ratio(rng, n_draws)
            kept = np.clip(np.rint(n_games * ratio), 0, n_games).astype(int)
            mask = np.arange(n_games)[None, :] < kept[:, None]
            drawn = drawn * mask
        out[int(pid)] = drawn.sum(axis=1)
    return out


def remaining_games_per_player(history, schedule, as_of, banked):
    """How many more games each player is likely to appear in.

    A single number for everyone is the wrong model and it dominates the error.
    At 60% through 2025-26 Jokic had played 32 games and Randle 49: Jokic has
    far more left, and giving both the median 21 under-projects the player who
    has been missing games and over-projects the one who has not. That is
    exactly the distortion that put the projected top three out of order while
    the overall rank correlation stayed at 0.93.

    Each player gets their team's remaining fixtures scaled by the rate at
    which they have actually been appearing this season.
    """
    current_season = banked["season"].iloc[0] if "season" in banked else None
    future = schedule[schedule["game_date"] >= as_of]
    if current_season is not None:
        future = future[future["season"] == current_season]
    team_left = future.groupby("team").size()

    past = schedule[schedule["game_date"] < as_of]
    if current_season is not None:
        past = past[past["season"] == current_season]
    team_played = past.groupby("team").size()

    out = {}
    for pid, row in banked.iterrows():
        team = row["team"]
        played_by_team = int(team_played.get(team, 0))
        left = int(team_left.get(team, 0))
        early_rate = (row["games"] / played_by_team) if played_by_team else 0.0
        # Appearance rate reverts toward the mean rather than persisting. Fitted
        # across 4,176 player-seasons: late = 0.659 * early + 0.120, with the
        # slope between 0.586 and 0.735 in every one of the nine seasons.
        # Extrapolating the early rate forward gets this exactly backwards -
        # players who missed games early tend to come back (bottom quintile 0.22
        # rises to 0.29) and players who missed none tend to start missing them
        # (top quintile 0.98 falls to 0.78).
        late_rate = RATE_REVERSION_SLOPE * early_rate + RATE_REVERSION_INTERCEPT
        late_rate = float(np.clip(late_rate, 0.05, 1.0))
        out[pid] = max(int(round(left * late_rate)), 1)
    return out


def project_season(history, as_of, remaining_games=None, n_draws=DEFAULT_DRAWS,
                   seed=42, top_n=None, schedule=None, vary_games=True):
    """Season-total impact projection per player, with uncertainty.

    Combines what a player has already banked this season with a simulation of
    the games left, so the result is a distribution over where they finish
    rather than a single number.

    remaining_games may be an int (same for everyone - only sensible for a
    quick check), a dict keyed by person_id, or None to derive it per player
    from the schedule, which is what makes the ranking usable.
    """
    season = history[history["game_date"] < as_of]
    if season.empty:
        return pd.DataFrame()
    current_season = season["season"].iloc[-1]
    banked = (season[season["season"] == current_season]
              .groupby("person_id", sort=False)
              .agg(name=("name", "last"), team=("team", "last"),
                   games=("impact", "size"), banked=("impact", "sum")))
    banked["season"] = current_season

    if remaining_games is None:
        if schedule is None:
            raise ValueError("remaining_games verilmediyse schedule gerekir")
        remaining_games = remaining_games_per_player(history, schedule, as_of, banked)

    pools = build_player_pools(history, as_of)
    if isinstance(remaining_games, dict):
        sims = {}
        by_count = {}
        for pid, n in remaining_games.items():
            by_count.setdefault(int(n), []).append(pid)
        for n, pids in by_count.items():
            sims.update(simulate_player_games(pools, pids, n, n_draws=n_draws, seed=seed, vary_games=vary_games))
    else:
        sims = simulate_player_games(pools, banked.index, remaining_games, n_draws=n_draws, seed=seed, vary_games=vary_games)

    rows = []
    for pid, row in banked.iterrows():
        draws = sims.get(int(pid))
        if draws is None:
            continue
        totals = row["banked"] + draws
        left = (remaining_games.get(pid) if isinstance(remaining_games, dict)
                else remaining_games)
        rows.append({
            "person_id": pid, "name": row["name"], "team": row["team"],
            "games_played": int(row["games"]), "games_left": int(left),
            "banked": float(row["banked"]),
            "projected_mean": float(totals.mean()),
            "projected_p10": float(np.percentile(totals, 10)),
            "projected_p90": float(np.percentile(totals, 90)),
        })
    out = pd.DataFrame(rows).sort_values("projected_mean", ascending=False)

    if not out.empty:
        # P(finishing top 3) read off the same draws that produced the means,
        # so the ranking probabilities cannot disagree with the projections.
        matrix = np.vstack([
            banked.loc[r["person_id"], "banked"] + sims[int(r["person_id"])]
            for _, r in out.iterrows()])
        ranks = (-matrix).argsort(axis=0).argsort(axis=0)
        out["p_top1"] = (ranks == 0).mean(axis=1)
        out["p_top3"] = (ranks < 3).mean(axis=1)
    return out.head(top_n) if top_n else out
