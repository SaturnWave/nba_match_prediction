"""
Opponent-adjusted team ratings — the information the current feature set lacks.

THE GAP THIS FILLS
    Of the 190 features the model uses today, 184 are self-referential: a
    team's rolling form, season averages, streaks and impact score are computed
    without any reference to WHO it played. Only the 6 matchup features know
    anything about the opponent. So a 9-1 run against the league's worst teams
    and a 9-1 run against contenders are the same number to the model.

    Every rating here answers the question those 184 features cannot: how good
    is this team once you account for the schedule it faced?

WHAT IS PRODUCED (18 columns, all as-of-date)
    rating_elo             Elo on point margin, margin-of-victory aware
    rating_elo_impact      Elo on the impact-score margin (sign only)
    rating_massey          ridge rating solved from point margins
    rating_massey_impact   ridge rating solved from impact-score margins
    rating_sos             mean pre-game Elo of the opponents faced this season
    rating_impact_vs_exp   trailing impact MINUS what the opponents usually allow
    each as home_/away_/diff_.

    The Massey ratings are the strongest form of the idea: every game becomes
    one equation, margin = rating_home - rating_away + home_advantage, and the
    whole schedule is solved at once. Elo is the sequential counterpart and is
    kept because it reacts faster to a team that has just changed.

LEAKAGE
    Elo and SOS use each game's PRE-game state, so they are safe by
    construction. The Massey systems are re-solved at month boundaries on games
    strictly before that month and applied to the month that follows.
    rating_impact_vs_exp averages a team's PRIOR adjusted games only. No column
    here can see the game it describes.

SEASONS
    Elo regresses toward the mean between seasons (rosters change), and the
    Massey window is a trailing block of games with exponential decay rather
    than a season reset, so a rating survives the calendar boundary instead of
    restarting blind.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --- Elo -------------------------------------------------------------------
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 100.0        # ~the league's long-run home edge in Elo points
ELO_SEASON_REGRESS = 0.75   # carry 75% of last season's deviation from 1500

# --- Massey ----------------------------------------------------------------
MASSEY_WINDOW = 1230        # trailing games ~= one full season
MASSEY_HALFLIFE = 400.0     # exponential decay within that window
MASSEY_ALPHA = 25.0         # ridge penalty; keeps thin schedules from blowing up

RATING_KINDS = ["elo", "elo_impact", "massey", "massey_impact", "sos", "impact_vs_exp"]
RATING_FEATURE_COLS = ([f"home_rating_{k}" for k in RATING_KINDS]
                       + [f"away_rating_{k}" for k in RATING_KINDS]
                       + [f"diff_rating_{k}" for k in RATING_KINDS])

IMPACT_VS_EXP_WINDOW = 10


def _mov_multiplier(margin, elo_diff_for_winner):
    """FiveThirtyEight's margin-of-victory damping.

    Without it a 40-point win moves a rating as much as forty 1-point wins, and
    with it alone a strong favourite blowing out a weak team would still be
    over-rewarded — hence the elo_diff term in the denominator.
    """
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_for_winner)


def _elo_pass(games, margin_col, use_mov):
    """One sequential sweep: record each game's PRE-game ratings, then update."""
    ratings = {}
    prev_season = None
    home_pre, away_pre = [], []

    for season, home, away, margin in zip(games["season"], games["home_team"],
                                          games["away_team"], games[margin_col]):
        if season != prev_season and prev_season is not None:
            for team in ratings:
                ratings[team] = ELO_START + ELO_SEASON_REGRESS * (ratings[team] - ELO_START)
        prev_season = season

        r_home = ratings.setdefault(home, ELO_START)
        r_away = ratings.setdefault(away, ELO_START)
        home_pre.append(r_home)
        away_pre.append(r_away)

        if not np.isfinite(margin):
            continue
        spread = r_home + ELO_HOME_ADV - r_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-spread / 400.0))
        actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

        step = ELO_K
        if use_mov:
            # The winner's own pre-game edge is what gets damped.
            edge = spread if margin > 0 else -spread
            step *= _mov_multiplier(margin, max(edge, -300.0))
        delta = step * (actual_home - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return np.asarray(home_pre), np.asarray(away_pre)


def _strength_of_schedule(games, home_elo, away_elo):
    """Mean pre-game Elo of the opponents a team has already met this season."""
    totals, counts = {}, {}
    prev_season = None
    home_sos, away_sos = [], []

    for i, (season, home, away) in enumerate(zip(games["season"], games["home_team"],
                                                 games["away_team"])):
        if season != prev_season:
            totals, counts = {}, {}
            prev_season = season
        home_sos.append(totals.get(home, 0.0) / counts[home] if counts.get(home) else np.nan)
        away_sos.append(totals.get(away, 0.0) / counts[away] if counts.get(away) else np.nan)
        totals[home] = totals.get(home, 0.0) + away_elo[i]
        counts[home] = counts.get(home, 0) + 1
        totals[away] = totals.get(away, 0.0) + home_elo[i]
        counts[away] = counts.get(away, 0) + 1

    return np.asarray(home_sos), np.asarray(away_sos)


def _solve_massey(block, teams, margin_col):
    """Ridge solve of margin = rating_home - rating_away + home_advantage.

    Recent games weigh more (exponential decay), and the ratings are centred so
    the scale is comparable across solves.
    """
    index = {t: i for i, t in enumerate(teams)}
    n, k = len(block), len(teams)
    X = np.zeros((n, k + 1))
    for row, (home, away) in enumerate(zip(block["home_team"], block["away_team"])):
        X[row, index[home]] = 1.0
        X[row, index[away]] = -1.0
        X[row, k] = 1.0
    y = block[margin_col].to_numpy(dtype=float)
    age = np.arange(n - 1, -1, -1, dtype=float)      # 0 = most recent game
    weights = 0.5 ** (age / MASSEY_HALFLIFE)

    model = Ridge(alpha=MASSEY_ALPHA, fit_intercept=False)
    model.fit(X, y, sample_weight=weights)
    ratings = model.coef_[:k]
    ratings = ratings - ratings.mean()
    return dict(zip(teams, ratings))


def _massey_pass(games, margin_col):
    """Re-solve at every month boundary on games strictly before that month."""
    months = games["game_date"].dt.to_period("M")
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    home_out = np.full(len(games), np.nan)
    away_out = np.full(len(games), np.nan)

    usable = games[margin_col].notna().to_numpy()
    for month in months.unique():
        in_month = (months == month).to_numpy()
        history = (months < month).to_numpy() & usable
        if history.sum() < 200:      # too thin to identify 30 team ratings
            continue
        block = games.loc[history].tail(MASSEY_WINDOW)
        ratings = _solve_massey(block, teams, margin_col)
        idx = np.flatnonzero(in_month)
        home_out[idx] = [ratings.get(t, 0.0) for t in games["home_team"].to_numpy()[idx]]
        away_out[idx] = [ratings.get(t, 0.0) for t in games["away_team"].to_numpy()[idx]]
    return home_out, away_out


def _impact_vs_expected(games):
    """Trailing (own impact - what this opponent usually allows), per team.

    Two passes. First, walk forward keeping each team's running mean of impact
    ALLOWED, and score every game against the opponent's value as it stood
    before tip-off. Then the feature for a given game is the mean of that
    team's PRIOR adjusted games, so the game itself never enters its own value.
    """
    allowed_sum, allowed_n = {}, {}
    per_team_history = {}
    home_out = np.full(len(games), np.nan)
    away_out = np.full(len(games), np.nan)

    home_imp = games["home_impact_score_agg"].to_numpy(dtype=float)
    away_imp = games["away_impact_score_agg"].to_numpy(dtype=float)

    for i, (home, away) in enumerate(zip(games["home_team"], games["away_team"])):
        for team, own, opponent in ((home, home_imp[i], away), (away, away_imp[i], home)):
            past = per_team_history.get(team, [])
            out = np.mean(past[-IMPACT_VS_EXP_WINDOW:]) if past else np.nan
            if team == home:
                home_out[i] = out
            else:
                away_out[i] = out

        for team, own, opponent in ((home, home_imp[i], away), (away, away_imp[i], home)):
            if not np.isfinite(own):
                continue
            baseline = (allowed_sum[opponent] / allowed_n[opponent]
                        if allowed_n.get(opponent) else np.nan)
            if np.isfinite(baseline):
                per_team_history.setdefault(team, []).append(own - baseline)

        for team, conceded in ((home, away_imp[i]), (away, home_imp[i])):
            if np.isfinite(conceded):
                allowed_sum[team] = allowed_sum.get(team, 0.0) + conceded
                allowed_n[team] = allowed_n.get(team, 0) + 1

    return home_out, away_out


def add_rating_features(master_df):
    """Attach the 18 opponent-adjusted rating columns, ordered by game date.

    The returned frame keeps master_df's original row order, so it can be
    merged or used in place without disturbing anything downstream.
    """
    df = master_df.copy()
    df = df.drop(columns=[c for c in RATING_FEATURE_COLS if c in df.columns], errors="ignore")
    required = {"game_date", "season", "home_team", "away_team", "point_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rating feature'lari icin eksik sutunlar: {sorted(missing)}")

    order = df.sort_values(["game_date", "game_id"]).index
    games = df.loc[order].reset_index(drop=True)
    if "impact_score_diff" not in games.columns:
        games["impact_score_diff"] = (games.get("home_impact_score_agg", np.nan)
                                      - games.get("away_impact_score_agg", np.nan))

    values = {}
    values["elo"] = _elo_pass(games, "point_diff", use_mov=True)
    values["elo_impact"] = _elo_pass(games, "impact_score_diff", use_mov=False)
    values["massey"] = _massey_pass(games, "point_diff")
    values["massey_impact"] = _massey_pass(games, "impact_score_diff")
    values["sos"] = _strength_of_schedule(games, values["elo"][0], values["elo"][1])
    values["impact_vs_exp"] = _impact_vs_expected(games)

    out = pd.DataFrame(index=order)
    for kind, (home_vals, away_vals) in values.items():
        out[f"home_rating_{kind}"] = home_vals
        out[f"away_rating_{kind}"] = away_vals
        out[f"diff_rating_{kind}"] = np.asarray(home_vals) - np.asarray(away_vals)

    covered = int(out["home_rating_massey"].notna().sum())
    print(f"  rating features: {covered}/{len(df)} mac icin Massey cozumu var "
          f"(ilk aylar tarihsiz kalir)")
    return df.join(out)
