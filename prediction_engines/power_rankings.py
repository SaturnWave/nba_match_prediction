"""
Overall ratings and power rankings - weekly, monthly, season-to-date - built
from what the system already measures.

WHAT A RATING MEANS HERE
    Everything is in POINTS PER GAME against a league-average opponent on a
    neutral floor, because that is the one unit every piece of the system
    shares: the margin regressor predicts it, Massey solves for it, Elo can be
    converted to it (28 Elo points ~ 1 point of spread). An "overall" is just
    that number put on a 2K-style scale so a phone screen can show it as 91
    instead of +4.3.

THE THREE NUMBERS PER TEAM
    strength   where the team stands this season: a Massey solve over the
               season's games to date (margin = r_home - r_away + home edge,
               ridge-regularised, every game weighted equally). In the first
               weeks of a season, before ~100 games have been played, the
               solve cannot separate 30 teams, so Elo - which carries 75% of
               last season across the summer - stands in, converted to points.
    form       what the window added: a second ridge solve on the window's
               games only, of the RESIDUAL margin after the season strength
               and home edge are taken out. Three games of +20 do not make a
               team +20 better; the ridge shrinks a short window toward zero,
               which is what a sensible reader does by eye.
    power      strength + form (season window: strength alone, see below).
    For the week and month windows power = strength + form; for the season
    window power = strength, and the last four weeks' form is shown beside it,
    so a season OVR describes the season and a weekly OVR describes the week.
    Team OVR = 85 + 1.1 x power, clipped to 60..99: an average team is 85, a
    +12 team is 98, a -14 team is 70. Elo and the window record ride along.

PLAYERS
    A player is rated on his VALUE score (value_engine.py: every play priced
    in points against a possession baseline and weighted by how much the game
    still hung on it), averaged over the window. The raw impact score - the
    production count - is rated the same way as a second number, and the gap
    between the two is shown as the "empty stats" index: a player whose
    production overall is well above his value overall piles up plays that
    do not move the score (high-volume misses, conceded rebounds, garbage
    time). Where no value score exists for a game the production score stands
    in. Players who averaged fewer than 12 minutes or too few games are
    not rated; the rest are ranked, and the percentile is bent into a 2K-shaped
    curve: OVR = 65 + 34 x p^2.6, so the median rotation player is ~71, the
    top 10% are 91+, the top 3% are 96+, the best player in the window is 99. The roster OVR of a
    team is the minutes-weighted mean of its eight most-used players' season
    OVRs - who they have, as opposed to what they did (that is the power).

HONESTY
    These are descriptions of games already played, not forecasts, and the
    impact score is this project's own heuristic: it rewards what happens in
    the play-by-play (shots, stops, boards, turnovers, fouls, with context
    multipliers) and sees nothing of spacing, screening or defence that does
    not end in a block or a steal. Read a player OVR as "production", a team
    OVR as "results, opponent-adjusted".

Run:  py prediction_engines/power_rankings.py [--window week|month|season] [--key 2026-W10]
"""
import argparse
import importlib.util
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")

WINDOWS = ("week", "month", "season")
ELO_PER_POINT = 28.0           # FiveThirtyEight's NBA conversion: 28 Elo ~ 1 point of spread
MIN_SEASON_GAMES_FOR_MASSEY = 100
SEASON_RIDGE_ALPHA = 2.0       # every team has dozens of games; only a nudge
FORM_RIDGE_ALPHA = 6.0         # a week has three games; shrink hard
SEASON_FORM_DAYS = 28
TEAM_OVR_CENTER, TEAM_OVR_PER_POINT = 85.0, 1.1
TEAM_OVR_RANGE = (60, 99)
PLAYER_OVR_FLOOR, PLAYER_OVR_SPAN, PLAYER_OVR_CURVE = 65.0, 34.0, 2.6
PLAYER_MIN_MINUTES = 12.0
PLAYER_MIN_GAMES_CAP = {"week": 1, "month": 3, "season": 10}
ROSTER_TOP_N = 8
TREND_DAYS_SEASON = 28

TR_MONTHS = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ratings_module = _load_sibling("team_ratings")


# ---------------------------------------------------------------------------
#  Solvers
# ---------------------------------------------------------------------------
def solve_margins(block, teams, alpha, target="point_diff", with_home_edge=True):
    """Ridge solve of target = r_home - r_away (+ home edge). Ratings centred."""
    index = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    X = np.zeros((len(block), k + (1 if with_home_edge else 0)))
    for row, (home, away) in enumerate(zip(block["home_team"], block["away_team"])):
        X[row, index[home]] = 1.0
        X[row, index[away]] = -1.0
        if with_home_edge:
            X[row, k] = 1.0
    y = block[target].to_numpy(dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    coef = model.coef_
    ratings = coef[:k] - coef[:k].mean()
    home_edge = float(coef[k]) if with_home_edge else 0.0
    return dict(zip(teams, ratings)), home_edge


def team_ovr(power):
    lo, hi = TEAM_OVR_RANGE
    return int(round(min(hi, max(lo, TEAM_OVR_CENTER + TEAM_OVR_PER_POINT * power))))


def player_ovr_from_percentile(p):
    return int(round(PLAYER_OVR_FLOOR + PLAYER_OVR_SPAN * float(p) ** PLAYER_OVR_CURVE))


def _label_range(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s.month == e.month and s.year == e.year:
        return f"{s.day}–{e.day} {TR_MONTHS[e.month - 1]} {e.year}"
    if s.year == e.year:
        return f"{s.day} {TR_MONTHS[s.month - 1]} – {e.day} {TR_MONTHS[e.month - 1]} {e.year}"
    return f"{s.day} {TR_MONTHS[s.month - 1]} {s.year} – {e.day} {TR_MONTHS[e.month - 1]} {e.year}"


# ---------------------------------------------------------------------------
#  Engine
# ---------------------------------------------------------------------------
class RankingEngine:
    """Rankings for any week, month or season-to-date in the dataset.

    games         one row per game: game_id, game_date, season, home_team,
                  away_team, home_score, away_score, point_diff
    player_games  one row per player-game: player, name, team, game_id,
                  game_date, season, impact, minutes
    """

    def __init__(self, games, player_games):
        g = games[["game_id", "game_date", "season", "home_team", "away_team",
                   "home_score", "away_score", "point_diff"]].copy()
        g["game_date"] = pd.to_datetime(g["game_date"]).dt.normalize()
        self.games = g.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        self.teams = sorted(set(self.games["home_team"]) | set(self.games["away_team"]))
        pg = player_games.copy()
        pg["game_date"] = pd.to_datetime(pg["game_date"]).dt.normalize()
        if "minutes" not in pg.columns:
            pg["minutes"] = np.nan
        self.player_games = pg
        self.elo_log = self._elo_log()
        self._periods = {w: self._build_periods(w) for w in WINDOWS}

    # ----- precomputation -------------------------------------------------
    def _elo_log(self):
        """Post-game Elo for both teams of every game, in date order.

        Same constants and margin-of-victory damping as team_ratings, so the
        number shown here is the number the model's rating features used.
        """
        r = ratings_module
        ratings, prev_season, rows = {}, None, []
        for gid, date, season, home, away, margin in self.games[
                ["game_id", "game_date", "season", "home_team", "away_team", "point_diff"]
        ].itertuples(index=False):
            if season != prev_season and prev_season is not None:
                for t in ratings:
                    ratings[t] = r.ELO_START + r.ELO_SEASON_REGRESS * (ratings[t] - r.ELO_START)
            prev_season = season
            r_home = ratings.setdefault(home, r.ELO_START)
            r_away = ratings.setdefault(away, r.ELO_START)
            if np.isfinite(margin):
                spread = r_home + r.ELO_HOME_ADV - r_away
                expected = 1.0 / (1.0 + 10.0 ** (-spread / 400.0))
                actual = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
                edge = spread if margin > 0 else -spread
                delta = r.ELO_K * r._mov_multiplier(margin, max(edge, -300.0)) * (actual - expected)
                ratings[home] = r_home + delta
                ratings[away] = r_away - delta
            rows.append((date, gid, home, ratings[home]))
            rows.append((date, gid, away, ratings[away]))
        return pd.DataFrame(rows, columns=["game_date", "game_id", "team", "elo"])

    def _build_periods(self, window):
        g = self.games
        out = []
        if window == "season":
            for season, s in g.groupby("season", sort=True):
                out.append({"key": season, "label": season.replace("_", "-"), "season": season,
                            "start": s["game_date"].min(), "end": s["game_date"].max(),
                            "n_games": int(len(s))})
            return out
        if window == "month":
            keys = g["game_date"].dt.strftime("%Y-%m")
        else:
            iso = g["game_date"].dt.isocalendar()
            keys = iso["year"].astype(str) + "-W" + iso["week"].astype(int).map("{:02d}".format)
        for key, s in g.groupby(keys, sort=True):
            season = s["season"].mode().iloc[0]
            start, end = s["game_date"].min(), s["game_date"].max()
            label = (f"{TR_MONTHS[start.month - 1]} {start.year}" if window == "month"
                     else _label_range(start, end))
            out.append({"key": str(key), "label": label, "season": season,
                        "start": start, "end": end, "n_games": int(len(s))})
        return out

    def periods(self, window):
        return [{"key": p["key"], "label": p["label"], "season": p["season"],
                 "start": str(p["start"].date()), "end": str(p["end"].date()),
                 "n_games": p["n_games"]} for p in self._periods[window]]

    def _period(self, window, key):
        for p in self._periods[window]:
            if p["key"] == key:
                return p
        raise KeyError(f"{window} donemi yok: {key}")

    # ----- building blocks ------------------------------------------------
    def _season_games_to(self, season, end):
        g = self.games
        return g[(g["season"] == season) & (g["game_date"] <= end) & g["point_diff"].notna()]

    def _elo_asof(self, end):
        log = self.elo_log[self.elo_log["game_date"] <= end]
        return log.groupby("team", sort=False)["elo"].last().to_dict()

    def _strength(self, season, end):
        """(points vs average per team, home edge, source) as of `end`."""
        block = self._season_games_to(season, end)
        elo = self._elo_asof(end)
        if len(block) >= MIN_SEASON_GAMES_FOR_MASSEY:
            ratings, home_edge = solve_margins(block, self.teams, SEASON_RIDGE_ALPHA)
            return ratings, home_edge, "massey"
        from_elo = {t: (elo.get(t, ratings_module.ELO_START) - ratings_module.ELO_START) / ELO_PER_POINT
                    for t in self.teams}
        home_edge = ratings_module.ELO_HOME_ADV / ELO_PER_POINT
        return from_elo, home_edge, "elo"

    def _form(self, block, strength, home_edge):
        """Opponent-adjusted deviation from season strength over `block`."""
        if block.empty:
            return {t: 0.0 for t in self.teams}
        b = block.copy()
        b["residual"] = (b["point_diff"]
                         - (b["home_team"].map(strength) - b["away_team"].map(strength) + home_edge))
        dev, _ = solve_margins(b, self.teams, FORM_RIDGE_ALPHA, target="residual",
                               with_home_edge=False)
        played = set(b["home_team"]) | set(b["away_team"])
        return {t: (dev[t] if t in played else 0.0) for t in self.teams}

    def _records(self, block):
        rec = {t: {"games": 0, "wins": 0, "losses": 0, "pts_for": 0.0, "pts_against": 0.0,
                   "margin_sum": 0.0, "results": []} for t in self.teams}
        for date, home, away, hs, as_, margin in block[
                ["game_date", "home_team", "away_team", "home_score", "away_score", "point_diff"]
        ].itertuples(index=False):
            if not np.isfinite(margin):
                continue
            for team, mine, theirs, m in ((home, hs, as_, margin), (away, as_, hs, -margin)):
                r = rec[team]
                r["games"] += 1
                r["wins"] += int(m > 0)
                r["losses"] += int(m < 0)
                r["pts_for"] += float(mine)
                r["pts_against"] += float(theirs)
                r["margin_sum"] += float(m)
                r["results"].append("W" if m > 0 else "L")
        return rec

    def _player_table(self, start, end, min_games):
        """Per player over [start, end]: games, minutes, impact per game, team."""
        pg = self.player_games
        w = pg[(pg["game_date"] >= start) & (pg["game_date"] <= end)]
        if w.empty:
            return pd.DataFrame(columns=["player", "name", "team", "games", "minutes", "impact", "ovr"])
        if "value" not in w.columns:
            w = w.assign(value=np.nan)
        agg = (w.sort_values("game_date").groupby("player", sort=False)
               .agg(name=("name", "last"), team=("team", "last"), games=("impact", "size"),
                    minutes=("minutes", "mean"), impact=("impact", "mean"),
                    value=("value", "mean"), minutes_total=("minutes", "sum")).reset_index())
        qualified = agg[(agg["games"] >= min_games)
                        & (agg["minutes"].fillna(PLAYER_MIN_MINUTES) >= PLAYER_MIN_MINUTES)].copy()
        if qualified.empty:
            qualified["ovr"] = []
            qualified["ovr_prod"] = []
            qualified["empty"] = []
            return qualified
        # Two overalls: value (plays priced in points, leverage-weighted) is the
        # one that ranks; production (the raw impact score) is kept so the gap
        # between them - the "empty stats" index - can be shown.
        prod_pct = qualified["impact"].rank(pct=True, method="average")
        qualified["ovr_prod"] = [player_ovr_from_percentile(p) for p in prod_pct]
        has_value = qualified["value"].notna()
        if has_value.sum() >= 10:
            value_pct = qualified.loc[has_value, "value"].rank(pct=True, method="average")
            qualified["ovr"] = qualified["ovr_prod"]
            qualified.loc[has_value, "ovr"] = [player_ovr_from_percentile(p) for p in value_pct]
            qualified["ovr_source"] = np.where(has_value, "value", "production")
        else:
            qualified["ovr"] = qualified["ovr_prod"]
            qualified["ovr_source"] = "production"
        qualified["ovr"] = qualified["ovr"].astype(int)
        qualified["empty"] = qualified["ovr_prod"] - qualified["ovr"]
        return qualified

    def _min_games(self, window, block):
        cap = PLAYER_MIN_GAMES_CAP[window]
        if block.empty:
            return 1
        per_team = pd.concat([block["home_team"], block["away_team"]]).value_counts()
        return max(1, min(cap, int(0.4 * per_team.median())))

    # ----- the ranking ----------------------------------------------------
    def _rank_core(self, window, season, start, end):
        block = self.games[(self.games["game_date"] >= start) & (self.games["game_date"] <= end)
                           & self.games["point_diff"].notna()]
        strength, home_edge, source = self._strength(season, end)
        form_block = block
        if window == "season":
            form_block = block[block["game_date"] > end - pd.Timedelta(days=SEASON_FORM_DAYS)]
        form = self._form(form_block, strength, home_edge)
        elo = self._elo_asof(end)
        records = self._records(block)

        season_start = self._period("season", season)["start"]
        season_players = self._player_table(season_start, end, self._min_games("season",
                                            self._season_games_to(season, end)))
        roster_ovr = {}
        if not season_players.empty:
            for team, grp in season_players.groupby("team", sort=False):
                top = grp.sort_values("minutes_total", ascending=False).head(ROSTER_TOP_N)
                w = top["minutes_total"].fillna(1.0).clip(lower=1.0)
                roster_ovr[team] = float(np.average(top["ovr"], weights=w))

        rows = []
        for t in self.teams:
            r = records[t]
            # A season overall is the season; the week and month windows are
            # where current form belongs.
            power = strength[t] + (0.0 if window == "season" else form[t])
            rows.append({
                "team": t, "power": round(power, 2), "ovr": team_ovr(power),
                "strength": round(strength[t], 2), "form": round(form[t], 2),
                "elo": round(elo.get(t, ratings_module.ELO_START), 0),
                "games": r["games"], "wins": r["wins"], "losses": r["losses"],
                "margin_avg": round(r["margin_sum"] / r["games"], 1) if r["games"] else None,
                "pts_for": round(r["pts_for"] / r["games"], 1) if r["games"] else None,
                "pts_against": round(r["pts_against"] / r["games"], 1) if r["games"] else None,
                "last_results": r["results"][-5:],
                "roster_ovr": round(roster_ovr[t], 1) if t in roster_ovr else None,
            })
        rows.sort(key=lambda x: (-x["power"], -x["wins"], x["team"]))
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows, {"strength_source": source, "home_edge": round(home_edge, 2),
                      "n_games": int(len(block)), "start": str(start.date()), "end": str(end.date())}

    def _previous_period(self, window, period):
        if window == "season":
            end = period["end"] - pd.Timedelta(days=TREND_DAYS_SEASON)
            if end <= period["start"] + pd.Timedelta(days=14):
                return None
            return {"season": period["season"], "start": period["start"], "end": end}
        plist = self._periods[window]
        idx = next(i for i, p in enumerate(plist) if p["key"] == period["key"])
        if idx == 0 or plist[idx - 1]["season"] != period["season"]:
            return None
        prev = plist[idx - 1]
        return {"season": prev["season"], "start": prev["start"], "end": prev["end"]}

    @lru_cache(maxsize=512)
    def rank(self, window, key):
        if window not in WINDOWS:
            raise KeyError(f"bilinmeyen pencere: {window}")
        period = self._period(window, key)
        rows, meta = self._rank_core(window, period["season"], period["start"], period["end"])
        prev = self._previous_period(window, period)
        if prev is not None:
            prev_rows, _ = self._rank_core(window, prev["season"], prev["start"], prev["end"])
            prev_rank = {r["team"]: r["rank"] for r in prev_rows}
            prev_ovr = {r["team"]: r["ovr"] for r in prev_rows}
            for r in rows:
                r["rank_change"] = prev_rank[r["team"]] - r["rank"]
                r["ovr_change"] = r["ovr"] - prev_ovr[r["team"]]
        else:
            for r in rows:
                r["rank_change"] = None
                r["ovr_change"] = None
        meta.update({"window": window, "key": key, "label": period["label"],
                     "season": period["season"]})
        return {"meta": meta, "teams": rows}

    @lru_cache(maxsize=512)
    def players(self, window, key, team=None, limit=60):
        period = self._period(window, key)
        block = self.games[(self.games["game_date"] >= period["start"])
                           & (self.games["game_date"] <= period["end"])]
        table = self._player_table(period["start"], period["end"], self._min_games(window, block))
        season_start = self._period("season", period["season"])["start"]
        season_table = self._player_table(season_start, period["end"],
                                          self._min_games("season",
                                                          self._season_games_to(period["season"],
                                                                                period["end"])))
        season_ovr = dict(zip(season_table["player"], season_table["ovr"])) if not season_table.empty else {}
        if team:
            table = table[table["team"] == team]
        table = table.sort_values(["ovr", "impact"], ascending=False).head(limit)
        return [{"player": int(p) if isinstance(p, (int, np.integer)) else str(p),
                 "name": str(n), "team": str(t), "games": int(g),
                 "minutes": round(float(m), 1) if pd.notna(m) else None,
                 "impact": round(float(i), 1),
                 "value": round(float(v), 2) if pd.notna(v) else None,
                 "ovr": int(o), "ovr_prod": int(op), "empty": int(em),
                 "ovr_source": str(src),
                 "season_ovr": season_ovr.get(p)}
                for p, n, t, g, m, i, v, o, op, em, src in table[
                    ["player", "name", "team", "games", "minutes", "impact", "value",
                     "ovr", "ovr_prod", "empty", "ovr_source"]].itertuples(index=False)]


# ---------------------------------------------------------------------------
#  Standalone use
# ---------------------------------------------------------------------------
def player_games_with_minutes(dataset):
    """Impact per player-game joined to minutes from the local box-score cache."""
    pregame = _load_sibling("pregame_roster")
    impacts = pregame.load_impacts()
    meta = dataset[["game_id", "game_date", "season", "home_team", "away_team"]].copy()
    meta["game_id"] = meta["game_id"].astype(str).str.zfill(10)
    rows = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        for pid, imp in impacts.get(gid, {}).items():
            rows.append((pid, gid, gdate, season, imp))
    pg = pd.DataFrame(rows, columns=["player", "game_id", "game_date", "season", "impact"])
    listings = pregame.load_listings()
    names = pd.read_pickle(pregame.BOX_PLAYER_PATH)[["game_id", "player_id", "player_name"]]
    names["game_id"] = names["game_id"].astype(str).str.zfill(10)
    listings = listings.merge(names.rename(columns={"player_id": "person_id", "player_name": "name"}),
                              on=["game_id", "person_id"], how="left")
    pg = pg.merge(listings.rename(columns={"person_id": "player"})[["game_id", "player", "team",
                                                                     "minutes", "name"]],
                  on=["game_id", "player"], how="left")
    return attach_value(pg)


def attach_value(player_games, cache_path=os.path.join(OUTPUT_DIR, "value_cache_v1.pkl")):
    """Join the value score per (game, player) when value_engine has run."""
    if not os.path.exists(cache_path):
        player_games["value"] = np.nan
        return player_games
    import pickle
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    rows = [(gid, str(pid), e["value"]) for gid, players in cache.items()
            for pid, e in players.items()]
    values = pd.DataFrame(rows, columns=["game_id", "key", "value"])
    pg = player_games.copy()
    pg["key"] = pg["player"].astype(str)
    pg = pg.merge(values, on=["game_id", "key"], how="left").drop(columns=["key"])
    return pg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=WINDOWS, default="season")
    parser.add_argument("--key", default=None)
    parser.add_argument("--dataset", default=DATASET_PATH)
    args = parser.parse_args()
    dataset = pd.read_pickle(args.dataset)["dataset"]
    engine = RankingEngine(dataset, player_games_with_minutes(dataset))
    key = args.key or engine.periods(args.window)[-1]["key"]
    result = engine.rank(args.window, key)
    m = result["meta"]
    print(f"{m['label']} ({m['start']} -> {m['end']}, {m['n_games']} mac, guc kaynagi {m['strength_source']})")
    print(f"{'#':>2} {'takim':5} {'OVR':>4} {'guc':>6} {'sezon':>6} {'form':>6} {'elo':>5} {'W-L':>6} {'marj':>6} {'kadro':>6} {'degisim':>7}")
    for r in result["teams"]:
        ch = "" if r["rank_change"] is None else f"{r['rank_change']:+d}"
        print(f"{r['rank']:>2} {r['team']:5} {r['ovr']:>4} {r['power']:>+6.1f} {r['strength']:>+6.1f} "
              f"{r['form']:>+6.1f} {r['elo']:>5.0f} {r['wins']:>2}-{r['losses']:<3} "
              f"{(r['margin_avg'] if r['margin_avg'] is not None else float('nan')):>+6.1f} "
              f"{(r['roster_ovr'] if r['roster_ovr'] is not None else float('nan')):>6.1f} {ch:>7}")
    print("\nen iyi 12 oyuncu:")
    for p in engine.players(args.window, key, None, 12):
        print(f"  {p['ovr']:>3}  {p['name']:24} {p['team']:4} {p['games']:>3} mac  "
              f"{(p['minutes'] or 0):>5.1f} dk  impact {p['impact']:>6.1f}  sezon {p['season_ovr']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
