"""
Value score: what a player's plays were worth in points, not how many there were.

WHY A SECOND SCORE
    The impact score (predict_2025_2026.compute_game_impact) counts production
    with context flavour: a made shot is +2 or +3 times a difficulty factor, a
    rebound +0.6 to +0.9 plus bonuses, a steal +1.4, a turnover about -1. Three
    things make it a stat-padder's friend, and 2017-18 shows all three at once -
    Westbrook (25/10/10 on .524 true shooting) scores 40.7 a game, Curry (27 on
    .675) 32.0, and the top of the table is big men who rebound:

      * a missed shot costs nothing. Twenty-one attempts a game at 45% earn
        every make and pay for no miss, so volume is rewarded and efficiency
        is invisible;
      * a defensive rebound is worth about a third of a basket, although most
        of them are the expected end of an opponent's miss - the team gets it
        whoever's hands it lands in;
      * assists and free throws do not exist, and nothing is discounted when
        the game is already decided.

    Across 167 regulars that season the impact score per 36 minutes correlates
    0.14 with the player's own on-court plus-minus per 36. That is the number
    this module is trying to raise.

HOW VALUE IS COUNTED
    Every event is priced by how it moves expected points against a baseline
    of BASELINE points per possession (1.00: a little below the league's ~1.08,
    so taking shots at ordinary efficiency is worth a little, not nothing):
        made shot        points - BASELINE                 (+1.0 for a two, +2.0 for a three)
        missed shot      -BASELINE x (1 - OREB_SHARE)       (-0.74: the possession is usually gone)
        free throw       +1 made, -0.44 x BASELINE per attempt
        assist           +0.35 to the passer, read from the description
        turnover         -BASELINE - TRANSITION             (-1.15)
        steal            +BASELINE + TRANSITION             (+1.15)
        offensive board  +BASELINE x (1 - OREB_SHARE)       (+0.74: a possession recovered)
        defensive board  +BASELINE x OREB_SHARE             (+0.26: a chance denied)
        block            +0.6
        foul             -0.3 personal, -0.6 shooting, 0 offensive (its turnover row pays)
    No style bonuses: a step-back three and a wide-open one are both worth
    +2.0, because they both put three points on the board.

LEVERAGE
    Each event is multiplied by how much the game still hung on it: 1.0 while
    the game is live, sliding to 0.3 when it is 25 points apart in the fourth,
    1.25 in the last five minutes of a one-possession-to-five-point game. A
    stat line built in garbage time shrinks; one built when it mattered does
    not.

    That is the right weighting for crediting a PLAYER and the wrong one for
    a TEAM feature: how far a team won by predicts its next game, garbage time
    included. So the player ratings use the weighted run (default) and the
    model's impact cache is built from --no-leverage (impact_v5.py), the same
    pricing with every weight at 1.0. Measured: weighted values as model
    features cost Brier +0.003 and raised winner-vs-margin contradictions from
    8.9% to 12.4%; unweighted values were indistinguishable from the old score.

WHAT IT IS NOT
    Still a box-plus-context score. It does not see screens, spacing, closeouts
    or who a player guarded; those live in the tracking and matchup tables and
    are a separate project. Read "value" as production priced in points and
    weighted by leverage, and "impact" as the raw production it is compared to.

Run:  py prediction_engines/value_engine.py --seasons 2017_2018 [--eval]
      py prediction_engines/value_engine.py --all           (every season, cached)
Output: output/value_cache_v1.pkl   {game_id: {person_id: {name, team, value, components..., events}}}
"""
import argparse
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_PATH = os.path.join(OUTPUT_DIR, "value_cache_v1.pkl")
GAMES_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "games.pkl")
BOX_PLAYER_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "box_player_traditional.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")

BASELINE = 1.00
OREB_SHARE = 0.26
TRANSITION = 0.15
ASSIST = 0.35
BLOCK = 0.6
FOUL_PERSONAL, FOUL_SHOOTING = -0.3, -0.6
CLUTCH_WEIGHT = 1.25
GARBAGE_FLOOR = 0.3

COMPONENTS = ("scoring", "misses", "free_throws", "assists", "rebounds",
              "stops", "turnovers", "fouls")
ASSIST_RE = re.compile(r"\(([^()]+?) (\d+) AST\)")


# ---------------------------------------------------------------------------
#  Game state
# ---------------------------------------------------------------------------
def clock_seconds(value):
    """'PT07M23.00S' -> 443.0; anything unreadable -> 0."""
    if isinstance(value, str) and value.startswith("PT"):
        try:
            minutes, rest = value[2:].split("M")
            return int(minutes) * 60 + float(rest.rstrip("S") or 0)
        except ValueError:
            return 0.0
    return 0.0


def leverage(period, seconds_left, margin):
    """How much the game still hung on this play, 0.3 .. 1.25."""
    m = abs(margin)
    if period >= 4:
        if seconds_left <= 300 and m <= 5:
            return CLUTCH_WEIGHT
        if m <= 10:
            return 1.0
        # 10 points apart -> 1.0, 25 or more -> the floor, linear between
        return max(GARBAGE_FLOOR, 1.0 - (m - 10) / 15.0 * (1.0 - GARBAGE_FLOOR))
    if m <= 15:
        return 1.0
    if m <= 25:
        return 0.7
    return 0.5


# ---------------------------------------------------------------------------
#  Pricing one play
# ---------------------------------------------------------------------------
def price(row, last_miss_team):
    """(component, value) for the player named on the row, or None.

    `last_miss_team` is who missed the most recent shot or free throw: the
    description of a rebound ("REBOUND (Off:0 Def:1)") carries the player's
    running totals, not the kind of this rebound, so the kind has to be read
    from whose miss it followed.
    """
    kind = row.get("actionType")
    desc = row.get("description") if isinstance(row.get("description"), str) else ""
    if kind == "Made Shot":
        points = 3.0 if row.get("shotValue") == 3 else 2.0
        return "scoring", points - BASELINE
    if kind == "Missed Shot":
        return "misses", -BASELINE * (1.0 - OREB_SHARE)
    if kind == "Free Throw":
        made = "MISS" not in desc
        return "free_throws", (1.0 if made else 0.0) - 0.44 * BASELINE
    if kind == "Rebound":
        offensive = last_miss_team is not None and row.get("teamTricode") == last_miss_team
        return "rebounds", BASELINE * ((1.0 - OREB_SHARE) if offensive else OREB_SHARE)
    if kind == "Turnover":
        return "turnovers", -(BASELINE + TRANSITION)
    if "STEAL" in desc:
        return "stops", BASELINE + TRANSITION
    if "BLOCK" in desc:
        return "stops", BLOCK
    if kind == "Foul":
        if "OFF.FOUL" in desc or "Offensive" in desc:
            return None                      # its Turnover row carries the cost
        if "S.FOUL" in desc or "Shooting" in desc:
            return "fouls", FOUL_SHOOTING
        if "T.FOUL" in desc or "Technical" in desc:
            return "fouls", -0.5
        return "fouls", FOUL_PERSONAL
    return None


def compute_game_value(pbp, use_leverage=True):
    """Per-player value for one game from its play-by-play frame.

    Returns {person_id: {name, team, value, <components>, events}}. Rows
    without a person id (pre-2019 feeds) fall back to 'name:<surname>' keys.
    """
    if pbp is None or pbp.empty:
        return {}
    df = pbp.sort_values("actionNumber") if "actionNumber" in pbp.columns else pbp
    df = df.copy()
    df["_t"] = df["clock"].map(clock_seconds)
    df["_h"] = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0.0)
    df["_a"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0.0)
    df["_p"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)

    players = {}
    surname_to_id = defaultdict(set)          # (team, surname) -> ids seen on that team

    def entry(pid, name, team):
        e = players.get(pid)
        if e is None:
            e = players[pid] = {"name": name, "team": team, "value": 0.0, "events": 0,
                                **{c: 0.0 for c in COMPONENTS}}
        if team and not e["team"]:
            e["team"] = team
        return e

    last_miss_team = None
    for row in df.to_dict("records"):
        kind = row.get("actionType")
        desc = row.get("description") if isinstance(row.get("description"), str) else ""
        team_of_row = row.get("teamTricode") if isinstance(row.get("teamTricode"), str) else None
        is_miss = kind == "Missed Shot" or (kind == "Free Throw" and "MISS" in desc)
        name = row.get("playerName")
        if not isinstance(name, str) or not name:
            if is_miss:
                last_miss_team = team_of_row
            continue
        pid = row.get("personId")
        if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pid == 0:
            pid = f"name:{name}"
        else:
            pid = int(pid)
        team = row.get("teamTricode") if isinstance(row.get("teamTricode"), str) else None
        if team:
            surname_to_id[(team, name)].add(pid)
        priced = price(row, last_miss_team)
        if is_miss:
            last_miss_team = team
        if priced is None:
            continue
        component, value = priced
        w = leverage(row["_p"], row["_t"], row["_h"] - row["_a"]) if use_leverage else 1.0
        e = entry(pid, name, team)
        e[component] += value * w
        e["value"] += value * w
        e["events"] += 1

        if component == "scoring":
            match = ASSIST_RE.search(row.get("description") or "")
            if match and team:
                ids = surname_to_id.get((team, match.group(1)), set())
                if len(ids) == 1:
                    a = entry(next(iter(ids)), match.group(1), team)
                    a["assists"] += ASSIST * w
                    a["value"] += ASSIST * w
    return players


# ---------------------------------------------------------------------------
#  Seasons and the cache
# ---------------------------------------------------------------------------
def season_game_ids(season):
    games = pd.read_pickle(GAMES_PATH)
    return sorted(games.loc[games["season"] == season, "game_id"].astype(str).str.zfill(10))


def pbp_path(season, game_id):
    return os.path.join(DATA_DIR, season, game_id, "play_by_play", f"{game_id}pbp.csv")


def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def run_seasons(seasons, cache, verbose=True, cache_path=CACHE_PATH, use_leverage=True):
    t0 = time.time()
    for season in seasons:
        ids = season_game_ids(season)
        todo = [g for g in ids if g not in cache]
        missing = 0
        for n, gid in enumerate(todo, 1):
            path = pbp_path(season, gid)
            if not (os.path.exists(path) and os.path.getsize(path) > 64):
                missing += 1
                continue
            try:
                cache[gid] = compute_game_value(pd.read_csv(path, low_memory=False), use_leverage)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                missing += 1
            if verbose and n % 200 == 0:
                print(f"    {season}: {n}/{len(todo)} ({time.time() - t0:.0f} sn)", flush=True)
        if verbose:
            print(f"  {season}: {len(ids)} mac, {len(todo)} yeni, {missing} dosyasiz "
                  f"({time.time() - t0:.0f} sn)", flush=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    return cache


def player_frame(cache, game_ids):
    rows = []
    for gid in game_ids:
        for pid, e in (cache.get(gid) or {}).items():
            rows.append({"game_id": gid, "player": pid, "name": e["name"], "team": e["team"],
                         "value": e["value"], **{c: e[c] for c in COMPONENTS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Evaluation against the impact score and on-court plus-minus
# ---------------------------------------------------------------------------
def evaluate(season, cache, min_games=40, min_minutes=24.0):
    ids = season_game_ids(season)
    value = player_frame(cache, ids)
    value = value[~value["player"].astype(str).str.startswith("name:")].copy()
    value["player"] = value["player"].astype(int)

    with open(IMPACT_CACHE, "rb") as f:
        impact_cache = pickle.load(f)
    imp_rows = []
    for gid in ids:
        e = impact_cache.get(gid)
        for v in ((e or {}).get("players") or {}).values():
            if isinstance(v, dict) and v.get("person_id") is not None:
                imp_rows.append((gid, int(v["person_id"]), float(v.get("impact", 0.0))))
    impact = pd.DataFrame(imp_rows, columns=["game_id", "player", "impact"])

    box = pd.read_pickle(BOX_PLAYER_PATH)
    box = box[box["game_id"].isin(ids)][["game_id", "player_id", "player_name", "min", "plus_minus",
                                          "pts", "reb", "ast", "to", "fga", "fta"]]
    parts = box["min"].astype("string").str.extract(r"^(\d+)(?:\.\d+)?(?::(\d+))?")
    box["minutes"] = (pd.to_numeric(parts[0], errors="coerce").fillna(0)
                      + pd.to_numeric(parts[1], errors="coerce").fillna(0) / 60.0)
    box = box.rename(columns={"player_id": "player"})

    m = box.merge(value, on=["game_id", "player"], how="inner").merge(impact, on=["game_id", "player"], how="left")
    m = m[m["minutes"] > 0]
    agg = (m.groupby("player_name")
           .agg(g=("value", "size"), minutes=("minutes", "mean"), value=("value", "mean"),
                impact=("impact", "mean"), pm=("plus_minus", "mean"), pts=("pts", "mean"),
                reb=("reb", "mean"), ast=("ast", "mean"), to=("to", "mean"),
                fga=("fga", "mean"), fta=("fta", "mean"),
                **{c: (c, "mean") for c in COMPONENTS}))
    agg = agg[(agg["g"] >= min_games) & (agg["minutes"] >= min_minutes)].copy()
    agg["ts"] = agg["pts"] / (2 * (agg["fga"] + 0.44 * agg["fta"]))
    for col in ("value", "impact", "pm"):
        agg[f"{col}36"] = agg[col] / agg["minutes"] * 36
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    agg["empty_index"] = z(agg["impact36"]) - z(agg["value36"])
    return agg


def print_evaluation(agg, season):
    print(f"\n=== {season}: {len(agg)} duzenli oyuncu (>=40 mac, >=24 dk) ===")
    print(f"  on-court +/- (36 dk) ile korelasyon:  impact {agg['impact36'].corr(agg['pm36']):.3f}"
          f"   value {agg['value36'].corr(agg['pm36']):.3f}")
    print(f"  mac basina:                           impact {agg['impact'].corr(agg['pm']):.3f}"
          f"   value {agg['value'].corr(agg['pm']):.3f}")
    cols = ["g", "minutes", "value", "impact", "pm", "pts", "reb", "ast", "to", "ts"]
    print("\n  DEGER'e gore ilk 15:")
    print(agg.sort_values("value", ascending=False).head(15)[cols].round(2).to_string())
    print("\n  en 'bos' istatistik (impact yuksek, deger dusuk):")
    print(agg.sort_values("empty_index", ascending=False).head(10)[cols + ["empty_index"]].round(2).to_string())
    print("\n  en 'sessiz' deger (deger yuksek, impact dusuk):")
    print(agg.sort_values("empty_index").head(8)[cols + ["empty_index"]].round(2).to_string())
    for name in ("Russell Westbrook", "Stephen Curry", "James Harden", "Kevin Durant",
                 "Andre Drummond", "Anthony Davis", "LeBron James"):
        if name in agg.index:
            r = agg.loc[name]
            comps = "  ".join(f"{c} {r[c]:+.1f}" for c in COMPONENTS)
            print(f"\n  {name}: value {r['value']:.1f}/mac (impact {r['impact']:.1f}), "
                  f"+/- {r['pm']:+.1f}, TS {r['ts']:.3f}\n    {comps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="veri setindeki her sezon")
    parser.add_argument("--eval", action="store_true", help="impact ve +/- ile karsilastir")
    parser.add_argument("--no-leverage", action="store_true",
                        help="oyun agirligi yok (her oyun 1.0) - takim feature deneyi icin")
    parser.add_argument("--cache", default=CACHE_PATH, help="onbellek dosyasi")
    args = parser.parse_args()
    seasons = args.seasons or []
    if args.all:
        seasons = sorted(pd.read_pickle(GAMES_PATH)["season"].unique())
    if not seasons:
        seasons = ["2017_2018"]
    cache = load_cache(args.cache)
    cache = run_seasons(seasons, cache, cache_path=args.cache, use_leverage=not args.no_leverage)
    print(f"onbellek: {len(cache):,} mac -> {args.cache}")
    if args.eval:
        for season in seasons:
            print_evaluation(evaluate(season, cache), season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
