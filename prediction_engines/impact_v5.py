"""
Impact score v5: the value engine's pricing, in the impact cache's format.

WHY A NEW CACHE VERSION
    game_impact_cache_v4.pkl was produced by compute_game_impact, which counts
    production without its cost: no debit for a missed shot, every rebound
    priced as an offensive one (a string test that matches the running totals
    in every description), no assists, no free throws, no discount once a
    game is decided. Measured on 2017-18 regulars the result correlates 0.14
    with on-court plus-minus per 36 minutes; value_engine.py's pricing
    correlates 0.42, and 0.39 / 0.43 on 2023-24 and 2025-26.

    Twelve of the model's features are rolling means of the TEAM sum of this
    score, and twelve more are the roster family built from the per-player
    values. Fixing the engine therefore means rebuilding the cache every
    consumer reads, not patching one function. This module writes
    game_impact_cache_v5.pkl with exactly v4's layout -
        {game_id: {"home_impact", "away_impact",
                   "players": {person_id: {"impact", "team", "name", "person_id"}}}}
    - so build_dataset_db, pregame_roster, player_source, db_build_derived and
    app.py switch by changing one path.

    The per-player number is the value score (points above a 1.00 per
    possession baseline, leverage-weighted); the team numbers are the sums over
    the players the box score lists for that side. Players the play-by-play
    could not identify (pre-2019 rows without a person id) are dropped from
    the per-player map but still counted in the team sums, which is what v4
    did as well.

Run:  py prediction_engines/impact_v5.py        (needs output/value_cache_unweighted.pkl,
                                                 from value_engine.py --all --no-leverage)
Output: game_impact_cache_v5.pkl, output/impact_v5_summary.json
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
# The MODEL's cache is built from the UNWEIGHTED value run (value_engine.py
# --no-leverage). Measured over 54 paired walk-forward cells: the leverage-
# weighted values cost the forecast Brier +0.0030 +/- 0.0015 and raised the
# winner-vs-margin contradiction rate from 8.9% to 12.4%, because for
# predicting the NEXT game the size of a blowout is information, not noise;
# the unweighted values were indistinguishable from the old score on every
# metric (accuracy -0.0015 +/- 0.0051, Brier +0.0009 +/- 0.0013) with a lower
# seed spread. The leverage-weighted values stay where they belong: in the
# player ratings (output/value_cache_v1.pkl), where crediting a player for
# garbage time would be the padding this engine exists to remove.
VALUE_CACHE = os.path.join(OUTPUT_DIR, "value_cache_unweighted.pkl")
V4_PATH = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
V5_PATH = os.path.join(PROJECT_ROOT, "game_impact_cache_v5.pkl")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "impact_v5_summary.json")
GAME_SUMMARY = os.path.join(PROJECT_ROOT, "phonedb_cache", "game_summary.pkl")


def build_v5(value_cache, games):
    """v4-shaped cache from the value cache; `games` gives each game's sides."""
    sides = games.set_index("game_id")[["home_abbr", "away_abbr"]].to_dict("index")
    out = {}
    for gid, players in value_cache.items():
        side = sides.get(gid)
        if side is None:
            continue
        home, away = side["home_abbr"], side["away_abbr"]
        home_sum = away_sum = 0.0
        detail = {}
        for pid, e in players.items():
            team = e.get("team")
            value = float(e.get("value", 0.0))
            if team == home:
                home_sum += value
            elif team == away:
                away_sum += value
            if isinstance(pid, str) and pid.startswith("name:"):
                continue
            detail[int(pid)] = {"impact": value, "team": team, "name": e.get("name"),
                                "person_id": int(pid)}
        out[gid] = {"home_impact": home_sum, "away_impact": away_sum, "players": detail}
    return out


def _pricing_version():
    """The value engine's pricing version, so a cache says which rules built it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("value_engine",
                                                  os.path.join(HERE, "value_engine.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["value_engine"] = module      # @dataclass resolves annotations through it
    spec.loader.exec_module(module)
    return module.PRICING_VERSION


def team_margin_check(cache, games):
    """How tightly the team-level difference tracks the real margin."""
    rows = []
    for gid, e in cache.items():
        g = games.loc[games["game_id"] == gid]
        if g.empty or not isinstance(e, dict):
            continue
        rows.append((e["home_impact"] - e["away_impact"], float(g["margin"].iloc[0])))
    if not rows:
        return None
    d = np.array(rows)
    return float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-cache", default=VALUE_CACHE)
    parser.add_argument("--out", default=V5_PATH)
    parser.add_argument("--summary", default=SUMMARY_PATH)
    args = parser.parse_args()
    games = pd.read_pickle(GAME_SUMMARY)
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    with open(args.value_cache, "rb") as f:
        value_cache = pickle.load(f)
    v5 = build_v5(value_cache, games)
    with open(args.out, "wb") as f:
        pickle.dump(v5, f)

    summary = {"games": len(v5), "source": os.path.basename(args.value_cache),
               "pricing_version": _pricing_version(),
               "team_diff_vs_margin_corr_v5": team_margin_check(v5, games)}
    if os.path.exists(V4_PATH):
        with open(V4_PATH, "rb") as f:
            v4 = pickle.load(f)
        summary["team_diff_vs_margin_corr_v4"] = team_margin_check(v4, games)
        summary["games_v4"] = len(v4)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Yazildi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
