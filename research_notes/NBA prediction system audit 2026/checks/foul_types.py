"""Foul, turnover and violation subtypes per team-game from the project's PlayByPlayV3 CSVs:
2026-27 games on disk vs 2025-26 opening week vs 2025-26 full season. Read-only."""
import collections
import pathlib
import sys

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DATA = pathlib.Path(sys.argv[1])

log = leaguegamelog.LeagueGameLog(season="2025-26", season_type_all_star="Regular Season",
                                  player_or_team_abbreviation="T", timeout=90).get_data_frames()[0]
date_by_game = dict(zip(log.GAME_ID, log.GAME_DATE))
opening_week = {g for g, d in date_by_game.items() if "2025-10-21" <= d <= "2025-10-27"}


def tally(files):
    counts = collections.Counter()
    games = 0
    for path in files:
        pbp = pd.read_csv(path, dtype=str).fillna("")
        games += 1
        action = pbp.actionType.str.strip().str.lower()
        subtype = pbp.subType.str.strip().str.lower()
        for kind in ("foul", "turnover", "violation"):
            for name, n in subtype[action == kind].value_counts().items():
                counts[f"{kind}:{name or '(blank)'}"] += int(n)
            counts[f"{kind}:ALL"] += int((action == kind).sum())
        counts["desc:flop"] += int(pbp.description.str.contains("flop", case=False).sum())
    team_games = 2 * games
    return games, {k: round(v / team_games, 3) for k, v in sorted(counts.items())}


season_now = sorted((DATA / "2026_2027").glob("*/play_by_play/*pbp.csv"))
season_prev = sorted((DATA / "2025_2026").glob("*/play_by_play/*pbp.csv"))
prev_open = [p for p in season_prev if p.name[:10] in opening_week]

results = {"2026-27 on disk": tally(season_now), "2025-26 opening week": tally(prev_open),
           "2025-26 full season": tally(season_prev)}
keys = sorted({k for _, rates in results.values() for k in rates})
print("games:", {label: n for label, (n, _) in results.items()})
print(f"{'per team-game':45s}" + "".join(f"{label:>24s}" for label in results))
for key in keys:
    values = [rates.get(key, 0.0) for _, rates in results.values()]
    if max(values) >= 0.05:
        print(f"{key:45s}" + "".join(f"{v:24.3f}" for v in values))
