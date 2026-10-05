"""Backtest five seasons from pre-season information under different settings."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import season_sim as sim
games = sim.load_games(); box = sim.load_box(games)
sim.load_games = lambda: games; sim.load_box = lambda g: box
SEASONS = ["2021_2022", "2022_2023", "2023_2024", "2024_2025", "2025_2026"]
real = {s: sim.real_reference(games, box, s)[1][["team", "wins"]] for s in SEASONS + ["2020_2021"]}
naive = []
for s in SEASONS:
    prev = sim.previous_seasons(games, s, 1)[0]
    n = real[s].merge(real[prev], on="team", suffixes=("", "_prev"))
    scale = 82 / (n["wins_prev"].sum() * 2 / 30)           # 2020-21 had 72 games
    naive.append((n["wins"].corr(n["wins_prev"]), (n["wins"] - n["wins_prev"] * scale).abs().mean()))
print(f"kiyas 'gecen sezonun aynisi': korelasyon {np.mean([a for a, _ in naive]):.3f}, ortalama mutlak hata {np.mean([b for _, b in naive]):.1f}")
settings = [dict(TEAM_EFFECT_KEPT=k, PRIOR_MINUTES=m) for k, m in ((0.0, 150.0), (0.5, 150.0), (0.75, 150.0), (1.0, 150.0), (1.0, 50.0))]
for setting in settings:
    for k, v in setting.items(): setattr(sim, k, v)
    rows = []
    for season in SEASONS:
        sandbox = sim.Sandbox(f"sweep_{season}")
        setup = sim.prepare_season("backtest", season, sandbox)      # reads the settings set above
        table = sim.replicate(setup, sandbox, [31, 32], 0.0)         # no team shock: the point estimate
        both = real[season].merge(table, on="team")
        rows.append((both["wins"].corr(both["mean_wins"]), (both["wins"] - both["mean_wins"]).abs().mean(), both["mean_wins"].std(), both["wins"].std()))
    r = np.array(rows)
    print(f"{setting}: korelasyon {r[:,0].mean():.3f} ({' '.join(f'{x:.2f}' for x in r[:,0])}) | MAE {r[:,1].mean():.1f} | yayilim {r[:,2].mean():.1f} / gercek {r[:,3].mean():.1f}", flush=True)
