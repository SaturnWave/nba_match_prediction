"""How well does a pre-season simulation foresee a season? Two past seasons, ten runs each."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import season_sim as sim
games = sim.load_games(); box = sim.load_box(games)
sim.load_games = lambda: games; sim.load_box = lambda g: box
for season in ("2024_2025", "2025_2026"):
    sandbox = sim.Sandbox(f"backtest_{season}")
    ranges = sim.replicate(sim.prepare_season("backtest", season, sandbox), sandbox, list(range(11, 21)), 0.0)
    _avg, real = sim.real_reference(games, box, season)
    both = real[["team", "wins"]].merge(ranges, on="team")
    err = both["wins"] - both["mean_wins"]
    inside = ((both["wins"] >= both["low"]) & (both["wins"] <= both["high"])).mean()
    print(f"\n{season} (sezon oncesi bilgiyle, 10 simulasyon ortalamasi):")
    print(f"  galibiyet korelasyonu {both['wins'].corr(both['mean_wins']):.3f} | ortalama mutlak hata {err.abs().mean():.1f} mac | "
          f"gercek, %10-%90 araliginda: {inside:.0%} | simule yayilim {both['mean_wins'].std():.1f} / gercek {both['wins'].std():.1f}")
    worst = both.assign(err=err).sort_values("err")
    show = lambda r: f"{r.team} gercek {r.wins} / tahmin {r.mean_wins:.0f}"
    print("  en cok fazla tahmin edilen:", "; ".join(show(r) for r in worst.head(3).itertuples()))
    print("  en cok az tahmin edilen:  ", "; ".join(show(r) for r in worst.tail(3).itertuples()))
    naive = real[["team", "wins"]].merge(sim.real_reference(games, box, sim.previous_seasons(games, season, 1)[0])[1][["team", "wins"]], on="team", suffixes=("", "_prev"))
    print(f"  kiyas - 'gecen sezonun galibiyeti aynen': korelasyon {naive['wins'].corr(naive['wins_prev']):.3f}, ortalama mutlak hata {(naive['wins'] - naive['wins_prev']).abs().mean():.1f}")
