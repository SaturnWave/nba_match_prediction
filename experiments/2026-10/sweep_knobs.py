"""Average the mechanics comparison over seeds for a few knob settings."""
import sys, itertools
import numpy as np
sys.path.insert(0, "prediction_engines")
import season_sim as sim

games = sim.load_games(); box = sim.load_box(games)
sim.load_games = lambda: games; sim.load_box = lambda g: box        # reuse the loaded frames
sandbox = sim.Sandbox("calib_sweep")
setup = sim.prepare_season("mechanics", "2025_2026", sandbox)   # the knobs are read at play time
settings = [dict(PACE_TIME_FACTOR=0.945, HOME_EDGE=0.010, SCORE_EFFECT=0.017, GAME_PACE_SD=0.012, FOULING_EFFECT=0.0),
            dict(PACE_TIME_FACTOR=0.945, HOME_EDGE=0.010, SCORE_EFFECT=0.017, GAME_PACE_SD=0.012, FOULING_EFFECT=1.0)]
keys = ["pts", "possessions", "fg_pct", "home_win_pct", "home_margin", "margin_sd", "total_sd", "overtime_share"]
for setting in settings:
    for k, v in setting.items(): setattr(sim, k, v)
    rows, corr = [], []
    for seed in (1, 2, 3, 4, 5, 6):
        g, l, v, extras = sim.simulate_season(setup, sandbox, seed, keep_events=False, verbose=False)
        c = sim.compare_with_real(l, g, games, box, "2025_2026")
        rows.append([c["averages"][k][1] for k in keys]); real = [c["averages"][k][0] for k in keys]
        corr.append((c["wins_correlation"], c["wins_mean_abs_error"], c["margin_correlation"], c["margin_spread"][1]))
    m = np.mean(rows, axis=0); cm = np.mean(corr, axis=0)
    print(setting)
    print("   " + "  ".join(f"{k} {a:.3f}/{b:.3f}" for k, a, b in zip(keys, real, m)))
    print(f"   galibiyet korelasyonu {cm[0]:.3f}, MAE {cm[1]:.1f}; sayi farki korelasyonu {cm[2]:.3f}; takim yayilimi {cm[3]:.1f} (gercek {c['margin_spread'][0]:.1f})")

tm = box[box.season == "2025_2026"].groupby(["game_id", "team"]).minutes.sum()
print(f"gercek uzatma orani (takim dakikasi > 250): {(tm.groupby('game_id').first() > 250).mean():.3f}")
