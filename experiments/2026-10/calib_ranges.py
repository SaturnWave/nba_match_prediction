"""Do the replications' 10-90% win ranges cover what really happened four times in five?"""
import sys, time
import numpy as np
sys.path.insert(0, "prediction_engines")
import season_sim as sim
t0 = time.time()
setup = sim.prepare_season("mechanics", "2025_2026", sim.Sandbox("calib_a"))
g, l, v, e = sim.simulate_season(setup, sim.Sandbox("calib_a"), 5, keep_events=False, verbose=False)
c = sim.compare_with_real(l, g, setup.history_games, setup.history_box, "2025_2026")
a = c["averages"]
print(f"mekanik kontrol: sayi {a['pts'][1]:.1f}/{a['pts'][0]:.1f}, pozisyon {a['possessions'][1]:.1f}/{a['possessions'][0]:.1f}, ev farki {a['home_margin'][1]:.2f}/{a['home_margin'][0]:.2f}, "
      f"fark yayilimi {a['margin_sd'][1]:.1f}/{a['margin_sd'][0]:.1f}; galibiyet korelasyonu {c['wins_correlation']:.3f}, sayi farki {c['margin_correlation']:.3f}, takim yayilimi {c['margin_spread'][1]:.1f}/{c['margin_spread'][0]:.1f}")
for sd in (0.08, 0.10):
    cover, width, corr, mae = [], [], [], []
    for season in ("2023_2024", "2024_2025", "2025_2026"):
        sandbox = sim.Sandbox(f"sweep_{season}")
        setup = sim.prepare_season("backtest", season, sandbox)
        ranges = sim.replicate(setup, sandbox, list(range(100, 114)), sd)
        real = sim.real_reference(setup.history_games, setup.history_box, season)[1][["team", "wins"]]
        both = real.merge(ranges, on="team")
        cover.append(((both.wins >= both.low) & (both.wins <= both.high)).mean()); width.append((both.high - both.low).mean())
        corr.append(both.wins.corr(both.mean_wins)); mae.append((both.wins - both.mean_wins).abs().mean())
    print(f"takim sezon sapmasi {sd}: %10-%90 araligi gercegi kapsama {np.mean(cover):.0%} ({' '.join(f'{x:.0%}' for x in cover)}), ortalama aralik genisligi {np.mean(width):.0f} mac; korelasyon {np.mean(corr):.3f}, MAE {np.mean(mae):.1f}  ({time.time()-t0:.0f} sn)", flush=True)
