"""Which assist rule tracks on-court plus-minus best? Post hoc, from the raw fields."""
import pickle, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import value_engine as ve

S = sys.argv[1]
SEASONS = ["2017_2018", "2020_2021", "2023_2024"]
old_cache = pickle.load(open("output/value_cache_v1.pkl", "rb"))
track = pd.read_pickle(ve.TRACKING_PATH)[["game_id", "player_id", "pass"]].rename(columns={"player_id": "player"})
box_all = pd.read_pickle(ve.BOX_PLAYER_PATH)[["game_id", "player_id", "ast"]].rename(columns={"player_id": "player", "ast": "box_ast"})

frames = {}
for season in SEASONS:
    cache = pickle.load(open(f"{S}/tune_{season}.pkl", "rb"))
    ids = ve.season_game_ids(season)
    v = ve.player_frame(cache, ids, ve.COMPONENTS + ve.RAW_FIELDS)
    v = v[~v["player"].astype(str).str.startswith("name:")].copy(); v["player"] = v["player"].astype(int)
    old = pd.DataFrame([(g, p, e["free_throws"]) for g in ids for p, e in (old_cache.get(g) or {}).items()
                        if not str(p).startswith("name:")], columns=["game_id", "player", "ft_old"])
    old["player"] = old["player"].astype(int)
    m = (ve.box_minutes(ids).merge(v, on=["game_id", "player"], how="left")
         .merge(old, on=["game_id", "player"], how="left").merge(track, on=["game_id", "player"], how="left")
         .merge(box_all, on=["game_id", "player"], how="left"))
    m = m[m["minutes"] > 0].fillna(0.0)
    frames[season] = m
    exact = (m["assists_n"] == m["box_ast"]).mean()
    print(f"{season}: asist  kutu {int(m.box_ast.sum())}  motor {int(m.assists_n.sum())}  "
          f"({m.assists_n.sum()/m.box_ast.sum():.2%});  oyuncu-mac birebir {exact:.4f}")

def corr(season_frame, value_col, min_games):
    a = season_frame.groupby("player_name").agg(g=("minutes", "size"), minutes=("minutes", "mean"),
                                                v=(value_col, "mean"), pm=("plus_minus", "mean"))
    a = a[(a.g >= min_games) & (a.minutes >= 24)]
    return float((a.v / a.minutes).corr(a.pm / a.minutes)), len(a)

def score(rule):
    out = []
    for season, m in frames.items():
        m = m.assign(_v=rule(m))
        out.append(corr(m, "_v", 35 if season == "2020_2021" else 40)[0])
    return out

lev = lambda m: m["leverage_mean"]
rules = {"asist yok (yalniz cekirdek)": lambda m: m["core"],
         "eski: +0.35 sabit, ustune ek": lambda m: m["core"] + 0.35 * m["assist_weight"]}
for s in (0.2, 0.3, 0.4, 0.5):
    rules[f"ustune ek, basketin %{int(s*100)}'i"] = lambda m, s=s: m["core"] + s * m["assist_base"]
for s in (0.2, 0.3, 0.4, 0.5):
    rules[f"PAYLASIM %{int(s*100)} (skorerden pasore)"] = lambda m, s=s: m["core"] - s * m["assisted_scoring"] + s * m["assist_base"]
print(f"\n{'kural':42} " + " ".join(f"{s:>10}" for s in SEASONS) + f" {'ortalama':>9}")
for name, rule in rules.items():
    c = score(rule); print(f"{name:42} " + " ".join(f"{x:10.3f}" for x in c) + f" {np.mean(c):9.3f}")

print("\nPAYLASIM %30 ustune tracking ekleri (ikincil asist k2, serbest atis asisti k3, pas basina kp):")
base = lambda m, s=0.30: m["core"] - s * m["assisted_scoring"] + s * m["assist_base"]
for k2, k3, kp in [(0, 0, 0), (0.15, 0, 0), (0.3, 0, 0), (0.5, 0, 0), (0, 0.2, 0), (0, 0.4, 0),
                   (0.15, 0.2, 0), (0.3, 0.4, 0), (0.15, 0.2, 0.01), (0.15, 0.2, 0.02)]:
    c = score(lambda m: base(m) + (k2 * m["secondary_assists"] + k3 * m["ft_assists"] + kp * m["pass"]) * lev(m))
    print(f"  k2={k2:<4} k3={k3:<4} kp={kp:<5} " + " ".join(f"{x:10.3f}" for x in c) + f" {np.mean(c):9.3f}")

print("\nserbest atis fiyatlamasi (PAYLASIM %30 + k2 0.15 + k3 0.2 uzerinde):")
full = lambda m: base(m) + (0.15 * m["secondary_assists"] + 0.2 * m["ft_assists"]) * lev(m)
c_new = score(full); c_old = score(lambda m: full(m) - m["free_throws"] + m["ft_old"])
print("  atis turune gore (yeni)   " + " ".join(f"{x:10.3f}" for x in c_new) + f" {np.mean(c_new):9.3f}")
print("  sabit 0.44 (eski)         " + " ".join(f"{x:10.3f}" for x in c_old) + f" {np.mean(c_old):9.3f}")
m = frames["2023_2024"]
print("\n2023-24 serbest atis bileseni, mac basina (eski -> yeni):")
a = m.groupby("player_name").agg(g=("minutes", "size"), old=("ft_old", "mean"), new=("free_throws", "mean"))
a = a[a.g >= 40]; a["fark"] = a.new - a.old
print(a.sort_values("fark").head(4).round(2).to_string()); print(a.sort_values("fark").tail(4).round(2).to_string())
