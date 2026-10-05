import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import value_engine as ve
S = sys.argv[1]
old_cache = pickle.load(open("output/value_cache_v1.pkl", "rb"))
box_ast = pd.read_pickle(ve.BOX_PLAYER_PATH)[["game_id", "player_id", "ast"]].rename(columns={"player_id": "player", "ast": "box_ast"})
print(f"sabitler: ASSIST_SHARE {ve.ASSIST_SHARE}, SECONDARY {ve.SECONDARY_ASSIST}, FT_ASSIST {ve.FT_ASSIST}")
for season in ["2017_2018", "2020_2021", "2023_2024"]:
    cache = pickle.load(open(f"{S}/tune_{season}.pkl", "rb")); ids = ve.season_game_ids(season)
    def frame(c, fields):
        v = ve.player_frame(c, ids, fields); v = v[~v["player"].astype(str).str.startswith("name:")].copy()
        v["player"] = v["player"].astype(int); return v
    new = frame(cache, ve.COMPONENTS + ve.RAW_FIELDS); old = frame(old_cache, ())[["game_id", "player", "value"]].rename(columns={"value": "value_old"})
    m = ve.box_minutes(ids).merge(new, on=["game_id", "player"], how="left").merge(old, on=["game_id", "player"], how="left").merge(box_ast, on=["game_id", "player"], how="left")
    m = m[m.minutes > 0].fillna(0.0)
    a = m.groupby("player_name").agg(g=("minutes", "size"), minutes=("minutes", "mean"), new=("value", "mean"), old=("value_old", "mean"), pm=("plus_minus", "mean"), ast=("assists", "mean"))
    a = a[(a.g >= (35 if season == "2020_2021" else 40)) & (a.minutes >= 24)]
    c = lambda col: (a[col] / a.minutes).corr(a.pm / a.minutes)
    print(f"{season}: asist yakalama {m.assists_n.sum()/m.box_ast.sum():.2%} (birebir {(m.assists_n == m.box_ast).mean():.4f}) | "
          f"+/- ile korelasyon: onceki v5 {c('old'):.3f} -> yeni {c('new'):.3f}  (n={len(a)})")
    if season == "2017_2018":
        for n in ("Russell Westbrook", "Stephen Curry", "James Harden", "LeBron James", "Chris Paul"):
            if n in a.index: print(f"    {n:18} onceki {a.loc[n, 'old']:.2f} -> yeni {a.loc[n, 'new']:.2f}  (asist bileseni {a.loc[n, 'ast']:.2f}/mac)")
