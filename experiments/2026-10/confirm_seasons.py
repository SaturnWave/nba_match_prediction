"""Old pricing vs 5.1 against on-court plus-minus, every season; tuned ones marked."""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import value_engine as ve
S = sys.argv[1]
TUNED = {"2017_2018", "2020_2021", "2023_2024"}
old_cache = pickle.load(open(f"{S}/value_cache_v1_old.pkl", "rb"))
new_cache = pickle.load(open("output/value_cache_v1.pkl", "rb"))
box_ast = pd.read_pickle(ve.BOX_PLAYER_PATH)[["game_id", "player_id", "ast"]].rename(columns={"player_id": "player", "ast": "box_ast"})
seasons = sorted(pd.read_pickle(ve.GAMES_PATH)["season"].unique())
rows = []
for season in seasons:
    ids = ve.season_game_ids(season)
    def frame(c, col):
        v = ve.player_frame(c, ids, ("assists",)); v = v[~v["player"].astype(str).str.startswith("name:")].copy()
        v["player"] = v["player"].astype(int); return v[["game_id", "player", "value", "assists"]].rename(columns={"value": col, "assists": col + "_ast"})
    m = ve.box_minutes(ids).merge(frame(new_cache, "new"), on=["game_id", "player"], how="left").merge(frame(old_cache, "old"), on=["game_id", "player"], how="left")
    m = m[m.minutes > 0].fillna(0.0)
    short = len(ids) < 1150
    a = m.groupby("player_name").agg(g=("minutes", "size"), minutes=("minutes", "mean"), new=("new", "mean"), old=("old", "mean"), pm=("plus_minus", "mean"))
    a = a[(a.g >= (35 if short else 40)) & (a.minutes >= 24)]
    c = lambda col: float((a[col] / a.minutes).corr(a.pm / a.minutes))
    rows.append((season, "ayar" if season in TUNED else "ayri", len(a), c("old"), c("new")))
r = pd.DataFrame(rows, columns=["sezon", "kume", "n", "onceki", "yeni"]); r["fark"] = r.yeni - r.onceki
print(r.round(3).to_string(index=False))
for k, g in r.groupby("kume"):
    print(f"  {k}: onceki {g.onceki.mean():.3f} -> yeni {g.yeni.mean():.3f}  (fark {g.fark.mean():+.3f}, {int((g.fark > 0).sum())}/{len(g)} sezonda daha iyi)")
