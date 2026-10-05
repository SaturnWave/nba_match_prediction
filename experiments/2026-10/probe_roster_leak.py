"""How much does 'who appeared in the game' carry about the game itself?"""
import pickle, numpy as np, pandas as pd
blob = pd.read_pickle("output/engineered_dataset_db.pkl")
ds = blob["dataset"].copy(); feats = blob["features"]
ds["game_date"] = pd.to_datetime(ds["game_date"])
cache = pickle.load(open("game_impact_cache_v4.pkl", "rb"))
sides = ds.set_index("game_id")[["home_team", "away_team", "point_diff", "home_win", "game_date", "season"]]
rows = []
for gid, e in cache.items():
    if gid not in sides.index or not isinstance(e, dict): continue
    s = sides.loc[gid]
    n = {s.home_team: 0, s.away_team: 0}
    for v in (e.get("players") or {}).values():
        if isinstance(v, dict) and v.get("team") in n: n[v["team"]] += 1
    rows.append((gid, n[s.home_team], n[s.away_team], abs(s.point_diff), s.point_diff, s.home_win, s.season))
r = pd.DataFrame(rows, columns=["game_id", "n_home", "n_away", "abs_margin", "margin", "home_win", "season"])
print("games in cache:", len(r))
print("players appearing per team-game: mean %.2f  sd %.2f  min %d  max %d" % (r.n_home.mean(), r.n_home.std(), r.n_home.min(), r.n_home.max()))
print("corr(n_home + n_away, |margin|) = %.3f" % np.corrcoef(r.n_home + r.n_away, r.abs_margin)[0, 1])
print("corr(n_home - n_away, margin)   = %.3f" % np.corrcoef(r.n_home - r.n_away, r.margin)[0, 1])
print("mean players when |margin|<=5: %.2f ; >=20: %.2f" % ((r.n_home + r.n_away)[r.abs_margin <= 5].mean() / 2, (r.n_home + r.n_away)[r.abs_margin >= 20].mean() / 2))
# production feature importances for roster/matchup features
import importlib.util, os
spec = importlib.util.spec_from_file_location("ensemble_model", "prediction_engines/ensemble_model.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
for tgt in ("home_win", "point_diff"):
    m = em.load_model(tgt, "models")
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"\n{tgt}: top 15 importances (split counts, mean over seeds)")
    print(imp.head(15).round(1).to_string())
    sel = imp[[f for f in feats if "roster" in f or "matchup" in f]]
    print(f"  roster+matchup share of importance: {sel.sum() / imp.sum():.3f}")
    print(sel.sort_values(ascending=False).round(1).to_string())
