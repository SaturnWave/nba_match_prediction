import os, sys, json, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R = r"C:\Users\arcan\OneDrive\Desktop\nba_match_prediction"
spec = importlib.util.spec_from_file_location("ensemble_model", os.path.join(R, "prediction_engines", "ensemble_model.py"))
em = importlib.util.module_from_spec(spec); sys.modules["ensemble_model"] = em; spec.loader.exec_module(em)
blob = pd.read_pickle(os.path.join(R, "output", "engineered_dataset_pregame.pkl"))
ds = blob["dataset"].copy()
ds["game_date"] = pd.to_datetime(ds["game_date"]); ds["game_id"] = ds["game_id"].astype(str).str.zfill(10)
feats = json.load(open(os.path.join(R, "models", "feature_list_2025_26.json"), encoding="utf-8"))["features"]
ctx = ds[ds.season != "2025_2026"]; tgt = ds[ds.season == "2025_2026"].sort_values("game_date"); split = int(len(tgt) * 0.75)
train = pd.concat([ctx, tgt.iloc[:split]]).sort_values("game_date"); test = tgt.iloc[split:]
X = lambda d: d[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
blend = em.load_blend(os.path.join(R, "models"))
from sklearn.linear_model import LogisticRegression
lg = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
p = blend.predict_proba(X(test))[:, 1]
odds = pd.read_csv(os.path.join(R, "odds_data", "nba_2025_26_heldout_odds.csv"), dtype={"game_id": str}); odds["game_id"] = odds.game_id.str.zfill(10)
d = pd.DataFrame({"game_id": test.game_id.values, "p": p, "y": test.home_win.astype(float).values}).merge(odds, on="game_id").dropna(subset=["dk_home_prob_novig_close"])
mp, pp, yy = d.dk_home_prob_novig_close.values, d.p.values, d.y.values
Z = np.column_stack([lg(mp), lg(pp)])
fit = LogisticRegression(C=1e6, max_iter=1000).fit(Z, yy)
rng = np.random.default_rng(1); coefs = []
for _ in range(2000):
    i = rng.integers(0, len(yy), len(yy))
    if len(np.unique(yy[i])) < 2: continue
    coefs.append(LogisticRegression(C=1e6, max_iter=1000).fit(Z[i], yy[i]).coef_[0])
coefs = np.array(coefs)
print("encompassing: coef market %.3f (boot se %.3f), coef model %.3f (boot se %.3f, 95%% CI [%.3f, %.3f])" % (fit.coef_[0][0], coefs[:,0].std(), fit.coef_[0][1], coefs[:,1].std(), np.percentile(coefs[:,1],2.5), np.percentile(coefs[:,1],97.5)))
ms = LogisticRegression(C=1e6).fit(lg(mp).reshape(-1,1), yy); print("market calib slope", round(float(ms.coef_[0][0]),3), "intercept", round(float(ms.intercept_[0]),3))
print("model-market prob corr", round(float(np.corrcoef(pp, mp)[0,1]),3), "sd model p", round(float(np.std(pp)),3), "sd market p", round(float(np.std(mp)),3))
print("market accuracy on held-out", round(float(np.mean((mp>0.5)==(yy==1))),3), "home win rate", round(float(yy.mean()),3))
Xtr = X(train)
c = np.corrcoef(Xtr.values, rowvar=False); np.fill_diagonal(c, 0)
print("pairs |r|>0.95:", int((np.abs(c) > 0.95).sum() // 2), " >0.9:", int((np.abs(c) > 0.9).sum() // 2))
Zs = (Xtr - Xtr.mean()) / Xtr.std().replace(0, 1)
ev = np.linalg.eigvalsh(np.cov(Zs.values, rowvar=False))[::-1]; cs = np.cumsum(ev) / ev.sum()
print("PCA comps for 90%/95%/99% var:", int(np.searchsorted(cs, .90) + 1), int(np.searchsorted(cs, .95) + 1), int(np.searchsorted(cs, .99) + 1))
nd = sum(1 for f in feats if f.startswith("diff_") and "home_"+f[5:] in feats and "away_"+f[5:] in feats and np.allclose(Xtr[f], Xtr["home_"+f[5:]] - Xtr["away_"+f[5:]], atol=1e-6))
print("diff_ columns exactly home-away:", nd, "of", sum(f.startswith("diff_") for f in feats))
def signs(df):
    mg = df.point_diff.astype(float).rank(); out = {}
    for f in feats:
        if not f.startswith("diff_"): continue
        x = pd.to_numeric(df[f], errors="coerce").fillna(0)
        if x.std() == 0: out[f] = (0, 0.0); continue
        rho = float(np.corrcoef(x.rank(), mg)[0, 1]); out[f] = (int(np.sign(rho)) if abs(rho) >= 0.02 else 0, round(rho, 3))
    return out
sa = signs(train); h1 = signs(train.iloc[: len(train)//2]); h2 = signs(train.iloc[len(train)//2:])
nz = [f for f in sa if sa[f][0] != 0]
print("diff_ constrained:", len(nz), "of", len(sa))
print("sign flips between halves:", [(f, h1[f][1], h2[f][1]) for f in sa if h1[f][0]*h2[f][0] == -1])
print("constraint on/off changes between halves:", [(f, h1[f][1], h2[f][1]) for f in sa if (h1[f][0]==0) != (h2[f][0]==0)])
print("weakly-signed (0.02<=|rho|<0.05):", [(f, sa[f][1]) for f in sa if 0.02 <= abs(sa[f][1]) < 0.05])
print("unconstrained:", [(f, sa[f][1]) for f in sa if sa[f][0] == 0])
print("negative-signed:", [(f, sa[f][1]) for f in nz if sa[f][0] < 0])
o = ds.sort_values(["game_date","game_id"])
hg = o.groupby(["season","home_team"]).cumcount()  # placeholder not used
played = {}; gpm = []
for s, h, a in zip(o.season, o.home_team, o.away_team):
    gpm.append(min(played.get((s,h),0), played.get((s,a),0))); played[(s,h)] = played.get((s,h),0)+1; played[(s,a)] = played.get((s,a),0)+1
gpm = np.array(gpm); print("share of games with min(gp)<10 (burn-in excluded):", round(float((gpm<10).mean()),3))
