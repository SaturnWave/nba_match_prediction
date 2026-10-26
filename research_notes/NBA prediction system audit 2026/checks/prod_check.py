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
print("rows", len(ds), "seasons", ds.season.value_counts().sort_index().to_dict())
print("game_id prefixes", ds.game_id.str[:3].value_counts().to_dict())
ctx = ds[ds.season != "2025_2026"]; tgt = ds[ds.season == "2025_2026"].sort_values("game_date"); split = int(len(tgt) * 0.75)
train = pd.concat([ctx, tgt.iloc[:split]]).sort_values("game_date"); test = tgt.iloc[split:]
X = lambda d: d[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
blend = em.load_blend(os.path.join(R, "models"))
def brier(y, p): return float(np.mean((p - y) ** 2))
def acc(y, p): return float(np.mean((p > 0.5) == (y == 1)))
for name, d in [("train_all", train), ("train_2024_25", train[train.season == "2024_2025"]), ("train_2025_26_first75", tgt.iloc[:split]), ("heldout", test)]:
    p = blend.predict_proba(X(d))[:, 1]; y = d.home_win.astype(float).values
    print(f"{name:22} n={len(d):5d} blend acc {acc(y,p):.3f} brier {brier(y,p):.4f}")
Xt = X(test); y = test.home_win.astype(float).values
views = blend.views(Xt); p = blend.predict_proba(Xt)[:, 1]
for k, v in views.items(): print(f"  view {k:10} acc {acc(y,v):.3f} brier {brier(y,v):.4f} sd {np.std(v):.3f}")
print(f"  blend      acc {acc(y,p):.3f} brier {brier(y,p):.4f} sd {np.std(p):.3f}")
from sklearn.linear_model import LogisticRegression
lg = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
m = LogisticRegression(C=1e6).fit(lg(p).reshape(-1,1), y); print("  calib slope (logit y ~ logit p):", round(float(m.coef_[0][0]),3), "intercept", round(float(m.intercept_[0]),3))
bins = pd.qcut(p, 5); print("  reliability quintiles:"); print(pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(n=("y","size"), pred=("p","mean"), actual=("y","mean")).round(3).to_string())
rng = np.random.default_rng(0); B = 4000; idx = rng.integers(0, len(y), (B, len(y)))
ab = np.array([acc(y[i], p[i]) for i in idx]); bb = np.array([brier(y[i], p[i]) for i in idx])
print(f"  bootstrap 95% CI acc [{np.percentile(ab,2.5):.3f}, {np.percentile(ab,97.5):.3f}] brier [{np.percentile(bb,2.5):.4f}, {np.percentile(bb,97.5):.4f}]")
odds = pd.read_csv(os.path.join(R, "odds_data", "nba_2025_26_heldout_odds.csv"), dtype={"game_id": str}); odds["game_id"] = odds.game_id.str.zfill(10)
print("odds cols:", list(odds.columns)[:30])
pdf = pd.DataFrame({"game_id": test.game_id.values, "p": p, "y": y, "pd": test.point_diff.values, "tot": test.total_score.values})
import importlib
for t in ("point_diff", "total_score"):
    pdf["m_"+t] = em.load_model(t, os.path.join(R, "models")).predict(Xt)
d = pdf.merge(odds, on="game_id").dropna(subset=["dk_home_prob_novig_close","dk_spread_home_close","dk_total_close"])
mp = d.dk_home_prob_novig_close.values; yy = d.y.values; pp = d.p.values
dbr = (pp-yy)**2 - (mp-yy)**2
ii = rng.integers(0, len(d), (B, len(d)))
print(f"n={len(d)} brier model {np.mean((pp-yy)**2):.4f} market {np.mean((mp-yy)**2):.4f} diff {dbr.mean():+.4f} paired-bootstrap 95% CI [{np.percentile([dbr[i].mean() for i in ii],2.5):+.4f}, {np.percentile([dbr[i].mean() for i in ii],97.5):+.4f}] se {dbr.std(ddof=1)/np.sqrt(len(d)):.4f}")
mm = -d.dk_spread_home_close.values; ae = np.abs(d.m_point_diff.values - d.pd.values) - np.abs(mm - d.pd.values)
print(f"margin MAE diff model-market {ae.mean():+.3f} se {ae.std(ddof=1)/np.sqrt(len(d)):.3f}; sd model margin {np.std(d.m_point_diff):.2f} sd market margin {np.std(mm):.2f} corr {np.corrcoef(d.m_point_diff, mm)[0,1]:.3f}")
te = np.abs(d.m_total_score.values - d.tot.values) - np.abs(d.dk_total_close.values - d.tot.values)
print(f"total MAE diff model-market {te.mean():+.3f} se {te.std(ddof=1)/np.sqrt(len(d)):.3f}; corr model total vs line {np.corrcoef(d.m_total_score, d.dk_total_close)[0,1]:.3f}; sd model total {np.std(d.m_total_score):.2f} sd line {np.std(d.dk_total_close):.2f}")
enc = LogisticRegression(C=1e6).fit(np.column_stack([lg(mp), lg(pp)]), yy)
import statsmodels.api as sm
try:
    res = sm.Logit(yy, sm.add_constant(np.column_stack([lg(mp), lg(pp)]))).fit(disp=0)
    print("encompassing logit y ~ logit(market) + logit(model): coef", np.round(res.params,3), "se", np.round(res.bse,3))
except Exception as e:
    print("encompassing (sklearn) coef", np.round(enc.coef_,3), enc.intercept_, "(statsmodels unavailable)", type(e).__name__)
# feature redundancy on training rows
Xtr = X(train)
c = np.corrcoef(Xtr.values, rowvar=False); np.fill_diagonal(c, 0)
print("pairs |r|>0.95:", int((np.abs(c) > 0.95).sum() // 2), " >0.9:", int((np.abs(c) > 0.9).sum() // 2))
Z = (Xtr - Xtr.mean()) / Xtr.std().replace(0, 1)
ev = np.linalg.eigvalsh(np.cov(Z.values, rowvar=False))[::-1]; cs = np.cumsum(ev) / ev.sum()
print("PCA comps for 90%/95%/99% var:", int(np.searchsorted(cs, .90) + 1), int(np.searchsorted(cs, .95) + 1), int(np.searchsorted(cs, .99) + 1))
# exact linear dependence diff = home - away
nd = 0
for f in feats:
    if f.startswith("diff_"):
        s = f[5:]
        if "home_"+s in feats and "away_"+s in feats:
            if np.allclose(Xtr[f], Xtr["home_"+s] - Xtr["away_"+s], atol=1e-6): nd += 1
print("diff_ columns exactly home-away:", nd, "of", sum(f.startswith("diff_") for f in feats))
# monotone sign stability
def signs(df):
    mg = df.point_diff.astype(float).rank(); out = {}
    for f in feats:
        if not f.startswith("diff_"): continue
        x = pd.to_numeric(df[f], errors="coerce").fillna(0)
        if x.std() == 0: out[f] = 0; continue
        rho = np.corrcoef(x.rank(), mg)[0, 1]; out[f] = (int(np.sign(rho)) if abs(rho) >= 0.02 else 0, round(rho, 3))
    return out
sa = signs(train); h1 = signs(train.iloc[: len(train)//2]); h2 = signs(train.iloc[len(train)//2:])
nz = [f for f in sa if sa[f][0] != 0]
flips = [f for f in sa if isinstance(h1[f], tuple) and isinstance(h2[f], tuple) and h1[f][0]*h2[f][0] == -1]
weak = [f for f in sa if isinstance(sa[f], tuple) and 0.02 <= abs(sa[f][1]) < 0.05]
print("diff_ constrained:", len(nz), "of", len(sa), "| sign flips between halves:", [(f, h1[f][1], h2[f][1]) for f in flips])
print("weakly-signed (|rho|<0.05) but constrained:", [(f, sa[f][1]) for f in weak])
print("unconstrained (|rho|<0.02):", [f for f in sa if isinstance(sa[f], tuple) and sa[f][0] == 0])
neg = [(f, sa[f][1]) for f in nz if sa[f][0] < 0]; print("negative-signed:", neg)
gp = np.minimum(*[ds.sort_values(["game_date","game_id"]).groupby(["season", s]).cumcount().values for s in ("home_team","away_team")])
print("share of games with min(gp)<10 (excluded by burn-in):", round(float((gp < 10).mean()), 3))
