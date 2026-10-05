import json, numpy as np, pandas as pd
r = json.load(open("output/consistency_lab.json", encoding="utf-8"))
cells = r["cells"]; arms = list(r["arm_sizes"])
def get(c, arm, m):
    e = c["arms"].get(arm)
    if e is None: return None
    if m.startswith("blend3."): return e.get("blend3", {}).get(m[7:])
    return e.get(m)
def paired(arm, base, m):
    d = [get(c, arm, m) - get(c, base, m) for c in cells if get(c, arm, m) is not None and get(c, base, m) is not None]
    d = np.array(d); 
    return d.mean(), d.std(ddof=1)/np.sqrt(len(d)), (d > 0).sum(), len(d)
print(f"{len(cells)} cells; months: {sorted({c['month'] for c in cells})}")
print("\nPAIRED vs pregame (mean ± se, cells ahead/n)")
for m in ["accuracy", "auc", "brier", "log_loss", "point_diff_mae", "contradiction_rate", "blend3.accuracy", "blend3.auc", "blend3.brier"]:
    row = []
    for arm in arms:
        if arm == "pregame": continue
        try:
            mu, se, ah, n = paired(arm, "pregame", m)
            row.append(f"{arm}: {mu:+.4f}±{se:.4f} ({ah}/{n})")
        except Exception: pass
    print(f"  {m:18} " + " | ".join(row))
print("\nblend3 means:")
for arm in arms:
    v = [get(c, arm, "blend3.accuracy") for c in cells if get(c, arm, "blend3.accuracy") is not None]
    b = [get(c, arm, "blend3.brier") for c in cells if get(c, arm, "blend3.brier") is not None]
    if v: print(f"  {arm:13} acc {np.mean(v):.4f}  brier {np.mean(b):.4f}  n={len(v)}")
# where did the observed roster help most? per month delta observed-pregame on accuracy and MAE
rows = []
for c in cells:
    rows.append({"month": c["month"], "seed": c["seed"], "d_acc": get(c,"observed","accuracy")-get(c,"pregame","accuracy"),
                 "d_mae": get(c,"observed","point_diff_mae")-get(c,"pregame","point_diff_mae"), "n": c["n_test"]})
df = pd.DataFrame(rows).groupby("month").agg(d_acc=("d_acc","mean"), d_mae=("d_mae","mean"), n=("n","first"))
print("\nobserved - pregame by month (acc, margin MAE):")
print(df.round(3).to_string())
# trees
for arm in arms:
    t = [get(c, arm, "n_trees") for c in cells]; s = [get(c, arm, "seconds") for c in cells]
    print(f"  {arm:13} trees mean {np.mean(t):.0f}  sec/cell {np.mean(s):.1f}")
