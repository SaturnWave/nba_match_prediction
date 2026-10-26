import json, os, re, collections, numpy as np, pandas as pd
R = r"C:\Users\arcan\OneDrive\Desktop\nba_match_prediction"
O = os.path.join(R, "output")
fl = json.load(open(os.path.join(R, "models", "feature_list_2025_26.json"), encoding="utf-8"))
feats = fl["features"]
print("n features", len(feats))
groups = collections.Counter()
for f in feats:
    g = re.sub(r"^(home|away|diff)_", "", f)
    g = re.sub(r"L\d+_|l\d+_|_l\d+|_L\d+", "", g)
    groups[f.split("_")[0]] += 1
print("prefix counts", groups)
print("non home/away/diff:", [f for f in feats if not f.startswith(("home_","away_","diff_"))])
print("diff_ count", sum(f.startswith("diff_") for f in feats))
# distinct stems
stems = sorted(set(re.sub(r"^(home|away|diff)_", "", f) for f in feats))
print("stems", len(stems)); print(stems)
# walk forward burn-in
wf = json.load(open(os.path.join(O, "walk_forward_2025_26.json"), encoding="utf-8"))
for b, blk in wf["summary_by_burn_in"].items():
    a = blk.get("A", {})
    print("burn", b, "cells", blk["n_cells"], "naive", round(blk["naive_baseline"],4), "A acc", round(a["accuracy"]["mean"],4), "auc", round(a["auc"]["mean"],4), "brier", round(a["brier"]["mean"],4))
# burn-in 0 vs 10 restricted: games with gp<10
months0 = collections.defaultdict(list)
for c in wf["cells"]:
    e0, e10 = c["by_burn_in"].get("0"), c["by_burn_in"].get("10")
    if e0 and e10:
        n0, n10 = e0["n_test"], e10["n_test"]
        if n0 > n10:
            a0, a10 = e0["A"]["accuracy"], e10["A"]["accuracy"]
            early = (a0*n0 - a10*n10)/(n0-n10)
            months0[c["month"]].append((n0-n10, early, a10))
print("early-season (gp<10) implied accuracy by month (arm A):")
for m, v in sorted(months0.items()):
    print(" ", m, "n_early", v[0][0], "acc_early %.3f" % np.mean([x[1] for x in v]), "acc_rest %.3f" % np.mean([x[2] for x in v]))
print("wf months:", sorted(set(c["month"] for c in wf["cells"]))[:5], "...", len(set(c["month"] for c in wf["cells"])))
for name in ["information_ceiling.json", "staleness.json", "seed_ensemble.json", "two_arm_comparison.json"]:
    d = json.load(open(os.path.join(O, name), encoding="utf-8"))
    s = {k: v for k, v in d.items() if k not in ("folds", "cells", "rows", "per_game")}
    txt = json.dumps(s)[:1800]
    print("=====", name, txt)
