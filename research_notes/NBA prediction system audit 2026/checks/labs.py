import json, numpy as np, os, collections
O = r"C:\Users\arcan\OneDrive\Desktop\nba_match_prediction\output"
def load(n):
    with open(os.path.join(O, n), encoding="utf-8") as f: return json.load(f)

def per_month(cells, arm, metric):
    d = collections.OrderedDict()
    for c in cells:
        a = c["arms"].get(arm)
        if not a: continue
        v = a.get(metric) if not metric.startswith("blend3.") else a.get("blend3", {}).get(metric.split(".")[1])
        if v is None: continue
        d.setdefault(c["month"], []).append((v, c["n_test"]))
    return d

def paired_cluster(cells, arm, base, metric):
    by = collections.defaultdict(list); flat=[]
    for c in cells:
        A, B = c["arms"].get(arm), c["arms"].get(base)
        if not A or not B: continue
        get = (lambda e: e.get(metric)) if not metric.startswith("blend3.") else (lambda e: e.get("blend3", {}).get(metric.split(".")[1]))
        if get(A) is None or get(B) is None: continue
        d = get(A) - get(B); flat.append(d); by[c["month"]].append(d)
    flat = np.array(flat); mm = np.array([np.mean(v) for v in by.values()])
    return dict(mean=flat.mean(), naive_se=flat.std(ddof=1)/np.sqrt(len(flat)), n_cells=len(flat),
                cluster_se=mm.std(ddof=1)/np.sqrt(len(mm)), n_months=len(mm), months_ahead=int((mm>0).sum()))

for name in ["impact_v5_lab.json", "consistency_lab_round2.json", "consistency_lab.json"]:
    r = load(name); cells = r["cells"]
    arms = list(cells[0]["arms"].keys())
    print("=====", name, "n_cells", len(cells), "arms", arms)
    months = sorted(set(c["month"] for c in cells))
    print("months", len(months), months[0], "->", months[-1], months)
    tot = sum(c["n_test"] for c in cells if c["seed"] == cells[0]["seed"])
    print("test games (one seed):", tot)
    for arm in arms[:3]:
        pm = per_month(cells, arm, "accuracy")
        pmb = per_month(cells, arm, "blend3.accuracy")
        row = []
        for m, v in pm.items():
            b = pmb.get(m)
            row.append(f"{m}:n={v[0][1]} c={np.mean([x[0] for x in v]):.3f}" + (f" b={np.mean([x[0] for x in b]):.3f}" if b else ""))
        print(" ", arm, " | ".join(row))
    base = r.get("summary", {}).get("_base", arms[0])
    for arm in arms:
        if arm == base: continue
        for metric in ["accuracy", "brier", "blend3.accuracy", "blend3.brier", "point_diff_mae"]:
            try:
                p = paired_cluster(cells, arm, base, metric)
                print(f"  {arm} vs {base} {metric}: mean {p['mean']:+.4f} naive_se {p['naive_se']:.4f} cluster_se(month) {p['cluster_se']:.4f} months_ahead {p['months_ahead']}/{p['n_months']}")
            except Exception as e:
                pass
