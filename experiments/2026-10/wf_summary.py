import json, numpy as np, pandas as pd
for path in ("output/walk_forward_7arms.json", "output/training_protocol_confirm.json", "output/seed_ensemble.json"):
    try:
        r = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(path, "ERR", e); continue
    print("\n==", path, "keys:", list(r)[:12], "n_cells", r.get("n_cells"))
    cells = r.get("cells") or []
    if not cells: 
        print(json.dumps({k: v for k, v in r.items() if k != "cells"}, indent=1)[:3000]); continue
    rows = []
    for c in cells:
        e = c["by_burn_in"].get("10")
        if not e: continue
        for arm, s in e.items():
            if isinstance(s, dict) and "accuracy" in s:
                rows.append({"month": c["month"], "seed": c["seed"], "arm": arm, "acc": s["accuracy"], "auc": s["auc"], "brier": s["brier"], "n": e["n_test"]})
    df = pd.DataFrame(rows)
    arms = [a for a in df.arm.unique()][:12]
    piv = df[df.arm.isin(arms)].groupby(["arm"]).agg(acc=("acc","mean"), auc=("auc","mean"), brier=("brier","mean"), n_cells=("acc","size"))
    print(piv.round(4).to_string())
    base_arm = "A" if "A" in set(df.arm) else ("base" if "base" in set(df.arm) else arms[0])
    b = df[df.arm == base_arm]
    bym = b.groupby("month")["acc"].agg(["mean","std","count"])
    print(f"\n{base_arm}: per-month accuracy (mean over seeds)")
    print(bym.round(3).to_string())
    # variance decomposition
    month_var = bym["mean"].var()
    seed_var = b.groupby("month")["acc"].var().mean()
    print(f"month-to-month var {month_var:.5f} (sd {np.sqrt(month_var):.4f}), within-month seed var {seed_var:.5f} (sd {np.sqrt(seed_var):.4f})")
