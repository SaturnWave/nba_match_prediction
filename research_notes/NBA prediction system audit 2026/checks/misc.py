import os, json, numpy as np, pandas as pd
R = r"C:\Users\arcan\OneDrive\Desktop\nba_match_prediction"
ds = pd.read_pickle(os.path.join(R, "output", "engineered_dataset_pregame.pkl"))["dataset"]
for s in ["L10_score", "season_win_pct", "L5_point_margin", "roster_impact_l10_sum", "L10_impact_score_agg"]:
    h, a, d = ds["home_"+s].astype(float), ds["away_"+s].astype(float), ds["diff_"+s].astype(float)
    print(s, "max|diff-(h-a)|", round(float((d-(h-a)).abs().max()),4), "corr", round(float(np.corrcoef(d, h-a)[0,1]),4), "median|d-(h-a)|", round(float((d-(h-a)).abs().median()),5))
wf = json.load(open(os.path.join(R, "output", "walk_forward_7arms.json"), encoding="utf-8"))
b = wf["summary_by_burn_in"].get("10") or list(wf["summary_by_burn_in"].values())[0]
print("7arms sizes", wf.get("arm_sizes"), "cells", b.get("n_cells"))
for arm, dlt in b.get("paired_vs_A", {}).items():
    if dlt: print(f"  {arm:18} acc delta {dlt['mean']:+.4f} se {dlt['standard_error']:.4f} ahead {dlt['cells_ahead']}/{dlt['n']}  brier {b[arm]['brier']['mean']:.4f} vs A {b['A']['brier']['mean']:.4f}")
