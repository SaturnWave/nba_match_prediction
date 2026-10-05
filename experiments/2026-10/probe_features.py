import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
blob = pd.read_pickle("output/engineered_dataset_db.pkl")
ds, feats = blob["dataset"], blob["features"]
print("dataset", ds.shape, "features", len(feats))
print("groups", {k: len(v) for k, v in blob.get("feature_groups", {}).items()})
print("seasons", ds.groupby("season").size().to_dict())
cols = list(ds.columns)
print("\nNON-FEATURE COLUMNS (%d):" % (len(cols) - len(feats)))
print(sorted(set(cols) - set(feats)))
print("\nFEATURE PREFIX COUNTS:")
import collections
pref = collections.Counter(f.split("_")[0] for f in feats)
print(dict(pref))
print("\nALL FEATURES:")
for i in range(0, len(feats), 4):
    print("  " + " | ".join(feats[i:i+4]))
# single-feature AUC vs home_win and |corr| with point_diff
y = ds["home_win"].astype(int).values
pdiff = ds["point_diff"].astype(float).values
rows = []
for f in feats:
    x = pd.to_numeric(ds[f], errors="coerce").fillna(0).values.astype(float)
    if np.nanstd(x) == 0:
        rows.append((f, 0.5, 0.0)); continue
    auc = roc_auc_score(y, x)
    auc = max(auc, 1 - auc)
    c = abs(np.corrcoef(x, pdiff)[0, 1])
    rows.append((f, auc, c))
r = pd.DataFrame(rows, columns=["feature", "auc_abs", "corr_pdiff"]).sort_values("auc_abs", ascending=False)
print("\nTOP 25 single-feature AUC (same-game outcome):")
print(r.head(25).to_string(index=False))
print("\nany feature with AUC>0.75:", (r.auc_abs > 0.75).sum())
