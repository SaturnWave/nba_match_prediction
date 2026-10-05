import time, pandas as pd, numpy as np, lightgbm as lgb
blob = pd.read_pickle("output/engineered_dataset_db.pkl")
ds, feats = blob["dataset"], blob["features"]
ds["game_date"] = pd.to_datetime(ds["game_date"])
ds = ds.sort_values("game_date")
train = ds[ds.game_date < "2025-11-01"]
n = len(train); a = int(n*0.8); b = int(n*0.9)
X = train[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
P = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
t0 = time.time()
m = lgb.LGBMClassifier(random_state=42, **P)
m.fit(X.iloc[:a], train.home_win.iloc[:a], eval_set=[(X.iloc[a:b], train.home_win.iloc[a:b])], eval_metric="auc",
      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
print(f"clf fit {time.time()-t0:.1f}s, trees {m.n_estimators_}, n_train {a}")
t0 = time.time()
r = lgb.LGBMRegressor(random_state=42, **P)
r.fit(X.iloc[:a], train.point_diff.iloc[:a], eval_set=[(X.iloc[a:b], train.point_diff.iloc[a:b])], eval_metric="mae",
      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
print(f"reg fit {time.time()-t0:.1f}s, trees {r.n_estimators_}")
