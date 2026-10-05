"""Recompute a few trailing features from the raw per-game columns and compare."""
import pandas as pd, numpy as np
blob = pd.read_pickle("output/engineered_dataset_db.pkl")
ds = blob["dataset"].copy()
ds["game_date"] = pd.to_datetime(ds["game_date"])
ds = ds.sort_values(["game_date", "game_id"]).reset_index(drop=True)

# long table: one row per team-game with score, allowed, win, impact
home = ds[["game_id", "season", "game_date", "home_team", "home_score", "away_score", "home_impact_score_agg"]].rename(
    columns={"home_team": "team", "home_score": "score", "away_score": "allowed", "home_impact_score_agg": "imp"})
home["side"] = "home"
away = ds[["game_id", "season", "game_date", "away_team", "away_score", "home_score", "away_impact_score_agg"]].rename(
    columns={"away_team": "team", "away_score": "score", "home_score": "allowed", "away_impact_score_agg": "imp"})
away["side"] = "away"
long = pd.concat([home, away]).sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
long["win"] = (long["score"] > long["allowed"]).astype(float)
long["margin"] = long["score"] - long["allowed"]
g = long.groupby("team", sort=False)           # NOTE: across seasons, like a plain groupby(team) would be
gs = long.groupby(["team", "season"], sort=False)

def cmp(label, mine, side_feat):
    """compare recomputed series (indexed like long) with stored home_/away_ feature"""
    tmp = long[["game_id", "side"]].copy(); tmp["mine"] = mine.values
    merged = []
    for side in ("home", "away"):
        t = tmp[tmp.side == side].merge(ds[["game_id", f"{side}_{side_feat}"]], on="game_id")
        t = t.rename(columns={f"{side}_{side_feat}": "stored"})
        merged.append(t)
    m = pd.concat(merged)
    ok = m["mine"].notna()
    diff = (m.loc[ok, "mine"] - m.loc[ok, "stored"]).abs()
    print(f"{label:48s} n={ok.sum():6d}  exact(<1e-6)={(diff<1e-6).mean():.4f}  mean|diff|={diff.mean():.4f}  "
          f"stored_when_mine_nan: mean={m.loc[~ok,'stored'].mean():.3f} (n={(~ok).sum()})")
    return m

# hypotheses: shifted rolling (correct) vs unshifted (leak)
for w in (3, 5, 10):
    cmp(f"L{w}_score  = rolling{w} mean of PRIOR games (team, all seasons)",
        g["score"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1)), f"L{w}_score")
    cmp(f"L{w}_score  = rolling{w} PRIOR (team, within season)",
        gs["score"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1)), f"L{w}_score")
    cmp(f"L{w}_score  = rolling{w} INCLUDING current game (leak?)",
        g["score"].transform(lambda s: s.rolling(w, min_periods=1).mean()), f"L{w}_score")
cmp("season_avg_score = expanding mean PRIOR within season",
    gs["score"].transform(lambda s: s.expanding().mean().shift(1)), "season_avg_score")
cmp("season_avg_score = expanding INCLUDING current",
    gs["score"].transform(lambda s: s.expanding().mean()), "season_avg_score")
cmp("L10_win_pct = rolling10 PRIOR win rate",
    g["win"].transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1)), "L10_win_pct")
cmp("season_win_pct = expanding PRIOR within season",
    gs["win"].transform(lambda s: s.expanding().mean().shift(1)), "season_win_pct")
cmp("L10_impact_score_agg = rolling10 PRIOR",
    g["imp"].transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1)), "L10_impact_score_agg")
cmp("L10_impact_score_agg = rolling10 INCLUDING current (leak?)",
    g["imp"].transform(lambda s: s.rolling(10, min_periods=1).mean()), "L10_impact_score_agg")

# streak: sign-run of prior results
def streak_prior(s):
    out, run = [], 0
    for w in s:
        out.append(run)
        run = run + 1 if (w == 1 and run >= 0) else (run - 1 if (w == 0 and run <= 0) else (1 if w == 1 else -1))
    return pd.Series(out, index=s.index, dtype=float)
cmp("streak = signed run of PRIOR results", g["win"].transform(streak_prior), "streak")

# roster impact: how does it relate to same-game impact vs previous-game impact?
for side in ("home", "away"):
    f = ds[f"{side}_roster_impact_l10_mean"]
    same = ds[f"{side}_impact_score_agg"]
    print(f"\n{side}: corr(roster_impact_l10_mean, SAME-game team impact) = {np.corrcoef(f, same)[0,1]:.3f}")
    print(f"{side}: corr(roster_impact_l10_sum,  SAME-game team impact) = {np.corrcoef(ds[f'{side}_roster_impact_l10_sum'], same)[0,1]:.3f}")
    print(f"{side}: corr(L10_impact_score_agg,   SAME-game team impact) = {np.corrcoef(ds[f'{side}_L10_impact_score_agg'], same)[0,1]:.3f}")
    print(f"{side}: corr(roster_impact_l10_sum,  SAME-game |point_diff|) = {np.corrcoef(ds[f'{side}_roster_impact_l10_sum'], ds['point_diff'].abs())[0,1]:.3f}")
    print(f"{side}: corr(roster_impact_l10_sum,  home_win) = {np.corrcoef(ds[f'{side}_roster_impact_l10_sum'], ds['home_win'])[0,1]:.3f}")

# cold-start: how many rows have zeroed rolling features
cold = (ds["home_L10_score"] == 0) | (ds["away_L10_score"] == 0)
print(f"\nrows with a zero L10_score (cold start, fillna(0)): {cold.sum()} / {len(ds)}")
print("season_avg_score==0 rows:", int(((ds['home_season_avg_score']==0)|(ds['away_season_avg_score']==0)).sum()))
print("roster_impact_l10_mean==0 rows:", int(((ds['home_roster_impact_l10_mean']==0)|(ds['away_roster_impact_l10_mean']==0)).sum()))
print("matchup_def_quality==0 rows:", int(((ds['home_matchup_def_quality']==0)|(ds['away_matchup_def_quality']==0)).sum()))
