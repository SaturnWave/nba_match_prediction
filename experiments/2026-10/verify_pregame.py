import pandas as pd, numpy as np, pickle, re
from sklearn.metrics import roc_auc_score
old = pd.read_pickle("output/engineered_dataset_db.pkl")["dataset"]
new = pd.read_pickle("output/engineered_dataset_pregame.pkl")["dataset"]
assert (old.game_id.astype(str).str.zfill(10).values == new.game_id.values).all()
absm = new.point_diff.abs(); y = new.home_win.astype(int)
print("%-32s %8s %8s %8s %8s" % ("feature", "old|r|m", "new|r|m", "oldAUC", "newAUC"))
for m in ["roster_impact_l10_mean", "roster_impact_l10_sum", "roster_form_l6", "roster_form_l3", "matchup_def_quality", "matchup_edge"]:
    for s in ("home", "away", "diff"):
        c = f"{s}_{m}"
        ro = np.corrcoef(old[c], absm)[0,1]; rn = np.corrcoef(new[c], absm)[0,1]
        ao = roc_auc_score(y, old[c]); an = roc_auc_score(y, new[c]); ao = max(ao,1-ao); an = max(an,1-an)
        print("%-32s %8.3f %8.3f %8.3f %8.3f" % (c, ro, rn, ao, an))
print("\nnew extra features:")
for c in [f"{s}_{m}" for m in ("roster_avail_minutes", "roster_missing_impact") for s in ("home","away","diff")]:
    v = new[c]; a = roc_auc_score(y, v); a = max(a, 1-a)
    print("%-32s corr|margin|=%6.3f  AUC=%.3f  corr(point_diff)=%6.3f  mean=%.3f sd=%.3f zero-share=%.3f" % (c, np.corrcoef(v, absm)[0,1], a, np.corrcoef(v, new.point_diff)[0,1], v.mean(), v.std(), (v==0).mean()))
# dressed count vs margin
bp = pd.read_pickle("phonedb_cache/box_player_traditional.pkl")
cm = bp.comment.astype("string").fillna("")
unav = cm.str.contains(r"Injur|Illness|Rest|Health|Not With Team|NWT|Suspen|Personal|DND", case=False).fillna(False)
dressed = bp[~unav.values].groupby(["game_id", "team_abbreviation"]).size()
played = bp[bp["min"].notna()].groupby(["game_id", "team_abbreviation"]).size()
g = new.set_index("game_id").point_diff.abs()
d = dressed.reset_index(name="n").merge(g.rename("absm"), left_on="game_id", right_index=True)
p = played.reset_index(name="n").merge(g.rename("absm"), left_on="game_id", right_index=True)
print(f"\ndressed per team-game: mean {dressed.mean():.2f} sd {dressed.std():.2f}; corr(dressed, |margin|) = {np.corrcoef(d.n, d.absm)[0,1]:.3f}")
print(f"played  per team-game: mean {played.mean():.2f} sd {played.std():.2f}; corr(played,  |margin|) = {np.corrcoef(p.n, p.absm)[0,1]:.3f}")
print("comments flagged unavailable: %d of %d non-empty" % (unav.sum(), (cm != "").sum()))
print(cm[(cm != "") & ~unav.values].str.slice(0, 30).value_counts().head(6).to_string())
# a spot check: find a 2025-26 game where a high-impact player was DND
sub = bp[(bp.game_id.str.startswith("00225")) & unav.values]
ex = sub.merge(new[["game_id", "home_team", "away_team", "home_roster_missing_impact", "away_roster_missing_impact"]], on="game_id").head(5)
print("\nspot check (unavailable players, 2025-26):")
print(ex[["game_id", "team_abbreviation", "player_name", "comment", "home_team", "home_roster_missing_impact", "away_roster_missing_impact"]].to_string(index=False))
