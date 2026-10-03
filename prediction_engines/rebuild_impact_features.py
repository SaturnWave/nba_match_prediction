"""
Rebuild every impact-derived column of the production dataset from a new
impact cache, without rerunning the whole dataset builder.

WHAT DEPENDS ON THE IMPACT CACHE
    Two families, 24 of the 196 production features:
      * team aggregates: home_/away_impact_score_agg and impact_score_diff are
        the per-game team sums; FeatureEngineer rolls them into
        L3/L5/L10_impact_score_agg and season_avg_impact_score_agg for each
        side plus the diffs (12 columns);
      * the pre-game roster family (roster_impact_l10_mean, roster_impact_l10_sum,
        roster_form_l6, roster_form_l3 x home/away/diff), built by
        pregame_roster.py from the per-player values, plus its two extras
        (roster_avail_minutes is minutes-only and does not change;
        roster_missing_impact does).
    Everything else - box-score rolling stats, streaks, head-to-head, matchup
    quality - never touches the cache and is copied through untouched.

    The rolling columns are recomputed with FeatureEngineer's exact recipe
    (per team across seasons, rolling(w, min_periods=1).mean().shift(1);
    season average as the expanding mean shifted by one row so a season's
    first game carries the previous season's final value). The recipe is
    checked here against the stored v4 columns before it is applied to the
    new values: if it cannot reproduce the old columns from the old sums, it
    has no business producing the new ones.

Run:  py prediction_engines/rebuild_impact_features.py --cache game_impact_cache_v5.pkl
         [--in output/engineered_dataset_pregame.pkl] [--out output/engineered_dataset_pregame_v5.pkl]
"""
import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_IN = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame_v5.pkl")
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v5.pkl")

STAT = "impact_score_agg"
WINDOWS = (3, 5, 10)
ROLLING_COLUMNS = ([f"{s}_L{w}_{STAT}" for w in WINDOWS for s in ("home", "away", "diff")]
                   + [f"{s}_season_avg_{STAT}" for s in ("home", "away", "diff")])


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def team_sums(cache):
    rows = [(gid, e["home_impact"], e["away_impact"]) for gid, e in cache.items()
            if isinstance(e, dict)]
    return pd.DataFrame(rows, columns=["game_id", f"home_{STAT}", f"away_{STAT}"])


def rolling_impact(ds):
    """FeatureEngineer's rolling recipe for the one stat, over the whole frame."""
    df = ds.sort_values("game_date").copy()
    frames = []
    for team_col, side in (("home_team_id", "home"), ("away_team_id", "away")):
        t = df[["game_id", "game_date", "season", team_col, f"{side}_{STAT}"]].rename(
            columns={team_col: "team_id", f"{side}_{STAT}": STAT})
        frames.append(t)
    long = pd.concat(frames).dropna(subset=["team_id"])
    long["team_id"] = long["team_id"].astype(int)
    out = []
    for _team, tg in long.groupby("team_id", sort=False):
        tg = tg.sort_values("game_date").drop_duplicates("game_id")
        feats = pd.DataFrame({"game_id": tg["game_id"].values, "team_id": tg["team_id"].values})
        for w in WINDOWS:
            feats[f"L{w}_{STAT}"] = tg[STAT].rolling(w, min_periods=1).mean().shift(1).values
        feats[f"season_avg_{STAT}"] = (tg.groupby("season")[STAT].expanding().mean()
                                       .reset_index(level=0, drop=True).shift(1).values)
        out.append(feats)
    allfeat = pd.concat(out, ignore_index=True)
    result = ds[["game_id", "home_team_id", "away_team_id"]].copy()
    for side in ("home", "away"):
        merged = result[["game_id", f"{side}_team_id"]].merge(
            allfeat, left_on=["game_id", f"{side}_team_id"], right_on=["game_id", "team_id"],
            how="left")
        for col in [c for c in allfeat.columns if c not in ("game_id", "team_id")]:
            result[f"{side}_{col}"] = merged[col].values
    for w in WINDOWS:
        result[f"diff_L{w}_{STAT}"] = result[f"home_L{w}_{STAT}"] - result[f"away_L{w}_{STAT}"]
    result[f"diff_season_avg_{STAT}"] = (result[f"home_season_avg_{STAT}"]
                                         - result[f"away_season_avg_{STAT}"])
    return result[["game_id"] + ROLLING_COLUMNS].fillna(0.0)


def check_recipe(ds):
    """The recipe must reproduce the stored columns from the stored sums."""
    recomputed = rolling_impact(ds).set_index("game_id")
    stored = ds.set_index("game_id")[ROLLING_COLUMNS]
    worst = 0.0
    for col in ROLLING_COLUMNS:
        diff = (recomputed[col] - stored[col]).abs()
        worst = max(worst, float(diff.max()))
    return worst


def rebuild(in_path, cache_path, out_path, verbose=True):
    blob = pd.read_pickle(in_path)
    ds = blob["dataset"].copy()
    ds["game_id"] = ds["game_id"].astype(str).str.zfill(10)
    ds["game_date"] = pd.to_datetime(ds["game_date"])

    worst = check_recipe(ds)
    if verbose:
        print(f"  tarif kontrolu: eski toplamlardan eski sutunlar, en buyuk fark {worst:.2e}")
    if worst > 1e-6:
        raise RuntimeError("rolling tarifi eski sutunlari yeniden uretemedi; durduruldu")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    sums = team_sums(cache)
    ds = ds.drop(columns=[f"home_{STAT}", f"away_{STAT}", "impact_score_diff"], errors="ignore")
    ds = ds.merge(sums, on="game_id", how="left")
    missing = int(ds[f"home_{STAT}"].isna().sum())
    ds[[f"home_{STAT}", f"away_{STAT}"]] = ds[[f"home_{STAT}", f"away_{STAT}"]].fillna(0.0)
    ds["impact_score_diff"] = ds[f"home_{STAT}"] - ds[f"away_{STAT}"]
    if verbose:
        print(f"  takim toplamlari: {len(sums):,} mac, {missing} mac onbellekte yok")

    rolled = rolling_impact(ds)
    ds = ds.drop(columns=ROLLING_COLUMNS).merge(rolled, on="game_id", how="left")

    pregame = _load_sibling("pregame_roster")
    listings = pregame.load_listings()
    impacts = pregame.load_impacts(cache_path)
    def_totals, off_totals = pregame.load_matchup_totals()
    roster = pregame.build_pregame_features(ds, listings, impacts, def_totals, off_totals, verbose)
    ds = pregame.replace_roster_features(ds, roster)

    blob = dict(blob)
    blob["dataset"] = ds
    blob["impact_cache"] = os.path.basename(cache_path)
    pd.to_pickle(blob, out_path)
    if verbose:
        print(f"Yazildi: {out_path}  ({ds.shape[0]} mac x {ds.shape[1]} sutun)")
    return blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    rebuild(args.in_path, args.cache, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
