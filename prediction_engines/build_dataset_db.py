"""
Build the engineered dataset from MariaDB instead of the CSV tree.

WHAT THIS REUSES AND WHAT IT REPLACES
    Only the SOURCE of the per-game rows changes. Roster-impact features, the
    matchup module, and FeatureEngineer are the same objects the CSV pipeline
    uses, called in the same order, so the two datasets are comparable feature
    for feature. predict_2025_2026.py is not modified at all â€” that keeps the
    csv-pipeline and db-pipeline branches cleanly separable, and means a bug
    found here cannot be a bug introduced into the CSV path.

    Replaced: GameDataLoader's per-game CSV reads, which take ~100 minutes for
    10,749 games. db_source.load_master_frame returns the same rows in 1.7
    seconds.

    Rebuilt rather than reused: NBAPredictor._player_records. The CSV path
    fills it as a side effect of walking every game; here it is reconstructed
    from the impact cache, which holds the same per-player values keyed by
    game_id. _build_player_history and _add_roster_features then run untouched.

WHAT IS NEW IN THE OUTPUT
    Five columns the CSV path never produced: home_rest, away_rest, rest_diff,
    home_b2b, away_b2b. _select_features works from a prefix whitelist that
    does not know about them, so they are returned as a SEPARATE list rather
    than smuggled into the base feature set â€” a caller that wants them has to
    ask, and an arm that does not want them is unaffected.

Run:  py prediction_engines/build_dataset_db.py [--seasons 2019_2020 ...]
Output: output/engineered_dataset_db.pkl   {dataset, features, rest_features, seasons}
"""
import os
import sys
import time
import pickle
import argparse
import importlib.util

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(PROJECT_ROOT, "prediction_engines")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def player_records_from_cache(master, cache_path=IMPACT_CACHE):
    """Rebuild NBAPredictor._player_records from the impact cache.

    One record per (player, game) carrying the player's impact and the team
    they played for, which is what _build_player_history needs to compute
    trailing form. A player whose team is missing or does not match either side
    of the game is dropped: the trailing windows are per-player, but the roster
    aggregation joins on (game_id, team), so a stray tricode would attach a
    player to a team that did not field them.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    meta = master[["game_id", "game_date", "season", "home_team", "away_team"]]
    records = []
    for gid, gdate, season, home, away in meta.itertuples(index=False):
        entry = cache.get(gid)
        if not isinstance(entry, dict):
            continue
        for key, value in (entry.get("players") or {}).items():
            if not isinstance(value, dict):
                continue
            team = value.get("team")
            if team not in (home, away):
                continue
            # Identity is the person id, not the surname. _build_player_history
            # groups by this column to build trailing form, and surnames are not
            # unique - two players sharing one would have their histories merged.
            person_id = value.get("person_id")
            records.append({"player": person_id if person_id is not None else str(key),
                            "player_name": value.get("name") or str(key),
                            "team": team, "game_id": gid,
                            "game_date": gdate, "season": season,
                            "impact": float(value.get("impact", 0.0))})
    return records


def build(seasons=None, verbose=True):
    """Return (dataset, base_features, rest_features, predictor)."""
    db = _load_sibling("db_source")
    predictor_module = _load_sibling("predict_2025_2026")

    t0 = time.time()
    master = db.load_master_frame(seasons=seasons, verbose=verbose)
    if master.empty:
        raise RuntimeError("veritabanindan hic mac gelmedi")
    master = db.attach_impact(master, cache_path=IMPACT_CACHE)
    if verbose:
        print(f"  master: {master.shape} ({time.time() - t0:.1f} sn)")

    predictor = predictor_module.NBAPredictor()
    predictor._player_records = player_records_from_cache(master)
    if verbose:
        print(f"  oyuncu-mac kaydi: {len(predictor._player_records):,}")

    master = predictor._add_roster_features(master)

    add_matchup_features = getattr(predictor_module, "add_matchup_features", None)
    if add_matchup_features is not None:
        try:
            master = add_matchup_features(master, predictor_module.BASE_DATA_DIR)
        except Exception as exc:  # noqa: BLE001 - matchups are optional, never fatal
            print(f"  [warn] matchup feature'lari atlandi: {exc}")

    dataset = predictor.fe.engineer(master.copy()).fillna(0)
    predictor.master_df = master
    predictor.dataset = dataset

    base_features = predictor._select_features()
    rest_features = [c for c in db.REST_COLUMNS if c in dataset.columns]
    if verbose:
        matchup = [f for f in base_features if "matchup_" in f]
        print(f"  dataset: {dataset.shape[0]} mac x {dataset.shape[1]} sutun")
        print(f"  temel feature: {len(base_features)} ({len(matchup)} matchup)")
        print(f"  ek rest/b2b feature: {rest_features}")
        print(f"  toplam sure: {time.time() - t0:.1f} sn")
    return dataset, base_features, rest_features, predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--out", default=DATASET_PATH)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset, features, rest_features, _ = build(args.seasons)

    pd.to_pickle({"dataset": dataset, "features": features,
                  "rest_features": rest_features,
                  "seasons": sorted(dataset["season"].unique().tolist())},
                 args.out)
    print(f"\nYazildi: {args.out}")
    print(dataset.groupby("season").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
