"""
What filtering the model to its confident games actually buys.

THE QUESTION
    "Keep only the games where the model is at least 60% sure, how good is it
    there?" The number always comes out higher than the all-games number, and
    on its own that means nothing: confidence is the model's own probability
    for its own pick, so on a calibrated model a 70%-confidence game is won
    about 70% of the time by arithmetic, not by skill. Selecting on it and then
    reporting the accuracy of the survivors is measuring the definition.

WHAT IS WORTH MEASURING, AND IS WHAT THIS PRINTS
    1. The calibration gap: observed accuracy minus the average confidence in
       the same band. Zero means the model's stated confidence is honest there.
       Negative means it is overconfident in that band - which is the only way
       a confidence filter can disappoint - and positive means it is leaving
       something on the table.
    2. The coverage cost: how many games the filter throws away. "Accuracy went
       up" is half a sentence without "and I gave up N% of the games".
    3. The market on the same filtered games, where a line is on file, since
       the line is the one external yardstick these forecasts have.

    Definitions, the same everywhere here:
        p           the model's probability that the HOME team wins
        pick        home when p > 0.5, away otherwise
        confidence  max(p, 1 - p), the probability the model gives its own pick

THREE SLICES, AND THEY ARE NOT THE SAME TEST
    forecast    output/forecasts/*.csv with before_tipoff true: real forecasts,
                nothing known that was not known then. The only unqualified
                number here, and far too small to carry a conclusion yet.
    season      this season's played games, scored now. A backtest: each team's
                sheet is read from the box score of the game being predicted,
                so the model is told who actually played.
    held-out    the last 25% of 2025-26, which the models were not fitted on.
                Same backtest regime as above, with the closing line beside it.

Nothing here is betting advice; a line is a yardstick and a feature, never a
thing to bet against.

Run:  py prediction_engines/confidence_buckets.py
      py prediction_engines/confidence_buckets.py --min 0.65
"""
import argparse
import gc
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ODDS_PATH = os.path.join(PROJECT_ROOT, "odds_data", "odds_history.csv")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list_2025_26.json")
TRAINED_SEASON = "2025_2026"
BANDS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
CUMULATIVE = (0.60, 0.65, 0.70)


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confidence(p):
    return np.maximum(p, 1.0 - p)


def correct(p, home_won):
    return (p > 0.5) == (home_won == 1)


def _row(label, n, share, mean_conf, acc):
    """One line, with the gap always beside the noise it has to beat.

    The standard error is built from the CLAIMED confidence, not the observed
    rate: the hypothesis on trial is "the model said 72%", so that is the p
    whose sampling noise matters. A gap smaller than its own standard error is
    printed as `gurultu` rather than as a number, because a reader who sees
    "+19.4" in a 38-game bucket will believe it.
    """
    gap = acc - mean_conf
    se = float(np.sqrt(mean_conf * (1 - mean_conf) / n)) if n else float("nan")
    verdict = "gurultu" if abs(gap) <= se else f"{gap:+.3f}"
    print(f"    {label:12s} {n:7d}  {share:5.1%}      {mean_conf:.3f}    {acc:.3f}   "
          f"{verdict:>8s}   +-{se:.3f}")


def band_table(p, home_won, label):
    """Accuracy by confidence band, and what the filter costs in games."""
    p = np.asarray(p, dtype=float)
    home_won = np.asarray(home_won, dtype=int)
    conf, hit = confidence(p), correct(p, home_won)
    print(f"\n  {label}  ({len(p)} mac, tumunde isabet {hit.mean():.3f})")
    print("    guven            n    pay   ort.guven   isabet       fark      +-SE")
    for low, high in BANDS:
        sel = (conf >= low) & (conf < high)
        band = f"{low:.2f}-{min(high, 1.0):.2f}"
        if not sel.any():
            print(f"    {band:12s}       0      -           -        -          -         -")
            continue
        _row(band, int(sel.sum()), float(sel.mean()), float(conf[sel].mean()),
             float(hit[sel].mean()))
    print("    --- kumulatif (esigin ustundeki her sey) ---")
    for threshold in CUMULATIVE:
        sel = conf >= threshold
        if not sel.any():
            print(f"    >= {threshold:.2f}            0 mac")
            continue
        _row(f">= {threshold:.2f}", int(sel.sum()), float(sel.mean()),
             float(conf[sel].mean()), float(hit[sel].mean()))
    return {"n": int(len(p)), "accuracy_all": float(hit.mean())}


def market_on_same_games(frame, threshold, label):
    """The closing line's record on exactly the games the filter kept."""
    priced = frame.dropna(subset=["p_market"])
    if priced.empty:
        print(f"    {label}: bu maclar icin cizgi yok")
        return
    conf = confidence(priced["p_model"].to_numpy())
    sel = conf >= threshold
    if not sel.any():
        print(f"    {label}: esigi gecen mac yok")
        return
    kept = priced[sel]
    model_hit = correct(kept["p_model"].to_numpy(), kept["home_won"].to_numpy())
    market_hit = correct(kept["p_market"].to_numpy(), kept["home_won"].to_numpy())
    same = (kept["p_model"] > 0.5).to_numpy() == (kept["p_market"] > 0.5).to_numpy()
    print(f"    {label} (guven >= {threshold:.2f}, {len(kept)} mac, cizgisi olan):")
    print(f"      model  {model_hit.sum():3d}/{len(kept)}  ({model_hit.mean():.3f})")
    print(f"      piyasa {market_hit.sum():3d}/{len(kept)}  ({market_hit.mean():.3f})")
    print(f"      ayni tarafi sectikleri: {int(same.sum())}/{len(kept)}"
          + (f"; ayrildiklarinda model {int(model_hit[~same].sum())}"
             f" - piyasa {int(market_hit[~same].sum())}" if (~same).any() else ""))


def score(dataset, features, models):
    """The shown home-win probability, in the dashboard's order of preference."""
    X = dataset[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    blend, classifier = models
    if blend is not None:
        return blend.predict_proba(X)[:, 1]
    return classifier.predict_proba(X)[:, 1]


def load_models():
    ensemble = _load_sibling("ensemble_model")
    return ensemble.load_blend(MODEL_DIR), ensemble.load_model("home_win", MODEL_DIR)


def trained_features(bundle):
    with open(FEATURE_LIST_PATH, encoding="utf-8") as f:
        listed = json.load(f)["features"]
    missing = [c for c in listed if c not in bundle["dataset"].columns]
    if missing:
        raise RuntimeError(f"{len(missing)} feature veri setinde yok: {missing[:5]}")
    return listed


def scheduled_finals():
    """How many regular-season games the newest schedule calls final.

    The yardstick for whether a saved build has fallen behind the season.
    """
    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "game_ids", "schedule_*.csv")))
    if not paths:
        return 0
    frame = pd.read_csv(paths[-1], dtype={"game_id": str})
    return int(((frame["season_type"] == "regular_season")
                & (frame["status"] == "final")).sum())


def odds():
    if not os.path.exists(ODDS_PATH):
        return pd.DataFrame(columns=["game_id", "p_home_close"])
    frame = pd.read_csv(ODDS_PATH, dtype={"game_id": str})
    frame["game_id"] = frame["game_id"].str.zfill(10)
    return frame[["game_id", "p_home_close"]].dropna()


def walk_forward_slice():
    """Slice 0: the biggest honest sample there is, derived from a summary.

    information_ceiling.json keeps 7,495 expanding-origin walk-forward games,
    but only as ten reliability deciles - the per-game probabilities were never
    written to disk, so this is derived, not counted. Each decile is assigned
    whole to the band its mean confidence falls in, which is exact enough for
    the 0.60 cut (no decile straddles it in a way that matters) and NOT exact
    for 0.65 or 0.70, where two deciles straddle each cut. Those two lines are
    printed as the direction they point in, not as numbers to quote.

    Two things this table is not:
      - its top row is the top decile of home-win probability, not "every game
        with confidence above 0.80": decile 0 and decile 8 also reach past 0.80
        and cannot be split out, so that set is nearer 1,050 games.
      - it predates the roster-leak fix (the file is from 2026-09-04, the fix
        from 2026-10-03) and its default dataset was the observed-roster build,
        so every LEVEL here may sit about 0.8 points high. The calibration gaps
        - the column that matters - are far less sensitive to that.
    """
    path = os.path.join(OUTPUT_DIR, "information_ceiling.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    block = payload.get("p_cal")
    if not block or not block.get("reliability"):
        return
    rows = []
    for entry in block["reliability"]:
        p, observed, n = entry["predicted"], entry["observed"], entry["n"]
        rows.append({"n": n, "conf": max(p, 1 - p),
                     "acc": observed if p > 0.5 else 1 - observed})
    total = sum(r["n"] for r in rows)

    print("\n" + "=" * 78)
    print("== 0. WALK-FORWARD  (en buyuk ornek: 7.495 mac, aylik expanding-origin)")
    print("   TURETILMIS: mac mac degil, 10 desilden. 0.60 kesimi saglam;")
    print("   0.65 ve 0.70 kesimlerini iki desil bolduyor, oradaki sayilar yon gosterir.")
    print(f"\n  walk-forward  ({total} mac, tumunde isabet {block['accuracy']:.3f})")
    print("    guven            n    pay   ort.guven   isabet       fark      +-SE")
    for low, high in BANDS:
        sel = [r for r in rows if low <= r["conf"] < high]
        band = f"{low:.2f}-{min(high, 1.0):.2f}"
        if not sel:
            print(f"    {band:12s}       0      -           -        -          -         -")
            continue
        n = sum(r["n"] for r in sel)
        _row(band, n, n / total,
             sum(r["n"] * r["conf"] for r in sel) / n,
             sum(r["n"] * r["acc"] for r in sel) / n)
    print("    --- kumulatif ---")
    for threshold in CUMULATIVE:
        sel = [r for r in rows if r["conf"] >= threshold]
        if not sel:
            continue
        n = sum(r["n"] for r in sel)
        _row(f">= {threshold:.2f}", n, n / total,
             sum(r["n"] * r["conf"] for r in sel) / n,
             sum(r["n"] * r["acc"] for r in sel) / n)
    print("    (en ust satir = ev sahibi olasiliginin en ust desili, 'guven > 0.80'")
    print("     kumesi degil; o kume ~1.050 mac ve desillerden ayristirilamiyor)")


def forecast_slice(args):
    """Slice 1: the forecasts that were written down before tip-off."""
    scorecard = _load_sibling("forecast_scorecard")
    made = scorecard.forecasts_before_tipoff()
    if made.empty:
        print("\n== GERCEK TAHMINLER: hic yok")
        return
    played = made.merge(scorecard.results(), on="game_id", how="inner")
    if played.empty:
        print("\n== GERCEK TAHMINLER: sonucu belli olan yok (once fetch_schedule.py)")
        return
    played["home_won"] = (played["home_score"] > played["away_score"]).astype(int)
    played = played.merge(scorecard.market_lines()[["game_id", "p_home_close"]],
                          on="game_id", how="left")
    print("\n" + "=" * 78)
    print("== 1. GERCEK TAHMINLER  (tip-off oncesi yazilmis; tek kosulsuz dilim)")
    band_table(played["home_win_prob"], played["home_won"], "tip-off oncesi")
    market_on_same_games(
        played.rename(columns={"home_win_prob": "p_model", "p_home_close": "p_market"}),
        args.min, "piyasa karsilastirmasi")
    print(f"\n    {len(played)} mac. Bu buyuklukte isabet orani +-%14 civarinda oynar;"
          f" buradan bir egilim okunmaz.")


def dataset_slice(path, name, selector, regime_note, args, with_market,
                  check_staleness=False):
    """Slices 2 and 3: games scored now from a saved dataset."""
    if not os.path.exists(path):
        print(f"\n== {name}: {os.path.basename(path)} yok, atlandi")
        return
    bundle = pd.read_pickle(path)
    dataset = bundle["dataset"].copy()
    dataset["game_id"] = dataset["game_id"].astype(str).str.zfill(10)
    dataset["game_date"] = pd.to_datetime(dataset["game_date"])
    features = trained_features(bundle)
    rows = selector(dataset)
    # A build is a snapshot. Games played after it was made are simply absent,
    # and a count quoted without that is wrong rather than merely incomplete.
    as_of = bundle.get("as_of")
    stale_note = ""
    if check_staleness and len(rows):
        played_now = scheduled_finals()
        missing = played_now - len(rows)
        if missing > 0:
            stale_note = (f"; takvimde {played_now} oynanmis mac var, "
                          f"{missing} tanesi bu veri setinde YOK")
    if rows.empty:
        print(f"\n== {name}: eslesen mac yok")
        del bundle, dataset
        gc.collect()
        return
    models = load_models()
    frame = pd.DataFrame({
        "game_id": rows["game_id"].to_numpy(),
        "p_model": score(rows, features, models),
        "home_won": rows["home_win"].astype(int).to_numpy(),
    })
    del bundle, dataset, rows, models
    gc.collect()

    print("\n" + "=" * 78)
    print(f"== {name}")
    print(f"   {regime_note}")
    if as_of:
        print(f"   veri seti {as_of} itibariyla kuruldu{stale_note}")
    band_table(frame["p_model"], frame["home_won"], name.lower())
    if with_market:
        frame = frame.merge(odds().rename(columns={"p_home_close": "p_market"}),
                            on="game_id", how="left")
        market_on_same_games(frame, args.min, "piyasa karsilastirmasi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=float, default=0.60,
                        help="piyasa karsilastirmasinda kullanilacak guven esigi")
    args = parser.parse_args()

    walk_forward_slice()
    forecast_slice(args)

    asof = os.path.join(OUTPUT_DIR, "engineered_dataset_asof.pkl")
    dataset_slice(
        asof, "2. BU SEZON (oynanmis maclar)",
        lambda d: d[d.get("is_played", True) & (d["season"] > TRAINED_SEASON)],
        "GERI TEST: kadro, tahmin edilen macin kutu skorundan okundu - model kimin "
        "oynadigini biliyor.", args, with_market=True, check_staleness=True)

    def held_out(dataset):
        target = dataset[dataset["season"] == TRAINED_SEASON].sort_values("game_date")
        return target.iloc[int(len(target) * 0.75):]

    dataset_slice(
        os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl"),
        "3. 2025-26 HELD-OUT (sezonun son %25'i)",
        held_out,
        "GERI TEST, ayni sebeple. Modeller bu maclara fit edilmedi.",
        args, with_market=True)

    print("\n" + "=" * 78)
    print("NASIL OKUNMALI")
    print("  'fark' sutunu = gerceklesen isabet - o aralikta ilan edilen ortalama guven.")
    print("  Sifira yakinsa modelin guveni durust demektir. Guven filtresinin isabeti")
    print("  yukseltmesi tek basina bir sey kanitlamaz: kalibre bir modelde bu tanim")
    print("  geregi olur. Bedeli 'pay' sutununda: esigi gecen maclarin orani.")
    print("  Oranlar bir olcut; burada bahis tavsiyesi yok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
