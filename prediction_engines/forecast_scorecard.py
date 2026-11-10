"""
Every forecast this project made before tip-off, against what actually happened.

WHY THIS IS SEPARATE FROM THE BACKTESTS
    The season-to-date numbers printed by forecast_upcoming.py and the third
    section of forecast_vs_market.py are backtests: those rows read each team's
    sheet from the box score of the game being predicted, so the model is told
    who actually played. Useful, but flattering.

    This file scores only what was written down before tip-off, in
    output/forecasts/*.csv, with the stamp that proves it: before_tipoff, set
    when the forecast was made by comparing it to the game's scheduled start.
    Nothing here can improve with hindsight, which is the whole point, and it
    is the only number in the project that answers "is it any good" without a
    caveat attached.

    It is also, for now, tiny. The count is printed first so no one reads a
    trend into a dozen games. It grows by one slate a day and by nothing else.

WHERE THE NUMBERS COME FROM
    forecasts   output/forecasts/forecast_*.csv (forecast_upcoming.py). When a
                game was forecast more than once, the earliest pre-tip-off one
                counts - a later re-score is not a forecast.
    results     game_ids/schedule_<season>.csv, the only file that carries a
                result without the dataset having been rebuilt. Refresh it with
                data_retriaval/fetch_schedule.py before running this.
    the market  odds_data/upcoming_lines_*.csv when a snapshot of that game's
                line exists (quoted before tip-off, so a fair comparison), else
                the closing line in odds_data/odds_history.csv, which is only
                known at tip-off and is marked as such.

    Nothing here is betting advice. The line is a yardstick.

Run:  py prediction_engines/forecast_scorecard.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
FORECAST_DIR = os.path.join(PROJECT_ROOT, "output", "forecasts")
ODDS_DIR = os.path.join(PROJECT_ROOT, "odds_data")
SCHEDULE_GLOB = os.path.join(PROJECT_ROOT, "game_ids", "schedule_*.csv")


def forecasts_before_tipoff():
    """Each game's earliest pre-tip-off forecast; empty when none were made."""
    paths = sorted(glob.glob(os.path.join(FORECAST_DIR, "forecast_*.csv")))
    if not paths:
        return pd.DataFrame()
    made = pd.concat([pd.read_csv(p, dtype={"game_id": str}) for p in paths], ignore_index=True)
    made = made[made["before_tipoff"].astype(str).str.lower() == "true"]
    if made.empty:
        return made
    made["game_id"] = made["game_id"].str.zfill(10)
    return (made.sort_values("generated_at")
            .drop_duplicates("game_id", keep="first")
            .reset_index(drop=True))


def results():
    """{game_id: (home score, away score)} for every game a schedule calls final."""
    rows = []
    for path in sorted(glob.glob(SCHEDULE_GLOB)):
        frame = pd.read_csv(path, dtype={"game_id": str})
        played = frame[(frame["season_type"] == "regular_season")
                       & (frame["status"] == "final")]
        rows.append(played[["game_id", "home_score", "away_score"]])
    if not rows:
        return pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    out = pd.concat(rows, ignore_index=True)
    out["game_id"] = out["game_id"].str.zfill(10)
    return out.dropna(subset=["home_score", "away_score"]).drop_duplicates("game_id")


def market_lines():
    """One line per game, and whether it was quoted before tip-off.

    A snapshot taken while the game was still unplayed wins over the archive:
    it is what the market said at the moment the forecast was made. The
    archive's closing line is only known at tip-off, so a model that beats it
    has done something harder than a model that beats an opening line.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(ODDS_DIR, "upcoming_lines_*.csv"))):
        snap = pd.read_csv(path, dtype={"game_id": str})
        if "game_id" not in snap.columns or "p_home_close" not in snap.columns:
            continue
        snap = snap[["game_id", "spread_close", "total_close", "p_home_close"]].copy()
        snap["line_when"] = "mac oncesi"
        frames.append(snap)
    history_path = os.path.join(ODDS_DIR, "odds_history.csv")
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path, dtype={"game_id": str})
        hist = hist[["game_id", "spread_close", "total_close", "p_home_close"]].copy()
        hist["line_when"] = "kapanis"
        frames.append(hist)
    if not frames:
        return pd.DataFrame(columns=["game_id", "spread_close", "total_close",
                                     "p_home_close", "line_when"])
    lines = pd.concat(frames, ignore_index=True)
    lines["game_id"] = lines["game_id"].str.zfill(10)
    # Snapshots come first, so keep="first" prefers the pre-game quote.
    return lines.dropna(subset=["p_home_close"]).drop_duplicates("game_id", keep="first")


def measure(frame, probability, margin, total):
    """Accuracy, Brier, margin MAE and total MAE on one set of games."""
    won = frame["home_won"].to_numpy()
    probability = np.asarray(probability, dtype=float)
    out = {"mac": int(len(frame)),
           "dogru": int(((probability > 0.5) == (won == 1)).sum()),
           "isabet": float(np.mean((probability > 0.5) == (won == 1))),
           "brier": float(np.mean((probability - won) ** 2))}
    margin = np.asarray(margin, dtype=float)
    usable = ~np.isnan(margin)
    if usable.any():
        out["fark_mae"] = float(np.mean(np.abs(margin[usable]
                                               - frame["real_margin"].to_numpy()[usable])))
    total = np.asarray(total, dtype=float)
    usable = ~np.isnan(total)
    if usable.any():
        out["toplam_mae"] = float(np.mean(np.abs(total[usable]
                                                 - frame["real_total"].to_numpy()[usable])))
    return out


def report_line(label, m):
    text = (f"  {label:7s} {m['dogru']}/{m['mac']} dogru ({m['isabet']:.3f})  "
            f"Brier {m['brier']:.3f}")
    if "fark_mae" in m:
        text += f"  fark MAE {m['fark_mae']:.1f}"
    if "toplam_mae" in m:
        text += f"  toplam MAE {m['toplam_mae']:.1f}"
    print(text)


def main():
    made = forecasts_before_tipoff()
    if made.empty:
        print("tip-off oncesi yapilmis tahmin yok - once forecast_upcoming.py")
        return 0

    played = made.merge(results(), on="game_id", how="inner")
    waiting = len(made) - len(played)
    if played.empty:
        print(f"{len(made)} tahmin var ama hicbirinin sonucu yok. "
              f"Once: py data_retriaval/fetch_schedule.py 2026-27")
        return 0

    played["real_margin"] = played["home_score"] - played["away_score"]
    played["real_total"] = played["home_score"] + played["away_score"]
    played["home_won"] = (played["real_margin"] > 0).astype(int)
    played = played.merge(market_lines(), on="game_id", how="left")
    played = played.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    hit = (played["home_win_prob"] > 0.5) == (played["home_won"] == 1)
    view = pd.DataFrame({
        "tarih": played["game_date"],
        "mac": played["away_team"] + " @ " + played["home_team"],
        "model_p": played["home_win_prob"].round(3),
        "piyasa_p": played["p_home_close"].round(3),
        "sonuc": (played["away_score"].astype(int).astype(str) + "-"
                  + played["home_score"].astype(int).astype(str)),
        "model_fark": played["margin"].round(1),
        "gercek_fark": played["real_margin"].astype(int),
        "model_toplam": played["total"].round(1),
        "gercek_toplam": played["real_total"].astype(int),
        "model": np.where(hit, "dogru", "yanlis"),
    })
    print(f"== Tip-off oncesi yapilmis ve oynanmis {len(played)} tahmin"
          + (f" ({waiting} tahminin sonucu henuz yok)" if waiting else ""))
    with pd.option_context("display.width", 160):
        print(view.to_string(index=False))

    print(f"\n== Toplam ({len(played)} mac)")
    report_line("model", measure(played, played["home_win_prob"],
                                 played["margin"], played["total"]))
    priced = played.dropna(subset=["p_home_close"])
    if len(priced):
        kinds = ", ".join(sorted(priced["line_when"].dropna().unique()))
        report_line("piyasa", measure(priced, priced["p_home_close"],
                                      -priced["spread_close"], priced["total_close"]))
        print(f"  (piyasa {len(priced)} macta var; cizgi: {kinds})")
    print(f"\n  {len(played)} mac hicbir seyi kanitlamaz. Gunde bir slate buyur;"
          f" anlamli bir fark icin birkac yuz mac gerekir.")
    print("  Oranlar bir olcut; burada bahis tavsiyesi yok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
