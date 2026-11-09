"""
The model's forecasts next to the betting market's line, for the season in progress.

WHY THE LINE IS HERE
    A line is the only forecast of an NBA game that is published before
    tip-off, priced by people who lose money when it is wrong, and archived
    afterwards. That makes it the one external yardstick this model can be held
    against on the same games in the same units. Nothing here looks for prices
    to bet against, and none of it is betting advice.

THREE SECTIONS, AND WHY THEY ARE NOT THE SAME TEST
  1. Games not played yet - the forecast beside the market's line as it stands
     right now. Both sides are pre-game, so this is the fair comparison. The
     lines are fetched fresh on every run and every response is kept as a dated
     snapshot, because a line is a fact only at the moment it was quoted. For an
     unplayed game ESPN's `close` fields hold the line quoted at fetch time, not
     a closing line.
  2. Forecasts that were made before tip-off on games since played - the honest
     record. It only grows as the season runs and starts far too small to mean
     anything, so the game count is printed first.
  3. Every played game of the season that has a line. This one flatters the
     model and says so: those rows read each team's sheet from the box score of
     the very game being predicted, so the model is told who actually played
     while the market only knew who was expected to. A sanity check on the
     season, not evidence about the market.

Inputs: output/forecasts/*.csv, output/engineered_dataset_asof.pkl,
        odds_data/odds_history.csv (data_retriaval/odds_build.py), ESPN live
Run:    py prediction_engines/forecast_vs_market.py
Output: printed report, odds_data/upcoming_lines_<fetched, UTC>.csv
"""
import datetime
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_RETRIEVAL = os.path.join(PROJECT_ROOT, "data_retriaval")
sys.path.insert(0, DATA_RETRIEVAL)

from odds_common import RAW_DIR, Refused, cached_json, nba_code  # noqa: E402  (needs sys.path above)

ODDS_DIR = os.path.join(PROJECT_ROOT, "odds_data")
FORECAST_DIR = os.path.join(PROJECT_ROOT, "output", "forecasts")
ASOF_PATH = os.path.join(PROJECT_ROOT, "output", "engineered_dataset_asof.pkl")
HISTORY_PATH = os.path.join(ODDS_DIR, "odds_history.csv")
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
REGULAR_SEASON = 2
LINE_COLUMNS = ["game_id", "spread_close", "total_close", "p_home_close"]


def _load_module(directory, name):
    """Import a project script by path - these are scripts, not an installed package."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_lines(dates, stamp):
    """One median line per game on `dates`, fetched now, from the home team's side.

    Each response goes to a folder of its own per run instead of the shared
    cache the history build uses. Reusing odds_data/raw/espn/odds/ would file a
    line quoted hours before tip-off as that game's closing line, and the
    history would carry it for good.
    """
    espn = _load_module(DATA_RETRIEVAL, "odds_espn_lines")
    snapshot = os.path.join(RAW_DIR, "espn", "snapshot", stamp)
    rows = []
    for date in dates:
        board = cached_json(SCOREBOARD_URL, os.path.join(snapshot, f"scoreboard_{date}.json"),
                            params={"dates": date.replace("-", "")})
        for event in board.get("events", []):
            if event.get("season", {}).get("type") != REGULAR_SEASON:
                continue
            sides = {c["homeAway"]: c for c in event["competitions"][0]["competitors"]}
            payload = cached_json(espn.ODDS_URL.format(event=event["id"]),
                                  os.path.join(snapshot, f"odds_{event['id']}.json"))
            books = [espn.book_line(item) for item in payload.get("items", [])
                     if "live" not in item.get("provider", {}).get("name", "").lower()]
            row = {"game_date": date, "books": len(books),
                   "home_team": nba_code(sides["home"]["team"]["abbreviation"]),
                   "away_team": nba_code(sides["away"]["team"]["abbreviation"])}
            for field in espn.FIELDS:
                values = [book[field] for book in books if book[field] is not None]
                row[field] = float(np.median(values)) if values else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def newest_forecast():
    """The most recent forecast file, or None when none was ever written."""
    paths = sorted(glob.glob(os.path.join(FORECAST_DIR, "forecast_*.csv")))
    return pd.read_csv(paths[-1], dtype={"game_id": str}) if paths else None


def upcoming_section(stamp):
    """Section 1: the newest forecast beside the market as it stands."""
    forecast = newest_forecast()
    if forecast is None:
        print("tahmin dosyasi yok - once forecast_upcoming.py")
        return
    dates = sorted(forecast["game_date"].unique())
    try:
        market = live_lines(dates, stamp)
    except Refused as refusal:
        print(f"piyasa oranlari alinamadi: {refusal}")
        return
    joined = forecast.merge(market, on=["game_date", "home_team", "away_team"], how="left")
    out_path = os.path.join(ODDS_DIR, f"upcoming_lines_{stamp}.csv")
    joined.to_csv(out_path, index=False)

    view = pd.DataFrame({
        "mac": joined["away_team"] + " @ " + joined["home_team"],
        "model_p": joined["home_win_prob"].round(3),
        "piyasa_p": joined["p_home_close"].round(3),
        "model_fark": joined["margin"].round(1),
        "piyasa_fark": (-joined["spread_close"]).round(1),
        "model_toplam": joined["total"].round(1),
        "piyasa_toplam": joined["total_close"].round(1),
        "kitap": joined["books"],
    })
    print(f"== Oynanmamis maclar ({', '.join(dates)}); oranlar {stamp} UTC'de cekildi")
    with pd.option_context("display.width", 150):
        print(view.to_string(index=False))
    both = joined.dropna(subset=["p_home_close", "spread_close"])
    if len(both):
        same = int(((both["home_win_prob"] > 0.5) == (both["p_home_close"] > 0.5)).sum())
        print(f"   kazanan secimi {same}/{len(both)} macta ayni taraf; "
              f"ortalama |model p - piyasa p| {np.mean(np.abs(both['home_win_prob'] - both['p_home_close'])):.3f}; "
              f"fark tahmini ortalama {np.mean(np.abs(both['margin'] + both['spread_close'])):.1f} sayi ayri")
    print(f"   Yazildi: {out_path}")


def measure(games, probability, margin, total):
    """Accuracy, Brier, margin MAE and total MAE for one forecaster on one set of games."""
    won = games["home_won"].to_numpy()
    out = {"mac": int(len(games)),
           "isabet": float(np.mean((probability > 0.5) == (won == 1))),
           "brier": float(brier_score_loss(won, probability)),
           "fark_mae": float(np.mean(np.abs(margin - games["real_margin"].to_numpy())))}
    usable = ~pd.isna(total)
    if usable.any():
        out["toplam_mae"] = float(np.mean(np.abs(np.asarray(total)[usable]
                                                 - games["real_total"].to_numpy()[usable])))
    return out


def report_line(label, metrics):
    text = (f"   {label:7s} mac {metrics['mac']:3d}  isabet {metrics['isabet']:.3f}  "
            f"Brier {metrics['brier']:.3f}  fark MAE {metrics['fark_mae']:.1f}")
    if "toplam_mae" in metrics:
        text += f"  toplam MAE {metrics['toplam_mae']:.1f}"
    print(text)


def forecasts_made_before_tipoff():
    """Every game's earliest pre-tip-off forecast; a later re-score is not a forecast."""
    paths = sorted(glob.glob(os.path.join(FORECAST_DIR, "forecast_*.csv")))
    if not paths:
        return pd.DataFrame()
    made = pd.concat([pd.read_csv(p, dtype={"game_id": str}) for p in paths], ignore_index=True)
    made = made[made["before_tipoff"].astype(str).str.lower() == "true"]
    return made.sort_values("generated_at").drop_duplicates("game_id", keep="first")


def season_sections():
    """Sections 2 and 3: the pre-game record so far, then the season as a whole."""
    history = pd.read_csv(HISTORY_PATH, dtype={"game_id": str})
    blob = pd.read_pickle(ASOF_PATH)
    dataset, features = blob["dataset"], blob["production_features"]
    forecaster = _load_module(HERE, "forecast_upcoming")
    model_rows, season = forecaster.season_so_far(dataset, features, forecaster.load_models())
    if not len(model_rows):
        print("\noynanmis mac yok")
        return
    # Same slice season_so_far used, in the same order, so the real total lines up.
    played = dataset[dataset["is_played"] & (dataset["season"] == season)]
    model_rows = model_rows.copy()
    model_rows["real_total"] = played["total_score"].to_numpy()
    lines = history.loc[history["season"] == season, LINE_COLUMNS]

    made = forecasts_made_before_tipoff()
    record = (made.merge(model_rows[["game_id", "home_won", "real_margin", "real_total"]], on="game_id")
              .merge(lines, on="game_id", how="left") if len(made) else pd.DataFrame())
    print(f"\n== Tip-off oncesi yapilmis, sonradan oynanmis tahminler ({len(record)} mac)")
    if len(record):
        report_line("model", measure(record, record["home_win_prob"].to_numpy(),
                                     record["margin"].to_numpy(), record["total"].to_numpy()))
        priced = record.dropna(subset=["p_home_close", "spread_close"])
        if len(priced):
            report_line("piyasa", measure(priced, priced["p_home_close"].to_numpy(),
                                          -priced["spread_close"].to_numpy(),
                                          priced["total_close"].to_numpy()))
        print(f"   {len(record)} mac hicbir seyi kanitlamaz; bu sayi sezon ilerledikce birikir.")

    joined = model_rows.merge(lines, on="game_id").dropna(subset=["p_home_close", "spread_close"])
    print(f"\n== {season}, orani olan butun oynanmis maclar ({len(joined)} mac)")
    print("   DIKKAT: bu satirlarda kadro, tahmin edilen macin kutu skorundan okundu. Model kimin")
    print("   oynadigini biliyor, piyasa yalnizca kimin oynamasi beklendigini biliyordu. Karsilastirma")
    print("   modelin lehine egilidir, piyasaya ustunluk kaniti degildir.")
    if len(joined):
        report_line("model", measure(joined, joined["home_win_prob"].to_numpy(),
                                     joined["margin"].to_numpy(), joined["total"].to_numpy()))
        report_line("piyasa", measure(joined, joined["p_home_close"].to_numpy(),
                                      -joined["spread_close"].to_numpy(), joined["total_close"].to_numpy()))
    unpriced = len(model_rows) - len(joined)
    if unpriced:
        print(f"   orani olmayan {unpriced} mac bu satirlarin disinda "
              f"(py data_retriaval/odds_run_all.py ile girer)")


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    upcoming_section(stamp)
    season_sections()
    print("\nOranlar bir olcut ve girdi; burada bahis tavsiyesi yok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
