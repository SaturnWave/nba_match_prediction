"""
Step 3: one table of betting lines per NBA game id, from both sources.

Each source is matched to the project's regular-season games by date and the
two teams, then every line is turned to the home team's side:
  spread_*  home team's line (negative: home favoured)
  total_*   points total
  p_home_*  home win probability implied by the moneylines, margin removed
Opening and closing values are kept apart, because only an opening line is
known well before a game. Sportsbook Reviews covers 2017-18 to 2023-01-16 and
gives one line; ESPN covers 2023-01-01 on and gives the median of the books it
carries. On the two weeks both cover, Sportsbook Reviews is kept and the two are
compared. Playoff games are not in the project's game list and drop out.

Checks printed: the share of archive games whose final score matches the
project's (a wrong match or a home/away flip shows up here), coverage per season,
and how far apart the sources are where they overlap.

Writes odds_data/odds_history.csv. The lines are an outside yardstick and
model input only; nothing here is betting advice.

Run:  py data_retriaval/odds_build.py
"""
import os

import numpy as np
import pandas as pd

from odds_common import ODDS_DIR, PROJECT_ROOT

SBR_PATH = os.path.join(ODDS_DIR, "sbr_lines.csv")
ESPN_PATH = os.path.join(ODDS_DIR, "espn_lines.csv")
OUT_PATH = os.path.join(ODDS_DIR, "odds_history.csv")
COLUMNS = ["game_id", "season", "game_date", "home", "away", "source", "books", "spread_open", "spread_close",
           "total_open", "total_close", "p_home_open", "p_home_close"]


def project_games():
    """Every regular-season game the project knows, with its score where played."""
    g = pd.read_pickle(os.path.join(PROJECT_ROOT, "phonedb_cache", "game_summary.pkl"))
    games = pd.DataFrame({"game_id": g["game_id"].astype(str).str.zfill(10), "season": g["season"].astype(str),
                          "game_date": pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m-%d"),
                          "home": g["home_abbr"], "away": g["away_abbr"],
                          "home_pts": g["home_pts"], "away_pts": g["away_pts"]})
    schedule_path = os.path.join(PROJECT_ROOT, "game_ids", "schedule_2026_2027.csv")
    if os.path.exists(schedule_path):
        s = pd.read_csv(schedule_path, dtype={"game_id": str})
        s = s[(s["season_type"] == "regular_season") & (s["status"] == "final")]
        games = pd.concat([games, pd.DataFrame({
            "game_id": s["game_id"], "season": "2026_2027", "game_date": s["game_date"], "home": s["home"],
            "away": s["away"], "home_pts": s["home_score"], "away_pts": s["away_score"]})], ignore_index=True)
    return games.drop_duplicates("game_id")


def no_vig_home(home_american, away_american):
    if pd.isna(home_american) or pd.isna(away_american) or not home_american or not away_american:
        return np.nan
    implied = [1 / (1 + (a / 100 if a > 0 else 100 / -a)) for a in (home_american, away_american)]
    return implied[0] / sum(implied)


def match_by_date_and_teams(lines, games, home_col, away_col, unordered):
    """Attach game ids: same date and teams, else one day off when that is unique.

    A game can then be claimed twice - ESPN keeps a postponed game and its
    make-up as two events - so an exact-date match wins over a shifted one.
    """
    def key(frame, a, b):
        return (frame[[a, b]].apply(lambda r: "|".join(sorted(r)), axis=1) if unordered
                else frame[a] + "|" + frame[b])
    games = games.assign(pair=key(games, "home", "away"))
    lines = lines.assign(pair=key(lines, home_col, away_col))
    exact = lines.merge(games, left_on=["date", "pair"], right_on=["game_date", "pair"], how="left",
                        suffixes=("", "_g"))
    missing = exact["game_id"].isna()
    shifted = []
    for idx in exact.index[missing]:
        day = pd.Timestamp(exact.at[idx, "date"])
        near = games[(games["pair"] == exact.at[idx, "pair"])
                     & games["game_date"].isin([(day + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 1)])]
        if len(near) == 1:
            shifted.append(idx)
            for col in games.columns:
                if col != "pair":
                    exact.at[idx, col] = near.iloc[0][col]
    exact["shifted"] = exact.index.isin(shifted)
    claimed = exact[exact["game_id"].notna()].sort_values("shifted", kind="stable")
    duplicates = claimed.index[claimed["game_id"].duplicated()]
    return exact.drop(index=duplicates), len(shifted)


def from_sbr(games):
    sbr = pd.read_csv(SBR_PATH).drop(columns="season")       # the project's own season label comes with the match
    matched, shifted = match_by_date_and_teams(sbr, games, "team_a", "team_b", unordered=True)
    matched = matched[matched["game_id"].notna()].copy()
    a_home = matched["team_a"] == matched["home"]
    flip = np.where(a_home, 1.0, -1.0)
    matched["spread_open"] = matched["spread_open_a"] * flip
    matched["spread_close"] = matched["spread_close_a"] * flip
    ml_home = np.where(a_home, matched["ml_a"], matched["ml_b"])
    ml_away = np.where(a_home, matched["ml_b"], matched["ml_a"])
    matched["p_home_close"] = [no_vig_home(h, a) for h, a in zip(ml_home, ml_away)]
    matched["p_home_open"] = np.nan                     # the archive keeps only closing moneylines
    score_home = np.where(a_home, matched["score_a"], matched["score_b"])
    score_away = np.where(a_home, matched["score_b"], matched["score_a"])
    scores_agree = (score_home == matched["home_pts"]) & (score_away == matched["away_pts"])
    print(f"  Sportsbook Reviews: {len(sbr)} satir, {len(matched)} normal sezon macina eslesti "
          f"({shifted} tanesi bir gun kaymayla); skoru tutan {scores_agree.mean():.1%}")
    return matched.assign(source="sportsbookreviews", books=1)


def from_espn(games):
    espn = pd.read_csv(ESPN_PATH, dtype={"event_id": str})
    matched, shifted = match_by_date_and_teams(espn, games, "home", "away", unordered=False)
    unmatched = int(matched["game_id"].isna().sum())
    matched = matched[matched["game_id"].notna()].copy()
    played = matched["home_score"].notna() & matched["home_pts"].notna()
    scores_agree = ((matched.loc[played, "home_score"].astype(float) == matched.loc[played, "home_pts"].astype(float))
                    & (matched.loc[played, "away_score"].astype(float) == matched.loc[played, "away_pts"].astype(float)))
    print(f"  ESPN: {len(espn)} mac, {len(matched)} eslesti ({shifted} bir gun kaymayla, {unmatched} eslesmedi); "
          f"skoru tutan {scores_agree.mean():.1%}")
    return matched.assign(source="espn")


def compare_overlap(sbr, espn):
    both = sbr[["game_id", "spread_close", "total_close", "p_home_close"]].merge(
        espn[["game_id", "spread_close", "total_close", "p_home_close"]], on="game_id", suffixes=("_sbr", "_espn"))
    if both.empty:
        return
    print(f"  ortak {len(both)} mac (Ocak 2023): kapanis handikabi farki ortalama "
          f"{(both['spread_close_sbr'] - both['spread_close_espn']).abs().mean():.2f}, toplam farki "
          f"{(both['total_close_sbr'] - both['total_close_espn']).abs().mean():.2f}, ev olasiligi farki "
          f"{(both['p_home_close_sbr'] - both['p_home_close_espn']).abs().mean():.3f}")


def main():
    games = project_games()
    sbr, espn = from_sbr(games), from_espn(games)
    compare_overlap(sbr, espn)
    table = pd.concat([sbr[COLUMNS], espn[~espn["game_id"].isin(set(sbr["game_id"]))][COLUMNS]], ignore_index=True)
    table = table.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    table.to_csv(OUT_PATH, index=False)
    known = games.groupby("season").size().rename("games")
    coverage = table.groupby("season").agg(with_close=("spread_close", "count"), with_prob=("p_home_close", "count"),
                                           with_open=("spread_open", "count"))
    print(pd.concat([known, coverage], axis=1).fillna(0).astype(int).to_string())
    print(f"{len(table)} mac -> {OUT_PATH}")


if __name__ == "__main__":
    main()
