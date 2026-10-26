"""
How a pre-season simulation compares with the games played so far.

The forecast run (sim_sandbox/forecast_<season>/) holds one simulated season -
a single draw per game - and each team's win range. One draw says little about
a single game, so the games played so far are replayed many times from the same
setup: the run's schedule and early-October rosters, the same team priors, and
the same doubt about each team's strength that the win ranges carry
(season_sim.TEAM_SEASON_SD). That gives every game a home-win probability, an
expected margin and an expected total, which are scored here against the real
results next to two plain yardsticks: "the home team wins" and last season's
records (log5 plus home court).

Before replaying, the run's own seed is simulated again and its scores checked
against the saved ones - proof that the setup is the one the run was made from.
The run is never written to; replays happen in sim_sandbox/vs_real_<season>/,
which git ignores.

Run:  py prediction_engines/sim_vs_real.py                  (2026_2027, 1000 replays)
      py prediction_engines/sim_vs_real.py --replays 300
Writes output/sim_vs_real_<season>.json.
"""
import argparse
import dataclasses
import json
import math
import os
import shutil

import numpy as np
import pandas as pd

import season_sim as sim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_SEASON = "2026_2027"
DEFAULT_REPLAYS = 1000
BASE_SEED = 7_000_000
PROB_FLOOR = 1e-3            # keeps log loss finite when no replay produced the upset
SHOWN_PLAYERS = 12
TEAM_BOX = {"PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm",
            "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "AST": "ast", "STL": "stl", "BLK": "blk",
            "TO": "tov", "PF": "pf"}


# ---------------------------------------------------------------------------
#  The real side
# ---------------------------------------------------------------------------
def real_results(season):
    """Finished regular-season games, from the league's published schedule."""
    schedule = pd.read_csv(os.path.join(GAME_IDS_DIR, f"schedule_{season}.csv"), dtype={"game_id": str})
    done = schedule[(schedule["season_type"] == "regular_season") & (schedule["status"] == "final")]
    return (done[["game_id", "game_date", "home", "away", "home_score", "away_score"]]
            .astype({"home_score": int, "away_score": int}).reset_index(drop=True))


def box_path(season, game_id, kind):
    return os.path.join(DATA_DIR, season, game_id, "box_scores", f"{game_id}box_score_{kind}.csv")


def real_team_lines(season, game_ids):
    frames = [pd.read_csv(box_path(season, g, "traditional_team")).assign(game_id=g) for g in game_ids]
    team = pd.concat(frames, ignore_index=True).rename(columns=TEAM_BOX)
    return team.rename(columns={"TEAM_ABBREVIATION": "team"})[["game_id", "team", *TEAM_BOX.values()]]


def clock_minutes(text):
    """'33:15' -> 33.25; a missing value (did not play) -> 0."""
    if not isinstance(text, str) or ":" not in text:
        return 0.0
    minutes, seconds = text.split(":", 1)
    return int(minutes) + float(seconds) / 60.0


def real_player_lines(season, game_ids):
    frames = [pd.read_csv(box_path(season, g, "traditional"), dtype={"COMMENT": str}).assign(game_id=g)
              for g in game_ids]
    box = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"game_id": box["game_id"], "team": box["TEAM_ABBREVIATION"],
                         "person_id": box["PLAYER_ID"].astype("int64"), "name": box["PLAYER_NAME"],
                         "minutes": box["MIN"].map(clock_minutes), "pts": box["PTS"].fillna(0),
                         "status": box["COMMENT"].fillna("")})


# ---------------------------------------------------------------------------
#  The simulated side
# ---------------------------------------------------------------------------
def replay_setup(season, forecast_dir):
    """The run's setup, built in a sandbox of its own from the run's cached inputs."""
    sandbox = sim.Sandbox(f"vs_real_{season}")
    for name in ("schedule.csv", "rosters.csv"):          # cached inputs: nothing is fetched anew
        shutil.copyfile(os.path.join(forecast_dir, name), sandbox.path(name))
    return sandbox, sim.prepare_season("forecast", season, sandbox)


def reproduction_check(setup, sandbox, forecast_dir, seed):
    """How many of the run's saved game scores the same seed gives back."""
    saved = pd.read_csv(os.path.join(forecast_dir, "games.csv"), dtype={"game_id": str})
    again, _lines, _values, _extras = sim.simulate_season(setup, sandbox, seed, keep_events=False, verbose=False)
    both = saved.merge(again, on="game_id", suffixes=("_saved", "_again"))
    same = ((both["home_points_saved"] == both["home_points_again"])
            & (both["away_points_saved"] == both["away_points_again"]))
    return int(same.sum()), len(saved)


def replay(setup, sandbox, game_ids, replays):
    """Scores and box lines of the given games over `replays` simulated seasons."""
    wanted = setup.schedule[setup.schedule["game_id"].isin(set(game_ids))].reset_index(drop=True)
    played = dataclasses.replace(setup, schedule=wanted)
    scores, lines = [], []
    for r in range(replays):
        games, game_lines, _values, _extras = sim.simulate_season(
            played, sandbox, BASE_SEED + r, keep_events=False, team_shock_sd=sim.TEAM_SEASON_SD, verbose=False)
        scores.append(games[["game_id", "home_points", "away_points"]].assign(replay=r))
        lines.append(game_lines[["game_id", "team", "person_id", "name", "minutes", *sim.LINE_STATS]]
                     .assign(replay=r))
        if (r + 1) % 200 == 0:
            print(f"    {r + 1}/{replays} tekrar", flush=True)
    return pd.concat(scores, ignore_index=True), pd.concat(lines, ignore_index=True)


def game_forecasts(scores):
    margin = scores["home_points"] - scores["away_points"]
    by_game = scores.assign(margin=margin, total=scores["home_points"] + scores["away_points"],
                            home_win=margin > 0).groupby("game_id")
    return pd.DataFrame({"p_home": by_game["home_win"].mean(), "exp_margin": by_game["margin"].mean(),
                         "exp_total": by_game["total"].mean(), "margin_p10": by_game["margin"].quantile(0.1),
                         "margin_p90": by_game["margin"].quantile(0.9)}).reset_index()


# ---------------------------------------------------------------------------
#  Yardsticks and scores
# ---------------------------------------------------------------------------
def last_season(history_games, season):
    """Last season's win rates, its home-win share and its mean home margin and total."""
    previous = max(s for s in history_games["season"].unique() if s < season)
    g = history_games[history_games["season"] == previous]
    home_won = g["home_pts"] > g["away_pts"]
    winners = pd.concat([g.loc[home_won, "home_abbr"], g.loc[~home_won, "away_abbr"]]).value_counts()
    played = pd.concat([g["home_abbr"], g["away_abbr"]]).value_counts()
    return {"season": previous, "win_rate": (winners.reindex(played.index, fill_value=0) / played).to_dict(),
            "home_share": float(home_won.mean()), "home_margin": float((g["home_pts"] - g["away_pts"]).mean()),
            "total": float((g["home_pts"] + g["away_pts"]).mean())}


def logit(p):
    return math.log(p / (1 - p))


def log5_with_home(home_rate, away_rate, home_share):
    """Log5 on two win rates, moved by the league's home edge in log-odds."""
    home_rate, away_rate = (min(max(r, 0.05), 0.95) for r in (home_rate, away_rate))
    base = home_rate * (1 - away_rate) / (home_rate * (1 - away_rate) + away_rate * (1 - home_rate))
    return 1 / (1 + math.exp(-(logit(base) + logit(home_share))))


def probability_scores(p, home_won):
    p = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1 - PROB_FLOOR)
    y = np.asarray(home_won, dtype=float)
    return {"accuracy": float(((p > 0.5) == (y == 1)).mean()), "brier": float(((p - y) ** 2).mean()),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}


def league_line(team_lines):
    """Per team-game averages, with possessions estimated the way season_sim does."""
    means = team_lines[list(TEAM_BOX.values())].mean()
    possessions = team_lines["fga"] - team_lines["oreb"] + team_lines["tov"] + 0.44 * team_lines["fta"]
    return {**{k: float(v) for k, v in means.items()}, "possessions": float(possessions.mean()),
            "fg3_pct": float(team_lines["fg3m"].sum() / team_lines["fg3a"].sum()),
            "ft_pct": float(team_lines["ftm"].sum() / team_lines["fta"].sum())}


def player_comparison(real_lines, sim_lines, replays, rostered):
    """Real minutes and points per player-game against the replays' expectation."""
    expected = (sim_lines.groupby(["game_id", "team", "person_id"])[["minutes", "pts"]].sum() / replays).reset_index()
    names = pd.concat([sim_lines[["person_id", "name"]], real_lines[["person_id", "name"]]]).drop_duplicates("person_id")
    both = real_lines.merge(expected, on=["game_id", "team", "person_id"], how="outer", suffixes=("_real", "_sim"))
    both[["minutes_real", "pts_real", "minutes_sim", "pts_sim"]] = both[
        ["minutes_real", "pts_real", "minutes_sim", "pts_sim"]].fillna(0.0)
    both = both.drop(columns="name").merge(names, on="person_id", how="left")
    played = both[(both["minutes_real"] > 0) | (both["minutes_sim"] > 0)]
    per_player = (played.groupby(["person_id", "name", "team"])
                  .agg(games=("game_id", "nunique"), minutes_real=("minutes_real", "sum"),
                       minutes_sim=("minutes_sim", "sum"), pts_real=("pts_real", "sum"), pts_sim=("pts_sim", "sum"),
                       status=("status", lambda s: "; ".join(sorted({x for x in s.dropna() if x})))).reset_index())
    per_player["minutes_gap"] = per_player["minutes_real"] - per_player["minutes_sim"]
    off_roster = played[(played["minutes_real"] > 0)
                        & ~played.apply(lambda r: (r["team"], r["person_id"]) in rostered, axis=1)]
    return {
        "minutes_correlation": float(played["minutes_real"].corr(played["minutes_sim"])),
        "points_correlation": float(played["pts_real"].corr(played["pts_sim"])),
        "real_minutes_by_players_off_the_sim_roster": float(off_roster["minutes_real"].sum()),
        "real_minutes_total": float(played["minutes_real"].sum()),
        "players_off_the_sim_roster": sorted(set(off_roster["name"])),
        "biggest_gaps": per_player.reindex(per_player["minutes_gap"].abs().sort_values(ascending=False).index)
                                  .head(SHOWN_PLAYERS).round(1).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
#  Putting it together
# ---------------------------------------------------------------------------
def compare(season, replays):
    forecast_dir = os.path.join(sim.SANDBOX_ROOT, f"forecast_{season}")
    with open(os.path.join(forecast_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    real = real_results(season)
    if real.empty:
        raise SystemExit(f"{season}: oynanmis normal sezon maci yok")
    sandbox, setup = replay_setup(season, forecast_dir)
    in_run = real[real["game_id"].isin(set(setup.schedule["game_id"]))]
    print(f"  {len(real)} oynanmis mac, {len(in_run)} tanesi simulasyonun fiksturunde")

    reproduced, saved_games = reproduction_check(setup, sandbox, forecast_dir, run["seed"])
    print(f"  ayni tohumla yeniden: {reproduced}/{saved_games} macin skoru kayitliyla ayni")

    scores, sim_lines = replay(setup, sandbox, in_run["game_id"], replays)
    games = in_run.merge(game_forecasts(scores), on="game_id")
    saved = pd.read_csv(os.path.join(forecast_dir, "games.csv"), dtype={"game_id": str})
    games = games.merge(saved[["game_id", "home_points", "away_points"]]
                        .rename(columns={"home_points": "draw_home", "away_points": "draw_away"}), on="game_id")
    games["real_margin"] = games["home_score"] - games["away_score"]
    games["home_won"] = games["real_margin"] > 0

    previous = last_season(setup.history_games, season)
    games["p_last_season"] = [log5_with_home(previous["win_rate"].get(h, 0.5), previous["win_rate"].get(a, 0.5),
                                             previous["home_share"]) for h, a in zip(games["home"], games["away"])]
    real_total = games["home_score"] + games["away_score"]
    results = {
        "simulation": probability_scores(games["p_home"], games["home_won"]),
        "home_team_wins": probability_scores([previous["home_share"]] * len(games), games["home_won"]),
        "last_season_records": probability_scores(games["p_last_season"], games["home_won"]),
    }
    margins = {"simulation_margin_mae": float((games["exp_margin"] - games["real_margin"]).abs().mean()),
               "naive_margin_mae": float((previous["home_margin"] - games["real_margin"]).abs().mean()),
               "simulation_total_mae": float((games["exp_total"] - real_total).abs().mean()),
               "naive_total_mae": float((previous["total"] - real_total).abs().mean()),
               "real_margin_inside_10_90": float(((games["real_margin"] >= games["margin_p10"])
                                                  & (games["real_margin"] <= games["margin_p90"])).mean()),
               "saved_draw_same_winner": int(((games["draw_home"] > games["draw_away"]) == games["home_won"]).sum())}

    sim_team = sim_lines.groupby(["replay", "game_id", "team"])[list(TEAM_BOX.values())].sum().reset_index()
    league = {"real": league_line(real_team_lines(season, in_run["game_id"])), "simulation": league_line(sim_team),
              "real_home_win_share": float(games["home_won"].mean()),
              "simulation_home_win_share": float(games["p_home"].mean())}

    rostered = {(team, int(p.person_id)) for team, players in setup.team_rosters.items() for p in players}
    players = player_comparison(real_player_lines(season, in_run["game_id"]), sim_lines, replays, rostered)

    ranges = pd.read_csv(os.path.join(forecast_dir, "win_ranges.csv"))
    teams = []
    for team in sorted(set(games["home"]) | set(games["away"])):
        mine = games[(games["home"] == team) | (games["away"] == team)]
        at_home = mine["home"] == team
        won = np.where(at_home, mine["home_won"], ~mine["home_won"])
        expected = np.where(at_home, mine["p_home"], 1 - mine["p_home"])
        season_row = ranges[ranges["team"] == team]
        teams.append({"team": team, "games": int(len(mine)), "wins": int(won.sum()),
                      "expected_wins": round(float(expected.sum()), 2),
                      "season_mean_wins": float(season_row["mean_wins"].iloc[0]) if len(season_row) else None})

    report = {"season": season, "run_seed": run["seed"], "replays": replays, "games_compared": int(len(games)),
              "through": str(games["game_date"].max()), "reproduced_scores": [reproduced, saved_games],
              "last_season": previous["season"], "scores": results, "margins_and_totals": margins,
              "league_per_team_game": league, "players": players, "teams": teams,
              "games": games[["game_id", "game_date", "home", "away", "home_score", "away_score", "p_home",
                              "p_last_season", "exp_margin", "exp_total", "margin_p10", "margin_p90",
                              "draw_home", "draw_away"]].round(3).to_dict(orient="records")}
    path = os.path.join(OUTPUT_DIR, f"sim_vs_real_{season}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"  rapor: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    args = parser.parse_args()
    report = compare(args.season, args.replays)
    for name, s in report["scores"].items():
        print(f"  {name:<22} isabet {s['accuracy']:.3f}  Brier {s['brier']:.3f}  log loss {s['log_loss']:.3f}")
    m = report["margins_and_totals"]
    print(f"  fark MAE: simulasyon {m['simulation_margin_mae']:.1f} / naif {m['naive_margin_mae']:.1f}; "
          f"toplam MAE: simulasyon {m['simulation_total_mae']:.1f} / naif {m['naive_total_mae']:.1f}")


if __name__ == "__main__":
    main()
