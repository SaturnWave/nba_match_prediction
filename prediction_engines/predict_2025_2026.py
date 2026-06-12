"""
NBA game outcome predictor — 2025-26 edition.

A corrected, runnable rebuild of prediction_engines/2023_2024.py:

  * project paths are derived from this file's location (no hard-coded desktop path);
  * a REAL play-by-play impact score (ported from asasa.py) replaces the constant
    stub, GENERALISED so the lead-change / frontcourt / trailing-team modifiers work
    for any matchup (the original hard-coded Denver=1610612743 / GSW=1610612744);
  * LightGBM 4.x training API (early-stopping via callbacks);
  * trains on 2024-25, validates/tests on 2025-26 (the freshly retrieved season);
  * caches per-game impact aggregates so re-runs are fast;
  * writes models/, metrics JSON, and a single-game test report.

Run:  py prediction_engines/predict_2025_2026.py
"""
import os
import json
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error)
from tqdm import tqdm
from nba_api.stats.static import teams as static_teams

try:
    from matchup_impact import add_matchup_features
except ImportError:                                   # imported as a package
    try:
        from prediction_engines.matchup_impact import add_matchup_features
    except ImportError:
        add_matchup_features = None

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
GAME_IDS_DIR = os.path.join(PROJECT_ROOT, "game_ids")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v3.pkl")

TARGETS = ["home_win", "point_diff", "total_score", "home_score", "away_score"]


# ===========================================================================
#  Impact score  (play-by-play only, generalised for any matchup)
# ===========================================================================
def clock_to_seconds(x):
    if isinstance(x, str) and "PT" in x and "M" in x and "S" in x:
        try:
            return int(x.split("PT")[1].split("M")[0]) * 60 + float(x.split("M")[1].replace("S", ""))
        except Exception:
            return 0.0
    return 0.0


def is_clutch(clock_seconds, period):
    return (period == 4 and clock_seconds <= 300) or period > 4


def is_last_2min(clock_seconds, period):
    return (period == 4 and clock_seconds <= 120) or period > 4


def margin(row):
    h, a = row.get("scoreHome"), row.get("scoreAway")
    if pd.notnull(h) and pd.notnull(a):
        return abs(float(h) - float(a))
    return 0.0


def expected_points(x, y, shot_value):
    if pd.isnull(x) or pd.isnull(y) or pd.isnull(shot_value):
        return None
    dist = np.sqrt(x ** 2 + y ** 2) / 10.0
    if shot_value == 3:
        return 1.1 if (abs(x) > 220 and y < 90) else 0.9
    if dist < 5:
        return 1.6
    if dist < 10:
        return 0.9
    if dist < 16:
        return 0.8
    return 0.7


def _scoring_run_team(previous_plays):
    """Team (tricode) with the most plays in the recent window, or None."""
    counts = defaultdict(int)
    for p in previous_plays[-5:]:
        t = p.get("teamTricode")
        if isinstance(t, str):
            counts[t] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _block_impact(row, nxt, prev):
    bi = 1.2
    if nxt is not None and nxt.get("teamTricode") == row.get("teamTricode"):
        bi -= 0.2
    if nxt is not None and isinstance(nxt.get("description"), str):
        if "Running" in nxt["description"]:
            bi += 0.2
        if "Shot Clock" in nxt["description"]:
            bi += 0.3
    recent = [p for p in prev[-3:] if isinstance(p.get("description"), str)
              and "BLOCK" in p["description"] and p.get("playerName") == row.get("playerName")]
    if len(recent) > 1:
        bi += 0.3
    if is_last_2min(row["clock_seconds"], row["period"]) and margin(row) <= 3:
        bi += 0.5
    if pd.notnull(row.get("shotDistance")) and row["shotDistance"] <= 5:
        bi += 0.2
    srt = _scoring_run_team(prev)
    if srt and srt != row.get("teamTricode"):
        bi += 0.3
    if nxt is not None and nxt.get("teamTricode") != row.get("teamTricode"):
        bi += 0.2
    return bi


def _steal_impact(row, nxt, prev, home_id, away_id):
    bi = 1.4
    desc = row.get("description")
    if isinstance(desc, str) and "Backcourt" in desc:
        bi += 0.1
    if nxt is not None and nxt.get("actionType") == "Made Shot":
        bi += 0.2
    recent = [p for p in prev[-5:] if isinstance(p.get("description"), str)
              and "STEAL" in p["description"] and p.get("playerName") == row.get("playerName")]
    if len(recent) > 1:
        bi += 0.2
    if is_clutch(row["clock_seconds"], row["period"]):
        m = margin(row)
        bi = 1.0 if m > 20 else 1.1 if m > 10 else 1.5
    if isinstance(desc, str):
        if "Bad Pass" in desc:
            bi += 0.1
        elif "Lost Ball" in desc:
            bi += 0.3
    if (nxt is not None and nxt.get("actionType") == "Made Shot"
            and pd.notnull(nxt.get("shotDistance")) and nxt["shotDistance"] <= 3):
        bi += 0.3
    # frontcourt steal — generalised to actual home/away team ids
    if pd.notnull(row.get("xLegacy")):
        tid = row.get("teamId")
        if (tid == home_id and row["xLegacy"] < 0) or (tid == away_id and row["xLegacy"] > 0):
            bi += 0.2
    return bi


def _rebound_impact(row, nxt, prev, home_id, away_id):
    desc = row.get("description")
    off = isinstance(desc, str) and "Off" in desc
    bi = 0.9 if off else 0.6
    if any(isinstance(p.get("description"), str) and "REBOUND" in p["description"] for p in prev[-2:]):
        bi += 0.2
    if nxt is not None and nxt.get("actionType") == "Made Shot":
        bi += 0.2
    if is_last_2min(row["clock_seconds"], row["period"]) and margin(row) <= 3:
        bi += 0.3
    if is_clutch(row["clock_seconds"], row["period"]):
        bi += 0.4
    if any(isinstance(p.get("description"), str) and "BLOCK" in p["description"] for p in prev[-1:]):
        bi += 0.3
    if off:
        tid = row.get("teamId")
        if ((tid == home_id and row.get("scoreHome", 0) < row.get("scoreAway", 0))
                or (tid == away_id and row.get("scoreHome", 0) > row.get("scoreAway", 0))):
            bi += 0.2
    return bi


def _scoring_impact(row, prev):
    bi = 3.0 if row.get("shotValue") == 3 else 2.0
    if any(isinstance(p.get("description"), str) and "Free Throw" in p["description"] for p in prev[:2]):
        bi += 0.3
    if any(isinstance(p.get("description"), str) and "Timeout" in p["description"] for p in prev[-3:]):
        bi += 0.2
    srt = _scoring_run_team(prev)
    if srt and srt != row.get("teamTricode"):
        bi += 0.2
    ep = expected_points(row.get("xLegacy"), row.get("yLegacy"), row.get("shotValue"))
    if ep is not None:
        bi *= ep
    desc = row.get("description")
    if isinstance(desc, str):
        for key, add in (("Fadeaway", 0.2), ("Step Back", 0.3), ("Alley Oop", 0.4),
                         ("Turnaround", 0.2), ("Pullup", 0.1), ("Bank", 0.1)):
            if key in desc:
                bi += add
                break
        if "Driving" in desc and "Dunk" in desc:
            bi += 0.3
    cs = row.get("clock_seconds")
    if pd.notnull(cs):
        sc = cs % 24
        if sc <= 4:
            bi += 0.3
        elif sc <= 7:
            bi += 0.1
    if is_clutch(row["clock_seconds"], row["period"]):
        m = margin(row)
        if m <= 5:
            bi *= 1.3
        elif m <= 10:
            bi *= 1.2
    return bi


def _turnover_impact(row, nxt, prev):
    bi = -1.0 if is_clutch(row["clock_seconds"], row["period"]) else -0.8
    desc = row.get("description")
    if isinstance(desc, str):
        if "Bad Pass" in desc:
            bi -= 0.2
        elif "Lost Ball" in desc:
            bi -= 0.3
        elif "Step Out of Bounds" in desc or "Traveling" in desc:
            bi -= 0.1
        elif "Shot Clock" in desc:
            bi -= 0.3
        elif "Offensive Foul" in desc:
            bi -= 0.2
        elif "Backcourt" in desc:
            bi -= 0.3
    if (nxt is not None and nxt.get("actionType") == "Made Shot"
            and nxt.get("teamTricode") != row.get("teamTricode")):
        bi -= 0.3
        if nxt.get("shotValue") == 3:
            bi -= 0.2
    m = margin(row)
    if m <= 5 and row["period"] >= 4:
        bi *= 1.3
    elif m >= 15:
        bi *= 0.7
    return bi


def _foul_impact(row, nxt, prev):
    desc = row.get("description")
    if isinstance(desc, str):
        if "S.FOUL" in desc:
            bi = -0.7
        elif "P.FOUL" in desc:
            bi = -0.3
        elif "OFF.FOUL" in desc or "Offensive" in desc:
            bi = -0.6
        elif "L.B.FOUL" in desc:
            bi = -0.4
        elif "T.FOUL" in desc:
            bi = -1.0
        elif "FLAGRANT" in desc.upper():
            bi = -1.5
        else:
            bi = -0.5
    else:
        bi = -0.5
    if nxt is not None and isinstance(nxt.get("description"), str) and "Free Throw" in nxt["description"]:
        if not (isinstance(desc, str) and "S.FOUL" in desc):
            bi -= 0.2
    if is_last_2min(row["clock_seconds"], row["period"]) and margin(row) <= 3:
        bi *= 1.2
    return bi


def compute_game_impact(pbp_df, home_tricode, away_tricode, home_id, away_id, team_possessions):
    """Aggregate per-team play-by-play impact for a single game.

    Returns dict: home_impact, away_impact (raw sums) and the per-player series.
    Iterates once over a list of dict records for speed.
    """
    if pbp_df is None or pbp_df.empty:
        return {"home_impact": 0.0, "away_impact": 0.0, "players": {}}

    df = pbp_df.copy()
    if "actionNumber" in df.columns:
        df = df.sort_values("actionNumber")
    df["clock_seconds"] = df["clock"].apply(clock_to_seconds)
    df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce").fillna(0.0)
    df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce").fillna(0.0)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)
    records = df.to_dict("records")

    player_impact = defaultdict(float)
    player_team = {}
    n = len(records)
    for i, row in enumerate(records):
        player = row.get("playerName")
        if pd.isna(player) or player is None or player == "":
            continue
        if isinstance(row.get("teamTricode"), str):
            player_team[player] = row["teamTricode"]
        nxt = records[i + 1] if i < n - 1 else None
        prev = records[max(0, i - 5):i]
        desc = row.get("description")
        at = row.get("actionType")
        impact = 0.0
        if isinstance(desc, str) and "BLOCK" in desc:
            impact = _block_impact(row, nxt, prev)
        elif isinstance(desc, str) and "STEAL" in desc:
            impact = _steal_impact(row, nxt, prev, home_id, away_id)
        elif at == "Rebound":
            impact = _rebound_impact(row, nxt, prev, home_id, away_id)
        elif at == "Made Shot":
            impact = _scoring_impact(row, prev)
        elif isinstance(desc, str) and "Foul" in desc:
            impact = _foul_impact(row, nxt, prev)
        elif at == "Turnover":
            impact = _turnover_impact(row, nxt, prev)
        else:
            continue
        if is_clutch(row.get("clock_seconds", 0), row.get("period", 0)):
            impact *= 1.5
        cs = row.get("clock_seconds")
        if pd.notnull(cs):
            impact *= 1 + (1.0 / (cs + 1))
        player_impact[player] += impact

    home_imp = away_imp = 0.0
    for player, val in player_impact.items():
        team = player_team.get(player)
        if team == home_tricode:
            home_imp += val
        elif team == away_tricode:
            away_imp += val
    # per-player detail keeps the team so a forward-looking roster impact can be built
    players_detail = {p: {"impact": v, "team": player_team.get(p)} for p, v in player_impact.items()}
    return {"home_impact": home_imp, "away_impact": away_imp, "players": players_detail}


# ===========================================================================
#  Data loading
# ===========================================================================
class GameDataLoader:
    def __init__(self, data_dir=BASE_DATA_DIR, game_ids_dir=GAME_IDS_DIR):
        self.data_dir = data_dir
        self.game_ids_dir = game_ids_dir

    def get_game_ids_for_season(self, season_str):
        fp = os.path.join(self.game_ids_dir, f"game_id_{season_str}.csv")
        if not os.path.exists(fp):
            print(f"  [warn] no game-id file for {season_str}")
            return pd.DataFrame()
        return pd.read_csv(fp)

    def load_game_data(self, game_id, season_str):
        gid = str(game_id).zfill(10)
        gdir = os.path.join(self.data_dir, season_str, gid)
        if not os.path.exists(gdir):
            return None
        pbp_path = os.path.join(gdir, "play_by_play", f"{gid}pbp.csv")
        if not (os.path.exists(pbp_path) and os.path.getsize(pbp_path) > 64):
            return None
        data = {"game_id": gid, "season": season_str}
        try:
            data["pbp"] = pd.read_csv(pbp_path)
        except Exception:
            return None
        box_dir = os.path.join(gdir, "box_scores")
        for bs in ["traditional", "advanced", "defensive",
                   "traditional_team", "advanced_team", "defensive_team"]:
            p = os.path.join(box_dir, f"{gid}box_score_{bs}.csv")
            if os.path.exists(p) and os.path.getsize(p) > 64:
                try:
                    data[f"box_{bs}"] = pd.read_csv(p)
                except Exception:
                    data[f"box_{bs}"] = None
        return data


# ===========================================================================
#  Feature engineering (rolling form, season averages, streaks, H2H)
# ===========================================================================
class FeatureEngineer:
    STATS = ["score", "score_allowed", "point_margin", "FGM", "FGA", "FG_PCT",
             "FG3M", "FG3A", "FG3_PCT", "REB", "AST", "TO",
             "impact_score_agg"]
    WINDOWS = [3, 5, 10]

    def _rolling(self, hist):
        out = pd.DataFrame(index=hist.index)
        for stat in self.STATS:
            if stat in hist.columns:
                for w in self.WINDOWS:
                    out[f"L{w}_{stat}"] = hist[stat].rolling(w, min_periods=1).mean().shift(1)
        if "won" in hist.columns:
            for w in self.WINDOWS:
                out[f"L{w}_win_pct"] = hist["won"].rolling(w, min_periods=1).mean().shift(1)
        return out

    def _streaks(self, hist):
        if "won" not in hist.columns or hist.empty:
            return pd.Series(0, index=hist.index, name="streak")
        streaks, cur = [], 0
        for w in hist["won"].shift(1).fillna(0.5):
            if w == 1:
                cur = max(1, cur + 1)
            elif w == 0:
                cur = min(-1, cur - 1)
            else:
                cur = 0
            streaks.append(cur)
        return pd.Series(streaks, index=hist.index, name="streak")

    def engineer(self, all_games_df):
        if all_games_df.empty:
            return pd.DataFrame()
        df = all_games_df.sort_values("game_date").copy()
        team_feature_frames = []
        team_ids = pd.concat([df["home_team_id"], df["away_team_id"]]).dropna().astype(int).unique()

        for team_id in tqdm(team_ids, desc="  team features", leave=False):
            frames = []
            for is_home in (True, False):
                side = df[df["home_team_id"] == team_id] if is_home else df[df["away_team_id"] == team_id]
                if side.empty:
                    continue
                t = side.copy()
                if is_home:
                    t["won"] = t["home_win"]
                    t["score"] = t["home_score"]
                    t["score_allowed"] = t["away_score"]
                    pref = "home_"
                else:
                    t["won"] = 1 - t["home_win"]
                    t["score"] = t["away_score"]
                    t["score_allowed"] = t["home_score"]
                    pref = "away_"
                t["point_margin"] = t["score"] - t["score_allowed"]
                for stat in self.STATS:
                    if stat in ("score", "score_allowed", "point_margin"):
                        continue
                    src = f"{pref}{stat}"
                    if src in t.columns:
                        t[stat] = t[src]
                keep = ["game_id", "game_date", "season", "won", "score", "score_allowed", "point_margin"]
                keep += [s for s in self.STATS if s in t.columns and s not in keep]
                frames.append(t[[c for c in keep if c in t.columns]])
            if not frames:
                continue
            tg = pd.concat(frames).sort_values("game_date").drop_duplicates("game_id")
            roll = self._rolling(tg)
            streak = self._streaks(tg)
            seas = pd.DataFrame(index=tg.index)
            for stat in [s for s in self.STATS if s in tg.columns]:
                seas[f"season_avg_{stat}"] = (tg.groupby("season")[stat].expanding().mean()
                                              .reset_index(level=0, drop=True).shift(1))
            if "won" in tg.columns:
                seas["season_win_pct"] = (tg.groupby("season")["won"].expanding().mean()
                                          .reset_index(level=0, drop=True).shift(1))
            feats = pd.concat([roll, streak, seas], axis=1)
            feats["game_id"] = tg["game_id"]
            feats["team_id_for_features"] = team_id
            team_feature_frames.append(feats)

        if not team_feature_frames:
            return df
        allfeat = pd.concat(team_feature_frames).reset_index(drop=True)
        df = pd.merge(df, allfeat.add_prefix("home_"), left_on=["game_id", "home_team_id"],
                      right_on=["home_game_id", "home_team_id_for_features"], how="left")
        df = pd.merge(df, allfeat.add_prefix("away_"), left_on=["game_id", "away_team_id"],
                      right_on=["away_game_id", "away_team_id_for_features"], how="left")
        df = df.drop(columns=[c for c in df.columns
                              if c.endswith(("_game_id", "_team_id_for_features")) and c != "game_id"],
                     errors="ignore")

        # head-to-head (last 5 meetings before this game)
        df["h2h_L5_home_wins"] = 0.0
        df["h2h_L5_avg_pt_diff"] = 0.0
        for idx, g in df.iterrows():
            h, a, d = g.get("home_team_id"), g.get("away_team_id"), g.get("game_date")
            if pd.isna(h) or pd.isna(a) or pd.isna(d):
                continue
            past = df[(((df["home_team_id"] == h) & (df["away_team_id"] == a)) |
                       ((df["home_team_id"] == a) & (df["away_team_id"] == h))) &
                      (df["game_date"] < d)].sort_values("game_date").tail(5)
            if past.empty:
                continue
            wins, diff = 0, 0.0
            for _, pg in past.iterrows():
                hs, as_ = pg.get("home_score", 0), pg.get("away_score", 0)
                if pg.get("home_team_id") == h:
                    wins += int(pg.get("home_win", 0))
                    diff += hs - as_
                else:
                    wins += int(not pg.get("home_win", 0))
                    diff += as_ - hs
            df.loc[idx, "h2h_L5_home_wins"] = wins
            df.loc[idx, "h2h_L5_avg_pt_diff"] = diff / len(past)

        # diff features (home - away) for every rolling / season stat
        for hc in [c for c in df.columns if c.startswith("home_L") or c.startswith("home_season_avg_")]:
            ac = hc.replace("home_", "away_", 1)
            if ac in df.columns:
                df[hc.replace("home_", "diff_", 1)] = df[hc] - df[ac]
        if "home_streak" in df.columns and "away_streak" in df.columns:
            df["diff_streak"] = df["home_streak"] - df["away_streak"]
        return df


# ===========================================================================
#  Predictor
# ===========================================================================
class NBAPredictor:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.loader = GameDataLoader()
        self.fe = FeatureEngineer()
        self.master_df = None
        self.dataset = None
        self.models = {}
        self.feature_columns = []
        self.metrics = {}
        # reliable static team map (30 NBA teams) — independent of box-score availability
        self.abbr_to_id = {t["abbreviation"]: int(t["id"]) for t in static_teams.get_teams()}
        self.id_to_abbr = {v: k for k, v in self.abbr_to_id.items()}
        self._impact_cache = self._load_cache()
        self._player_records = []     # (player, team, game_id, game_date, season, impact)
        self.player_games = None      # per-player impact history with trailing form columns

    @staticmethod
    def _load_cache():
        if os.path.exists(IMPACT_CACHE):
            try:
                with open(IMPACT_CACHE, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        with open(IMPACT_CACHE, "wb") as f:
            pickle.dump(self._impact_cache, f)

    @staticmethod
    def _parse_matchup(matchup):
        # NBA convention: "HOME vs. AWAY"  and  "AWAY @ HOME"
        if " vs. " in matchup:
            h, a = matchup.split(" vs. ")
            return h.strip(), a.strip()
        if " @ " in matchup:
            a, h = matchup.split(" @ ")
            return h.strip(), a.strip()
        parts = matchup.split()
        return parts[-1], parts[0]

    def _register_teams(self, box_team_df):
        if box_team_df is None or box_team_df.empty:
            return
        if not {"TEAM_ID", "TEAM_ABBREVIATION"}.issubset(box_team_df.columns):
            return
        for _, r in box_team_df[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().iterrows():
            try:
                tid = int(r["TEAM_ID"])
                self.abbr_to_id[str(r["TEAM_ABBREVIATION"])] = tid
                self.id_to_abbr[tid] = str(r["TEAM_ABBREVIATION"])
            except Exception:
                pass

    def _extract_features(self, meta, bundle):
        pbp = bundle["pbp"]
        home_abbr, away_abbr = self._parse_matchup(str(meta["MATCHUP"]))
        self._register_teams(bundle.get("box_traditional_team"))
        self._register_teams(bundle.get("box_advanced_team"))
        home_id = self.abbr_to_id.get(home_abbr)
        away_id = self.abbr_to_id.get(away_abbr)

        last = pbp.iloc[-1]
        hs = pd.to_numeric(last.get("scoreHome"), errors="coerce")
        as_ = pd.to_numeric(last.get("scoreAway"), errors="coerce")
        hs = 0 if pd.isna(hs) else float(hs)
        as_ = 0 if pd.isna(as_) else float(as_)

        feats = {
            "game_id": str(meta["GAME_ID"]).zfill(10),
            "game_date": pd.to_datetime(meta["GAME_DATE"]),
            "season": bundle["season"],
            "home_team": home_abbr, "away_team": away_abbr,
            "home_team_id": home_id, "away_team_id": away_id,
            "home_score": hs, "away_score": as_,
            "point_diff": hs - as_, "total_score": hs + as_,
            "home_win": 1 if hs > as_ else 0,
        }

        trad = bundle.get("box_traditional")
        adv_team = bundle.get("box_advanced_team")
        for pref, abbr in (("home", home_abbr), ("away", away_abbr)):
            if trad is not None and "TEAM_ABBREVIATION" in trad.columns:
                ts = trad[trad["TEAM_ABBREVIATION"] == abbr]
                for stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                             "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF"]:
                    feats[f"{pref}_{stat}"] = ts[stat].sum() if (not ts.empty and stat in ts) else 0
                feats[f"{pref}_FG_PCT"] = (feats[f"{pref}_FGM"] / feats[f"{pref}_FGA"]
                                           if feats.get(f"{pref}_FGA", 0) else 0)
                feats[f"{pref}_FG3_PCT"] = (feats[f"{pref}_FG3M"] / feats[f"{pref}_FG3A"]
                                            if feats.get(f"{pref}_FG3A", 0) else 0)
            if adv_team is not None and "TEAM_ABBREVIATION" in adv_team.columns:
                a = adv_team[adv_team["TEAM_ABBREVIATION"] == abbr]
                for stat in ["PACE", "OFF_RATING", "DEF_RATING", "TS_PCT"]:
                    feats[f"{pref}_{stat}"] = a[stat].iloc[0] if (not a.empty and stat in a.columns) else np.nan

        # impact aggregates (cached) + per-player records for roster impact
        gid = feats["game_id"]
        if gid in self._impact_cache:
            imp = self._impact_cache[gid]
        else:
            imp = compute_game_impact(pbp, home_abbr, away_abbr, home_id, away_id, {})
            self._impact_cache[gid] = imp
        feats["home_impact_score_agg"] = imp["home_impact"]
        feats["away_impact_score_agg"] = imp["away_impact"]
        feats["impact_score_diff"] = imp["home_impact"] - imp["away_impact"]
        for player, d in imp["players"].items():
            team = d.get("team") if isinstance(d, dict) else None
            impact_val = d.get("impact", 0.0) if isinstance(d, dict) else d
            self._player_records.append({
                "player": player, "team": team, "game_id": gid,
                "game_date": feats["game_date"], "season": feats["season"],
                "impact": impact_val})
        return feats

    # ------------------------------------------------------------------
    #  Forward-looking, roster-aware impact  (get_current_roster_impact)
    # ------------------------------------------------------------------
    def _build_player_history(self):
        """Build per-player impact history with trailing-form columns.

        For every (player, game) we record the mean of the player's impact over
        their last 10 / 6 / 3 games *before* that game (shift(1) excludes the game
        itself). Because the history is sorted by date across ALL loaded seasons,
        a player's window during the first ~10 games of a season automatically
        rolls back into the previous season — implementing the season-warmup rule
        (early-season games lean on last season; from ~game 11 they use this
        season's last 10). l3 / l6 capture short-term form.
        """
        pg = pd.DataFrame(self._player_records)
        if pg.empty:
            self.player_games = pg
            return pg
        pg = pg.dropna(subset=["player", "team"]).sort_values(["player", "game_date"])
        for w, name in ((10, "l10"), (6, "l6"), (3, "l3")):
            pg[f"p_{name}"] = (pg.groupby("player")["impact"]
                               .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1)))
        self.player_games = pg
        return pg

    def get_current_roster_impact(self, roster_players, as_of_date):
        """Forward-looking impact for a set of players as of a date (user spec).

        Per player: mean impact over their last 10 games strictly before
        `as_of_date` (rolls into the previous season for early-season dates), plus
        last-6 and last-3 form means. Aggregated across the roster. Usable for true
        future predictions once self.player_games is built.
        """
        if self.player_games is None or self.player_games.empty:
            return {"roster_impact_l10_mean": 0.0, "roster_impact_l10_sum": 0.0,
                    "roster_form_l6": 0.0, "roster_form_l3": 0.0}
        as_of = pd.to_datetime(as_of_date)
        pg = self.player_games
        l10s, l6s, l3s = [], [], []
        for p in roster_players:
            hist = pg[(pg["player"] == p) & (pg["game_date"] < as_of)]
            if hist.empty:
                continue
            imp = hist.sort_values("game_date")["impact"]
            l10s.append(imp.tail(10).mean())
            l6s.append(imp.tail(6).mean())
            l3s.append(imp.tail(3).mean())
        if not l10s:
            return {"roster_impact_l10_mean": 0.0, "roster_impact_l10_sum": 0.0,
                    "roster_form_l6": 0.0, "roster_form_l3": 0.0}
        return {"roster_impact_l10_mean": float(np.mean(l10s)),
                "roster_impact_l10_sum": float(np.sum(l10s)),
                "roster_form_l6": float(np.nanmean(l6s)),
                "roster_form_l3": float(np.nanmean(l3s))}

    def _add_roster_features(self, df):
        """Vectorised version of get_current_roster_impact over the whole dataset."""
        pg = self._build_player_history()
        if pg is None or pg.empty:
            return df
        metrics = ("roster_impact_l10_mean", "roster_impact_l10_sum",
                   "roster_form_l6", "roster_form_l3")
        agg = (pg.groupby(["game_id", "team"])
               .agg(roster_impact_l10_mean=("p_l10", "mean"),
                    roster_impact_l10_sum=("p_l10", "sum"),
                    roster_form_l6=("p_l6", "mean"),
                    roster_form_l3=("p_l3", "mean"))
               .reset_index())
        for side in ("home", "away"):
            side_agg = agg.rename(columns={m: f"{side}_{m}" for m in metrics})
            df = df.merge(side_agg, left_on=["game_id", f"{side}_team"],
                          right_on=["game_id", "team"], how="left")
            df = df.drop(columns=["team"], errors="ignore")
        for feat in metrics:
            h, a = f"home_{feat}", f"away_{feat}"
            if h in df.columns and a in df.columns:
                df[f"diff_{feat}"] = df[h].fillna(0) - df[a].fillna(0)
        return df

    def load_and_prepare(self, seasons):
        print(f"Loading seasons: {seasons}")
        self._player_records = []
        rows = []
        for season in seasons:
            ids = self.loader.get_game_ids_for_season(season)
            if ids.empty:
                continue
            for _, meta in tqdm(ids.iterrows(), total=len(ids), desc=f"  {season}", leave=False):
                gid = str(meta["GAME_ID"]).zfill(10)
                bundle = self.loader.load_game_data(gid, season)
                if not bundle or bundle.get("pbp") is None or bundle["pbp"].empty:
                    continue
                feats = self._extract_features(meta, bundle)
                if feats["home_team_id"] is None or feats["away_team_id"] is None:
                    continue
                rows.append(feats)
        self._save_cache()
        if not rows:
            print("No games loaded.")
            self.master_df = pd.DataFrame()
            self.dataset = pd.DataFrame()
            return self
        self.master_df = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
        print(f"Loaded {len(self.master_df)} games. Building roster-impact features...")
        self.master_df = self._add_roster_features(self.master_df)
        if add_matchup_features is not None:
            try:
                print("Adding matchup-level (player-vs-player) features...")
                self.master_df = add_matchup_features(self.master_df, BASE_DATA_DIR)
            except Exception as exc:
                print(f"  [warn] matchup features skipped: {exc}")
        print("Engineering rolling/season/H2H features...")
        self.dataset = self.fe.engineer(self.master_df.copy()).fillna(0)
        print(f"Dataset: {self.dataset.shape[0]} games x {self.dataset.shape[1]} cols")
        return self

    def _select_features(self):
        exclude = {"game_id", "game_date", "home_team", "away_team", "season",
                   "home_score", "away_score", "home_team_id", "away_team_id"} | set(TARGETS)
        cols = []
        for c in self.dataset.columns:
            if c in exclude:
                continue
            # keep only engineered, leakage-free features
            if (c.startswith(("home_L", "away_L", "diff_L", "home_season_avg_", "away_season_avg_",
                              "diff_season_avg_", "h2h_", "home_streak", "away_streak", "diff_streak",
                              "home_season_win_pct", "away_season_win_pct"))
                    or "roster_impact" in c or "roster_form" in c or "matchup_" in c):
                cols.append(c)
        return cols

    def train(self, train_season, target_season):
        self.feature_columns = self._select_features()
        print(f"\n{len(self.feature_columns)} features selected.")
        ctx = self.dataset[self.dataset["season"] == train_season]
        tgt = self.dataset[self.dataset["season"] == target_season].sort_values("game_date")
        if tgt.empty:
            raise RuntimeError(f"No games for target season {target_season}")
        split = int(len(tgt) * 0.75)
        train_df = pd.concat([ctx, tgt.iloc[:split]]) if not ctx.empty else tgt.iloc[:split]
        test_df = tgt.iloc[split:]
        print(f"Train games: {len(train_df)} | Test games (held-out {target_season}): {len(test_df)}")

        X_train = train_df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test = test_df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        params = dict(random_state=42, n_estimators=400, learning_rate=0.03,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      verbose=-1, n_jobs=-1)

        for tgt_name in TARGETS:
            y_train = train_df[tgt_name].astype(float)
            y_test = test_df[tgt_name].astype(float)
            if tgt_name == "home_win":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc",
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]
                self.metrics[tgt_name] = {"accuracy": float(accuracy_score(y_test, preds)),
                                          "auc": float(roc_auc_score(y_test, probs)),
                                          "baseline_home_rate": float(y_test.mean())}
                print(f"  home_win  acc={self.metrics[tgt_name]['accuracy']:.4f} "
                      f"auc={self.metrics[tgt_name]['auc']:.4f} "
                      f"(home-pick baseline {self.metrics[tgt_name]['baseline_home_rate']:.3f})")
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="mae",
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                preds = model.predict(X_test)
                self.metrics[tgt_name] = {"mae": float(mean_absolute_error(y_test, preds)),
                                          "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))}
                print(f"  {tgt_name:12s} mae={self.metrics[tgt_name]['mae']:.3f} "
                      f"rmse={self.metrics[tgt_name]['rmse']:.3f}")
            self.models[tgt_name] = model
            with open(os.path.join(MODEL_DIR, f"{tgt_name}_model_2025_26.pkl"), "wb") as f:
                pickle.dump(model, f)
            if hasattr(model, "feature_importances_"):
                imp = (pd.DataFrame({"feature": self.feature_columns,
                                     "importance": model.feature_importances_})
                       .sort_values("importance", ascending=False).head(20))
                plt.figure(figsize=(9, 7))
                sns.barplot(x="importance", y="feature", data=imp)
                plt.title(f"Top features — {tgt_name} (2025-26)")
                plt.tight_layout()
                plt.savefig(os.path.join(MODEL_DIR, f"{tgt_name}_feat_imp_2025_26.png"))
                plt.close()

        self.test_df = test_df
        with open(os.path.join(OUTPUT_DIR, "metrics_2025_26.json"), "w") as f:
            json.dump({"train_season": train_season, "target_season": target_season,
                       "n_train": int(len(train_df)), "n_test": int(len(test_df)),
                       "n_features": len(self.feature_columns), "metrics": self.metrics}, f, indent=2)
        return self

    def test_single_game(self, game_id=None):
        """Predict one held-out game and compare to the actual result."""
        td = self.test_df
        if game_id:
            row = td[td["game_id"] == str(game_id).zfill(10)]
            if row.empty:
                row = td.iloc[[len(td) // 2]]
        else:
            row = td.iloc[[len(td) // 2]]
        g = row.iloc[0]
        X = row[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        out = {
            "game_id": g["game_id"], "date": str(g["game_date"].date()),
            "matchup": f"{g['away_team']} @ {g['home_team']}",
            "actual": {"home_score": float(g["home_score"]), "away_score": float(g["away_score"]),
                       "point_diff": float(g["point_diff"]), "total_score": float(g["total_score"]),
                       "home_win": int(g["home_win"]),
                       "winner": g["home_team"] if g["home_win"] else g["away_team"]},
            "predicted": {},
        }
        for name, model in self.models.items():
            if name == "home_win":
                p = float(model.predict_proba(X)[:, 1][0])
                out["predicted"]["home_win_prob"] = round(p, 3)
                out["predicted"]["predicted_winner"] = g["home_team"] if p > 0.5 else g["away_team"]
            else:
                out["predicted"][name] = round(float(model.predict(X)[0]), 2)
        out["correct_winner"] = (out["predicted"]["predicted_winner"] == out["actual"]["winner"])
        with open(os.path.join(OUTPUT_DIR, "single_game_test_2025_26.json"), "w") as f:
            json.dump(out, f, indent=2)
        return out


def main():
    print("=" * 70)
    print("NBA 2025-26 PREDICTOR")
    print("=" * 70)
    predictor = NBAPredictor()
    predictor.load_and_prepare(seasons=["2024_2025", "2025_2026"])
    if predictor.dataset is None or predictor.dataset.empty:
        print("No data — aborting.")
        return
    predictor.train(train_season="2024_2025", target_season="2025_2026")

    print("\n--- Single-game test (held-out 2025-26) ---")
    result = predictor.test_single_game()
    print(json.dumps(result, indent=2))
    print("\nDone. Metrics -> output/metrics_2025_26.json")


if __name__ == "__main__":
    main()
