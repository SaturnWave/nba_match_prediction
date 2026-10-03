"""
Pre-game roster features: value the players who are DRESSED tonight by what
they did BEFORE tonight.

WHY THIS REPLACES _add_roster_features AND add_matchup_features
    Both read the roster of a game from the game itself. The impact cache lists
    every player who recorded a play-by-play event; the matchup file lists every
    player who defended a possession. The trailing values were computed strictly
    before the game - that part was verified column by column - but the SET of
    players they were summed over was read after the final buzzer, and that set
    carries the result: the number of players who appear per team-game
    correlates 0.64 with the margin of victory, because a 25-point game empties
    both benches and a 3-point game does not. roster_impact_l10_sum, a sum over
    that set, was the single most important feature of the point-differential
    model, and seven of the classifier's top eight splits were roster or
    matchup columns. A model trained that way learns "many players appeared, so
    the favourite won big", which is true and unusable: before tip-off nobody
    knows how many players will appear.

WHAT COUNTS AS KNOWN BEFORE TIP-OFF
    The box score lists ~13 players per team. Those who did not play carry a
    comment, and the comments split cleanly by WHEN they are decided:
        "DND - Injury/Illness", "NWT - Not With Team", "DND - Rest",
        "NWT - League Suspension" ...  announced on the injury report before
                                       the game: the player is unavailable
        "DNP - Coach's Decision"       decided during the game, and decided
                                       differently in a blowout
    So the pre-game roster is every listed player whose comment is not an
    unavailability designation. That set is ~13 players whether the game was
    close or not, and nothing read from it - not minutes, not events, not who
    actually got on the floor - comes from the game being predicted.

    The one honest caveat: the listing is taken from the post-game box score,
    so a player scratched minutes before tip-off appears as unavailable here
    when a forecaster working from the morning injury report would have counted
    him. That is a small timing error in the pre-game direction, not a view of
    the result.

PLAYER VALUE AS OF THE GAME
    One chronological sweep over all games. Each player carries his last ten
    appearances (impact and minutes) and his cumulative matchup totals; a game's
    features are computed from that state, and only then is the state updated
    with the game. A value used for game g therefore includes every game before
    g and nothing from g, across season boundaries - a player's October form is
    his previous April.

FEATURES (home_, away_, diff_)
    roster_impact_l10_mean   trailing-minutes-weighted mean of the dressed
                             players' trailing-10 impact
    roster_impact_l10_sum    summed trailing-10 impact of the eight dressed
                             players with the most trailing minutes. Eight
                             regardless of the game, so the count cannot leak
    roster_form_l6, _l3      the weighted mean over shorter windows
    matchup_def_quality      prior points allowed per 100 partial possessions,
                             weighted by prior possessions, over the dressed
    matchup_edge             dressed offence's prior scoring rate minus the
                             opponent's dressed defence quality
    roster_avail_minutes     sum of the dressed players' trailing minutes / 240:
                             how much of the usual rotation is available tonight
    roster_missing_impact    trailing impact x minutes share of the players who
                             appeared in the team's last three games but are not
                             dressed tonight - the value of who is out
    The first six keep their old names so a feature list built for the observed
    roster is still valid. The last two are new and are returned as a separate
    group, in the same spirit as rest/clutch/avail.

Run:  py prediction_engines/pregame_roster.py [--in output/engineered_dataset_db.pkl]
                                              [--out output/engineered_dataset_pregame.pkl]
"""
import argparse
import os
import pickle
import re
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BOX_PLAYER_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "box_player_traditional.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v5.pkl")
MATCHUP_CACHE = os.path.join(PROJECT_ROOT, "matchup_cache_v1.pkl")
DEFAULT_IN = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")

# A comment that means "not available tonight", decided before the game.
# "DNP - Coach's Decision" deliberately does not match: that player dressed.
UNAVAILABLE_RE = re.compile(
    r"Injur|Illness|Rest|Health|Not With Team|NWT|Suspen|Personal|DND|Concussion|Trade",
    re.IGNORECASE)
IMPACT_WINDOW = 10
TOP_N_SUM = 8
RECENT_GAMES_FOR_MISSING = 3
TEAM_MINUTES = 240.0

ROSTER_METRICS = ["roster_impact_l10_mean", "roster_impact_l10_sum",
                  "roster_form_l6", "roster_form_l3"]
MATCHUP_METRICS = ["matchup_def_quality", "matchup_edge"]
EXTRA_METRICS = ["roster_avail_minutes", "roster_missing_impact"]
REPLACED_COLUMNS = [f"{s}_{m}" for m in ROSTER_METRICS + MATCHUP_METRICS
                    for s in ("home", "away", "diff")]
EXTRA_COLUMNS = [f"{s}_{m}" for m in EXTRA_METRICS for s in ("home", "away", "diff")]


# ---------------------------------------------------------------------------
#  Inputs
# ---------------------------------------------------------------------------
def parse_minutes(raw):
    """'35:23' -> 35.38; the feed also writes '4.000000:06', which is 4:06."""
    s = raw.astype("string")
    parts = s.str.extract(r"^(\d+)(?:\.\d+)?(?::(\d+))?")
    minutes = (pd.to_numeric(parts[0], errors="coerce").fillna(0)
               + pd.to_numeric(parts[1], errors="coerce").fillna(0) / 60.0)
    return minutes.where(s.notna(), np.nan).astype(float)


def load_listings(box_path=BOX_PLAYER_PATH):
    """One row per listed player-game: who was on the team sheet, who was
    unavailable, and how many minutes they ended up playing (used only to
    update state AFTER the game)."""
    box = pd.read_pickle(box_path)
    out = pd.DataFrame({
        "game_id": box["game_id"].astype(str).str.zfill(10),
        "team": box["team_abbreviation"].astype(str),
        "person_id": box["player_id"].astype("int64"),
        "minutes": parse_minutes(box["min"]).fillna(0.0),
    })
    comment = box["comment"].astype("string").fillna("")
    out["unavailable"] = comment.str.contains(UNAVAILABLE_RE).fillna(False).astype(bool).values
    return out


def load_impacts(cache_path=IMPACT_CACHE):
    """{game_id: {person_id: impact}} from the v4 impact cache."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    by_game = {}
    for gid, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        players = {}
        for value in (entry.get("players") or {}).values():
            if isinstance(value, dict) and value.get("person_id") is not None:
                players[int(value["person_id"])] = float(value.get("impact", 0.0))
        by_game[str(gid).zfill(10)] = players
    return by_game


def load_matchup_totals(cache_path=MATCHUP_CACHE):
    """Per game: each defender's (points allowed, partial possessions) and each
    scorer's (points, partial possessions), summed over the game."""
    if not os.path.exists(cache_path):
        return {}, {}
    with open(cache_path, "rb") as f:
        long_df = pickle.load(f)["long_df"]
    long_df = long_df[long_df["partial_poss"] > 0]
    defence = (long_df.groupby(["game_id", "person_id_def"])
               .agg(pts=("player_points", "sum"), poss=("partial_poss", "sum")).reset_index())
    offence = (long_df.groupby(["game_id", "person_id_off"])
               .agg(pts=("player_points", "sum"), poss=("partial_poss", "sum")).reset_index())

    def to_dict(frame, id_col):
        out = defaultdict(dict)
        for gid, pid, pts, poss in zip(frame["game_id"], frame[id_col], frame["pts"], frame["poss"]):
            out[str(gid).zfill(10)][int(pid)] = (float(pts), float(poss))
        return out
    return to_dict(defence, "person_id_def"), to_dict(offence, "person_id_off")


# ---------------------------------------------------------------------------
#  Player state, updated game by game
# ---------------------------------------------------------------------------
class PlayerState:
    __slots__ = ("impacts", "minutes", "def_pts", "def_poss", "off_pts", "off_poss")

    def __init__(self):
        self.impacts = deque(maxlen=IMPACT_WINDOW)
        self.minutes = deque(maxlen=IMPACT_WINDOW)
        self.def_pts = self.def_poss = self.off_pts = self.off_poss = 0.0

    def trailing_impact(self, window):
        if not self.impacts:
            return None
        recent = list(self.impacts)[-window:]
        return sum(recent) / len(recent)

    @property
    def trailing_minutes(self):
        return sum(self.minutes) / len(self.minutes) if self.minutes else 0.0


def _weighted_mean(values, weights):
    w = np.asarray(weights, float)
    if w.sum() <= 0:
        return np.nan
    return float(np.dot(np.asarray(values, float), w) / w.sum())


def _rate(pts, poss):
    return 100.0 * pts / poss if poss > 0 else None


def side_features(dressed, recent_players, states, league_def_rate):
    """Features for one team from the state as it stood before the game.

    dressed         person ids on tonight's sheet without an unavailability tag
    recent_players  ids who played in the team's last few games (for who is out)
    """
    valued = []          # (impact_l10, l6, l3, trailing minutes) for dressed with history
    def_pairs, off_pairs = [], []
    for pid in dressed:
        st = states.get(pid)
        if st is None:
            continue
        minutes = st.trailing_minutes
        imp = st.trailing_impact(IMPACT_WINDOW)
        if imp is not None and minutes > 0:
            valued.append((imp, st.trailing_impact(6), st.trailing_impact(3), minutes))
        d = _rate(st.def_pts, st.def_poss)
        if d is not None:
            def_pairs.append((d, st.def_poss))
        o = _rate(st.off_pts, st.off_poss)
        if o is not None:
            off_pairs.append((o, st.off_poss))

    out = {m: np.nan for m in ROSTER_METRICS + EXTRA_METRICS + ["_off_quality", "matchup_def_quality"]}
    if valued:
        imp10, imp6, imp3, mins = (np.array(v, float) for v in zip(*valued))
        out["roster_impact_l10_mean"] = _weighted_mean(imp10, mins)
        out["roster_form_l6"] = _weighted_mean(imp6, mins)
        out["roster_form_l3"] = _weighted_mean(imp3, mins)
        top = np.argsort(-mins)[:TOP_N_SUM]
        out["roster_impact_l10_sum"] = float(imp10[top].sum())
        out["roster_avail_minutes"] = float(mins.sum() / TEAM_MINUTES)
    if def_pairs:
        out["matchup_def_quality"] = _weighted_mean(*zip(*def_pairs))
    elif league_def_rate is not None:
        out["matchup_def_quality"] = league_def_rate
    if off_pairs:
        out["_off_quality"] = _weighted_mean(*zip(*off_pairs))

    # Who is out: played in the team's recent games, not dressed tonight.
    missing = 0.0
    total_minutes = sum(states[p].trailing_minutes for p in recent_players if p in states)
    if total_minutes > 0:
        for pid in recent_players - set(dressed):
            st = states.get(pid)
            if st is None:
                continue
            imp = st.trailing_impact(IMPACT_WINDOW)
            if imp is not None:
                missing += imp * st.trailing_minutes / total_minutes
    out["roster_missing_impact"] = missing if recent_players else np.nan
    return out


def build_pregame_features(master, listings, impacts, def_totals, off_totals, verbose=True):
    """One row per game with home_/away_/diff_ pre-game roster features.

    `master` needs game_id, game_date, home_team, away_team. Games are swept in
    date order; a team plays at most once a day, so updating after each game is
    the same as updating after each day.
    """
    t0 = time.time()
    games = master[["game_id", "game_date", "season", "home_team", "away_team"]].copy()
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    listed_by_game = {}
    for gid, frame in listings.groupby("game_id", sort=False):
        listed_by_game[gid] = {
            team: (t["person_id"].to_numpy(), t["unavailable"].to_numpy(), t["minutes"].to_numpy())
            for team, t in frame.groupby("team", sort=False)}

    states = defaultdict(PlayerState)
    recent_by_team = defaultdict(lambda: deque(maxlen=RECENT_GAMES_FOR_MISSING))
    season_of_team = {}
    league_def_pts = league_def_poss = 0.0
    rows = []
    no_sheet = 0
    empty_sheet = (np.array([], dtype="int64"), np.array([], dtype=bool), None)

    for gid, gdate, season, home, away in games.itertuples(index=False):
        sheet = listed_by_game.get(gid, {})
        league_rate = _rate(league_def_pts, league_def_poss)
        row = {"game_id": gid}
        per_side = {}
        for side, team in (("home", home), ("away", away)):
            # A new season starts with no "recent games": last April's roster
            # says nothing about who is missing in October, it only says who
            # left in the summer.
            if season_of_team.get(team) != season:
                recent_by_team[team].clear()
                season_of_team[team] = season
            ids, unavailable, _minutes = sheet.get(team, empty_sheet)
            if len(ids) == 0:
                no_sheet += 1
            dressed = [int(p) for p, u in zip(ids, unavailable) if not u]
            recent = set().union(*recent_by_team[team]) if recent_by_team[team] else set()
            per_side[side] = side_features(dressed, recent, states, league_rate)
        for side in ("home", "away"):
            other = "away" if side == "home" else "home"
            feats = per_side[side]
            off_q, opp_def = feats.pop("_off_quality"), per_side[other]["matchup_def_quality"]
            feats["matchup_edge"] = (off_q - opp_def
                                     if not (pd.isna(off_q) or pd.isna(opp_def)) else np.nan)
            for metric, value in feats.items():
                row[f"{side}_{metric}"] = value
        rows.append(row)

        # Now, and only now, the game happened: update every player who played.
        game_def = def_totals.get(gid, {})
        game_off = off_totals.get(gid, {})
        game_imp = impacts.get(gid, {})
        for team in (home, away):
            ids, _unavailable, minutes = sheet.get(team, empty_sheet)
            if minutes is None:
                continue
            played = set()
            for pid, mins in zip(ids, minutes):
                if mins <= 0:
                    continue
                pid = int(pid)
                played.add(pid)
                st = states[pid]
                st.minutes.append(float(mins))
                if pid in game_imp:
                    st.impacts.append(game_imp[pid])
                if pid in game_def:
                    pts, poss = game_def[pid]
                    st.def_pts += pts
                    st.def_poss += poss
                    league_def_pts += pts
                    league_def_poss += poss
                if pid in game_off:
                    pts, poss = game_off[pid]
                    st.off_pts += pts
                    st.off_poss += poss
            recent_by_team[team].append(played)

    out = pd.DataFrame(rows)
    for metric in ROSTER_METRICS + MATCHUP_METRICS + EXTRA_METRICS:
        out[f"diff_{metric}"] = out[f"home_{metric}"] - out[f"away_{metric}"]
    if verbose:
        print(f"  pre-game roster: {len(out):,} mac, {no_sheet} takim-mac listesiz, "
              f"{time.time() - t0:.1f} sn")
    return out


# ---------------------------------------------------------------------------
#  Dataset surgery
# ---------------------------------------------------------------------------
def replace_roster_features(dataset, pregame, keep_observed_as=None):
    """Swap the observed-roster columns for the pre-game ones.

    keep_observed_as: a prefix under which the old columns are kept, so an
    experiment can fit the observed and pre-game arms on one frame.
    """
    ds = dataset.copy()
    ds["game_id"] = ds["game_id"].astype(str).str.zfill(10)
    present = [c for c in REPLACED_COLUMNS if c in ds.columns]
    if keep_observed_as:
        ds = ds.rename(columns={c: f"{keep_observed_as}{c}" for c in present})
    else:
        ds = ds.drop(columns=present)
    # A frame that already went through this once carries the extras too;
    # they are rebuilt here, so the old copies go before the merge.
    ds = ds.drop(columns=[c for c in EXTRA_COLUMNS if c in ds.columns])
    ds = ds.merge(pregame, on="game_id", how="left")
    # The rest of the dataset was fillna(0)'d by the builder; match that so a
    # team with no history is treated the same way in both regimes.
    ds[REPLACED_COLUMNS + EXTRA_COLUMNS] = ds[REPLACED_COLUMNS + EXTRA_COLUMNS].fillna(0.0)
    return ds


def build_from_dataset(in_path=DEFAULT_IN, verbose=True):
    blob = pd.read_pickle(in_path)
    dataset = blob["dataset"]
    if verbose:
        print("  listeler, impact ve matchup okunuyor...")
    listings = load_listings()
    impacts = load_impacts()
    def_totals, off_totals = load_matchup_totals()
    pregame = build_pregame_features(dataset, listings, impacts, def_totals, off_totals, verbose)
    new_dataset = replace_roster_features(dataset, pregame)
    groups = dict(blob.get("feature_groups") or {})
    groups["pregame_extra"] = EXTRA_COLUMNS
    return {"dataset": new_dataset, "features": blob["features"], "feature_groups": groups,
            "rest_features": groups.get("rest", []),
            "seasons": sorted(new_dataset["season"].unique().tolist()),
            "roster": "pregame"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    blob = build_from_dataset(args.in_path)
    pd.to_pickle(blob, args.out)
    ds = blob["dataset"]
    print(f"Yazildi: {args.out}  ({ds.shape[0]} mac x {ds.shape[1]} sutun, "
          f"{len(blob['features'])} feature + {len(EXTRA_COLUMNS)} pregame_extra)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
