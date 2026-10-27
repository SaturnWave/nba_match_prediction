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
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import csv_tables  # noqa: E402 - sibling module; one instance per process, so one read of the tables

PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v5.pkl")
MATCHUP_CACHE = os.path.join(PROJECT_ROOT, "matchup_cache_v1.pkl")
DEFAULT_IN = os.path.join(OUTPUT_DIR, "engineered_dataset_db.pkl")
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")

# Every comment the box scores have used, by what it says about the player before
# tip-off. A substring pattern used to decide this, and it counted the 2026-27
# "Inactive - G League - Two-Way" and "Inactive - Coach's Decision" listings as
# dressed. An exact list cannot drift that way: a comment in neither set stops the
# build until someone decides which set it belongs to. "DNP - Coach's Decision" is
# the one non-empty comment of a player who dressed. "DNP - Injury/Illness" stays
# unavailable as it always was here, although such a player was on the active
# list (the audit's finding A4) - changing that changes the features the
# production models were trained on.
DRESSED_COMMENTS = frozenset({"", "DNP - Coach's Decision"})
UNAVAILABLE_COMMENTS = frozenset({
    "DND - Injury/Illness", "DNP - Injury/Illness", "NWT - Injury/Illness", "Inactive - Injury/Illness",
    "NWT - Not With Team", "Inactive - Not With Team", "DND - Rest", "DNP - Rest", "NWT - Rest",
    "NWT - Personal", "DND - Personal", "DNP - Personal", "DNP_PERSONAL",
    "NWT - League Suspension", "DNP - League Suspension", "NWT - Suspended", "NWT_LEAGUE_SUSPENSION",
    "DND_LEAGUE_SUSPENSION", "DNP_LEAGUE_SUSPENSION", "NWT_TEAM_SUSPENSION", "DNP_TEAM_SUSPENSION",
    "NWT - Health and Safety Protocols", "DND - Health and Safety Protocols",
    "DNP - Health and Safety Protocols", "NWT_HEALTH_AND_SAFETY_PROTOCOLS", "DND_HEALTH_AND_SAFETY_PROTOCOLS",
    "DND-Return to Competition Reconditioning", "NWT-Return to Competition Reconditioning",
    "NWT_RETURN_TO_COMPETITION_RECONDITIONING", "DND - Concussion Protocol", "DNP - Concussion Protocol",
    "NWT_CONCUSSION_PROTOCOL", "NWT - Trade Pending", "DND - Trade Pending", "DNP_TRADE",
    "DND - Coach's Decision", "DND_COACH", "NWT_COACH",
    "Inactive - Coach's Decision", "Inactive - G League - Two-Way",
})
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


class UnknownComment(ValueError):
    """A box-score comment that neither DRESSED_COMMENTS nor UNAVAILABLE_COMMENTS knows."""


def classify_comments(comments):
    """True where a comment marks the player unavailable before tip-off; raises on a comment it does not know."""
    text = comments.astype("string").fillna("").str.strip()
    known = text.isin(DRESSED_COMMENTS | UNAVAILABLE_COMMENTS)
    if not known.all():
        unseen = text[~known].value_counts()
        raise UnknownComment("taninmayan kutu skoru yorumu: "
                             + "; ".join(f"{value!r} ({count})" for value, count in unseen.items())
                             + " - pregame_roster.DRESSED_COMMENTS ya da UNAVAILABLE_COMMENTS'e eklenmeli")
    return text.isin(UNAVAILABLE_COMMENTS).to_numpy()


def load_box():
    """Every listed player-game, from the tables built off the CSVs (csv_tables.py)."""
    return csv_tables.load("box_player_traditional")


def load_listings(box=None):
    """One row per listed player-game: who was on the team sheet, who was
    unavailable, and how many minutes they ended up playing (used only to
    update state AFTER the game)."""
    box = load_box() if box is None else box
    out = pd.DataFrame({
        "game_id": box["game_id"].astype(str).str.zfill(10),
        "team": box["team_abbreviation"].astype(str),
        "person_id": box["player_id"].astype("int64"),
        "minutes": parse_minutes(box["min"]).fillna(0.0),
    })
    out["unavailable"] = classify_comments(box["comment"])
    return out


class EmptySheet(ValueError):
    """A team has no listed sheet to carry forward into an unplayed game."""


def carried_forward_listings(listings, played, upcoming, unavailable_overrides=None):
    """Team sheets for unplayed games: each team's sheet from its last game, carried forward.

    No team publishes tonight's sheet before tonight. Until the injury report
    is archived, a team's upcoming sheet is the one it listed for its last
    game, with the same players marked unavailable. `unavailable_overrides`
    ({(team, person_id): bool}) is where report-time statuses plug in. Minutes
    are zero, so an unplayed game updates nobody's state.

    played    game_id, game_date, home_team, away_team of the games already listed
    upcoming  game_id, home_team, away_team of the games to sheet
    """
    order = played.sort_values(["game_date", "game_id"])
    last_game = {}
    for gid, home, away in order[["game_id", "home_team", "away_team"]].itertuples(index=False):
        last_game[home] = last_game[away] = gid
    by_game_team = {key: frame for key, frame in listings.groupby(["game_id", "team"], sort=False)}
    overrides = unavailable_overrides or {}
    sheets = []
    for gid, home, away in upcoming[["game_id", "home_team", "away_team"]].itertuples(index=False):
        for team in (home, away):
            source = by_game_team.get((last_game.get(team), team))
            if source is None or source.empty:
                raise EmptySheet(f"{gid}: {team} icin tasinacak kadro listesi yok")
            sheet = source.assign(game_id=gid, minutes=0.0)
            sheet["unavailable"] = [overrides.get((team, int(pid)), flag)
                                    for pid, flag in zip(sheet["person_id"], sheet["unavailable"])]
            sheets.append(sheet)
    return pd.concat(sheets, ignore_index=True) if sheets else listings.iloc[0:0]


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
