"""
Value score: what a player's plays were worth in points, not how many there were.

WHY A SECOND SCORE
    The impact score (predict_2025_2026.compute_game_impact) counts production
    with context flavour: a made shot is +2 or +3 times a difficulty factor, a
    rebound +0.6 to +0.9 plus bonuses, a steal +1.4, a turnover about -1. Three
    things make it a stat-padder's friend, and 2017-18 shows all three at once -
    Westbrook (25/10/10 on .524 true shooting) scores 40.7 a game, Curry (27 on
    .675) 32.0, and the top of the table is big men who rebound:

      * a missed shot costs nothing. Twenty-one attempts a game at 45% earn
        every make and pay for no miss, so volume is rewarded and efficiency
        is invisible;
      * a defensive rebound is worth about a third of a basket, although most
        of them are the expected end of an opponent's miss - the team gets it
        whoever's hands it lands in;
      * assists and free throws do not exist, and nothing is discounted when
        the game is already decided.

    Across 167 regulars that season the impact score per 36 minutes correlates
    0.14 with the player's own on-court plus-minus per 36. That is the number
    this module is trying to raise.

HOW VALUE IS COUNTED
    Every event is priced by how it moves expected points against a baseline
    of BASELINE points per possession (1.00: a little below the league's ~1.08,
    so taking shots at ordinary efficiency is worth a little, not nothing):
        made shot        points - BASELINE                 (+1.0 for a two, +2.0 for a three)
        missed shot      -BASELINE x (1 - OREB_SHARE)       (-0.74: the possession is usually gone)
        free throw       +1 if made, minus the share of a possession the attempt used
        turnover         -BASELINE - TRANSITION             (-1.15)
        steal            +BASELINE + TRANSITION             (+1.15)
        offensive board  +BASELINE x (1 - OREB_SHARE)       (+0.74: a possession recovered)
        defensive board  +BASELINE x OREB_SHARE             (+0.26: a chance denied)
        block            +0.6
        foul             -0.3 personal, -0.6 shooting, 0 offensive (its turnover row pays)
    No style bonuses: a step-back three and a wide-open one are both worth
    +2.0, because they both put three points on the board.

PASSING
    An assisted basket is one play made by two players. The passer is paid
    ASSIST_SHARE of the basket's value ON TOP of the scorer's credit, not out
    of it, and a three pays its passer twice what a two does - most threes
    exist because of the pass.

    "On top" is a measurement, not a preference. Against on-court plus-minus
    per 36 minutes, regulars of 2017-18, 2020-21 and 2023-24:
        no assist credit                     0.394
        a flat +0.35 per assist              0.442
        30% of the basket, on top            0.445
        30% of the basket, out of the scorer 0.430   (20%: 0.434, 40%: 0.413)
    Taking the share from the scorer tracked worse in all three seasons and
    worse the more was taken: making the shot is still the scorer's work, and
    the pass is what made it a better shot.

    Who passed is read from the description, "(Williams 4 AST)". Four things
    make that harder than it looks, and the first version of this engine lost
    12% of all assists to them:
      * the name has to be matched against everyone in the game, not only the
        players the play-by-play has named so far - a passer whose first
        appearance is an assist was not being recognised;
      * the description spells names its own way: without accents, and
        without a suffix ("Butler" for the player column's "Butler III");
      * where teammates share a surname it prefixes the first name ("G.
        Antetokounmpo", "Ja. Green"), which the player column never does;
      * and where even that leaves two candidates, the number in the
        description is the passer's running total for the game, so it is
        whoever stands one below it, with the box score's totals breaking ties.
    With all four handled the engine credits 99.8-100% of the box score's
    assists, and 99.7-99.98% of player-games match it exactly.

    Two kinds of pass never reach the play-by-play, and both come from the
    tracking table: the secondary assist (the pass to the passer) and the
    free-throw assist (a pass that drew a shooting foul and at least one made
    free throw). They are added per game at SECONDARY_ASSIST and FT_ASSIST
    each. Their measured effect is small (+0.001 to +0.004) but in the same
    direction every season. One caveat: the feed records about twice as many
    free-throw assists per game from 2019-20 on as before it, so that count is
    not comparable across the boundary.

    What is deliberately NOT paid for is passing volume. The tracking table's
    PASS column is every pass made - potential assists, the passes that would
    have been assists had the shot fallen, are not in the data this project
    holds - and a credit per pass made the score track plus-minus worse
    (-0.007 at 0.01 a pass, -0.014 at 0.02). The first single-game engine had
    such a term (a bonus when passes outnumbered assists five to one); it is
    not carried over.

FREE THROWS
    The play-by-play's free throws match the box score's exactly, made and
    attempted, in every player-game checked. What they cost is the question.
    A flat 0.44 possessions per attempt is the box-score approximation, used
    when nothing says what kind of trip it was. The play-by-play says: an
    n-shot trip uses one possession (1/n per attempt), and an and-one, a
    technical, a flagrant or a clear-path free throw uses none, because the
    ball stays with the shooting team or the possession was already paid for
    by the basket. This is bookkeeping made exact rather than a gain: against
    plus-minus the two rules are indistinguishable (0.431 and 0.433). It moves
    a player by a few tenths a game at most - up for those who draw and-ones.

LEVERAGE
    Each event is multiplied by how much the game still hung on it: 1.0 while
    the game is live, sliding to 0.3 when it is 25 points apart in the fourth,
    1.25 in the last five minutes of a one-possession-to-five-point game. A
    stat line built in garbage time shrinks; one built when it mattered does
    not.

    That is the right weighting for crediting a PLAYER and the wrong one for
    a TEAM feature: how far a team won by predicts its next game, garbage time
    included. So the player ratings use the weighted run (default) and the
    model's impact cache is built from --no-leverage (impact_v5.py), the same
    pricing with every weight at 1.0. Measured: weighted values as model
    features cost Brier +0.003 and raised winner-vs-margin contradictions from
    8.9% to 12.4%; unweighted values were indistinguishable from the old score.

WHAT IT IS NOT
    Still a box-plus-context score. It does not see screens, spacing, closeouts
    or who a player guarded; those live in the tracking and matchup tables and
    are a separate project. Read "value" as production priced in points and
    weighted by leverage, and "impact" as the raw production it is compared to.

Run:  py prediction_engines/value_engine.py --seasons 2017_2018 [--eval]
      py prediction_engines/value_engine.py --all --refresh      (every season, recomputed)
      py prediction_engines/value_engine.py --all --refresh --no-leverage --cache output/value_cache_unweighted.pkl
Output: output/value_cache_v1.pkl   {game_id: {person_id: {name, team, value, components..., events}}}
"""
import argparse
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "nba_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_PATH = os.path.join(OUTPUT_DIR, "value_cache_v1.pkl")
GAMES_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "games.pkl")
BOX_PLAYER_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "box_player_traditional.pkl")
TRACKING_PATH = os.path.join(PROJECT_ROOT, "phonedb_cache", "player_tracking.pkl")
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")

PRICING_VERSION = "5.1"

BASELINE = 1.00
OREB_SHARE = 0.26
TRANSITION = 0.15
ASSIST_SHARE = 0.30            # of an assisted basket's value, paid to the passer on top
SECONDARY_ASSIST = 0.15        # per tracked secondary assist: half the share of a two
FT_ASSIST = 0.17               # per tracked free-throw assist: the share of an average trip
BLOCK = 0.6
FOUL_PERSONAL, FOUL_SHOOTING, FOUL_TECHNICAL = -0.3, -0.6, -0.5
FT_UNKNOWN_TRIP = 0.44         # possessions per attempt when the description names no trip
CLUTCH_WEIGHT = 1.25
GARBAGE_FLOOR = 0.3

COMPONENTS = ("scoring", "misses", "free_throws", "assists", "rebounds",
              "stops", "turnovers", "fouls")
# Kept only on request (--raw): the pieces a different assist rule would be
# rebuilt from, so a rule can be measured without recomputing every game.
RAW_FIELDS = ("core", "assisted_scoring", "assist_base", "assist_weight", "assists_n",
              "leverage_mean", "secondary_assists", "ft_assists")
ASSIST_RE = re.compile(r"\(([^()]+?) (\d+) AST\)")
# "Ja. Green" -> ("Ja", "Green"); "Porter Jr." -> (None, "Porter Jr.")
DESCRIBED_NAME_RE = re.compile(r"^(?:([A-Z][a-z]{0,3})\. )?(.+)$")
NAME_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv)\.?$")
UNDECOMPOSABLE = str.maketrans({"\u00f8": "o", "\u0142": "l", "\u0111": "d", "\u0131": "i",
                                "\u00df": "ss", "\u00d8": "O", "\u0141": "L", "\u0110": "D"})
FREE_THROW_RE = re.compile(r"Free Throw\s*(Technical|Flagrant|Clear Path)?\s*(?:\d of (\d))?")


# ---------------------------------------------------------------------------
#  What the play-by-play does not carry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackedPassing:
    """One player's passes the play-by-play never names, for one game."""
    secondary: float
    free_throw: float
    team: str
    surname: str


@dataclass(frozen=True)
class GameContext:
    """Box-score and tracking facts about one game.

    final_assists   person id -> assists the box score credits him with
    passers         team -> [(person id, first name, family name)] for everyone
                    with an assist
    tracking        person id -> TrackedPassing
    Empty when the local caches are absent: the engine then resolves passers
    from the play-by-play alone and adds no tracked passing.
    """
    final_assists: dict = field(default_factory=dict)
    passers: dict = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)


NO_CONTEXT = GameContext()


def load_contexts(game_ids):
    """{game_id: GameContext} for the given games, from the local caches."""
    wanted = set(game_ids)
    contexts = {gid: {"final": {}, "passers": defaultdict(list), "tracking": {}} for gid in wanted}

    if os.path.exists(BOX_PLAYER_PATH):
        box = pd.read_pickle(BOX_PLAYER_PATH)[["game_id", "team_abbreviation", "player_id",
                                               "player_name", "ast"]]
        box = box[box["game_id"].isin(wanted) & (box["ast"].fillna(0) > 0)]
        for gid, team, pid, full, ast in box.itertuples(index=False):
            first, _, surname = str(full).partition(" ")
            contexts[gid]["final"][int(pid)] = int(ast)
            contexts[gid]["passers"][team].append((int(pid), first, surname or first))
    else:
        print(f"  [uyari] {os.path.basename(BOX_PLAYER_PATH)} yok - asist toplamlari kullanilamiyor")

    if os.path.exists(TRACKING_PATH):
        track = pd.read_pickle(TRACKING_PATH)[["game_id", "team_abbreviation", "player_id",
                                               "player_name", "sast", "ftast"]]
        track = track[track["game_id"].isin(wanted)
                      & ((track["sast"].fillna(0) > 0) | (track["ftast"].fillna(0) > 0))]
        for gid, team, pid, full, sast, ftast in track.itertuples(index=False):
            contexts[gid]["tracking"][int(pid)] = TrackedPassing(
                float(sast or 0.0), float(ftast or 0.0), team, _surname(full))
    else:
        print(f"  [uyari] {os.path.basename(TRACKING_PATH)} yok - ikincil ve serbest atis "
              f"asistleri eklenmiyor")

    return {gid: GameContext(c["final"], dict(c["passers"]), c["tracking"])
            for gid, c in contexts.items()}


def _surname(full_name):
    parts = str(full_name).split(" ", 1)
    return parts[1] if len(parts) == 2 else parts[0]


# ---------------------------------------------------------------------------
#  Game state
# ---------------------------------------------------------------------------
def clock_seconds(value):
    """'PT07M23.00S' -> 443.0; anything unreadable -> 0."""
    if isinstance(value, str) and value.startswith("PT"):
        try:
            minutes, rest = value[2:].split("M")
            return int(minutes) * 60 + float(rest.rstrip("S") or 0)
        except ValueError:
            return 0.0
    return 0.0


def leverage(period, seconds_left, margin):
    """How much the game still hung on this play, 0.3 .. 1.25."""
    m = abs(margin)
    if period >= 4:
        if seconds_left <= 300 and m <= 5:
            return CLUTCH_WEIGHT
        if m <= 10:
            return 1.0
        # 10 points apart -> 1.0, 25 or more -> the floor, linear between
        return max(GARBAGE_FLOOR, 1.0 - (m - 10) / 15.0 * (1.0 - GARBAGE_FLOOR))
    if m <= 15:
        return 1.0
    if m <= 25:
        return 0.7
    return 0.5


# ---------------------------------------------------------------------------
#  Pricing one play
# ---------------------------------------------------------------------------
def free_throw_possessions(description):
    """The share of a possession one free-throw attempt used up.

    "Free Throw 1 of 2" -> 0.5, "1 of 3" -> 1/3. An and-one ("1 of 1"), a
    technical, a flagrant or a clear-path attempt uses none: the basket has
    already paid for the possession, or the ball stays with the shooting team.
    """
    match = FREE_THROW_RE.search(description)
    if match is None:
        return FT_UNKNOWN_TRIP
    retained_kind, trip_size = match.groups()
    if retained_kind is not None:
        return 0.0
    if trip_size is None:
        return FT_UNKNOWN_TRIP
    shots = int(trip_size)
    return 0.0 if shots == 1 else 1.0 / shots


def price(row, last_miss_team):
    """(component, value) for the player named on the row, or None.

    `last_miss_team` is who missed the most recent shot or free throw: the
    description of a rebound ("REBOUND (Off:0 Def:1)") carries the player's
    running totals, not the kind of this rebound, so the kind has to be read
    from whose miss it followed.
    """
    kind = row.get("actionType")
    desc = row.get("description") if isinstance(row.get("description"), str) else ""
    if kind == "Made Shot":
        points = 3.0 if row.get("shotValue") == 3 else 2.0
        return "scoring", points - BASELINE
    if kind == "Missed Shot":
        return "misses", -BASELINE * (1.0 - OREB_SHARE)
    if kind == "Free Throw":
        made = "MISS" not in desc
        return "free_throws", (1.0 if made else 0.0) - BASELINE * free_throw_possessions(desc)
    if kind == "Rebound":
        offensive = last_miss_team is not None and row.get("teamTricode") == last_miss_team
        return "rebounds", BASELINE * ((1.0 - OREB_SHARE) if offensive else OREB_SHARE)
    if kind == "Turnover":
        return "turnovers", -(BASELINE + TRANSITION)
    if "STEAL" in desc:
        return "stops", BASELINE + TRANSITION
    if "BLOCK" in desc:
        return "stops", BLOCK
    if kind == "Foul":
        if "OFF.FOUL" in desc or "Offensive" in desc:
            return None                      # its Turnover row carries the cost
        if "S.FOUL" in desc or "Shooting" in desc:
            return "fouls", FOUL_SHOOTING
        if "T.FOUL" in desc or "Technical" in desc:
            return "fouls", FOUL_TECHNICAL
        return "fouls", FOUL_PERSONAL
    return None


# ---------------------------------------------------------------------------
#  Who passed
# ---------------------------------------------------------------------------
def _person_id(row):
    """The row's person id, or a 'name:<surname>' stand-in for feeds without one."""
    pid = row.get("personId")
    if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pid == 0:
        return f"name:{row.get('playerName')}"
    return int(pid)


def name_key(name):
    """A surname as the feed might spell it anywhere: no accents, no suffix.

    The description of a basket writes "Jokic" and "Butler" where the same
    game's player column writes the accented name and "Butler III".
    """
    plain = unicodedata.normalize("NFKD", str(name).translate(UNDECOMPOSABLE))
    plain = plain.encode("ascii", "ignore").decode("ascii")
    return NAME_SUFFIX_RE.sub("", plain.strip().lower())


class PasserIndex:
    """Who a name in an assist description can be, over the WHOLE game.

    Built before the plays are walked, so a passer is recognised even when his
    first appearance in the play-by-play is the assist itself. The box score
    adds the rare passer who is named on no row of his own, and everyone's
    first name: where two teammates share a surname the description prefixes
    it ("G. Antetokounmpo", "Ja. Green").
    """

    def __init__(self, records, context):
        self._ids = defaultdict(list)       # (team, surname key) -> person ids
        self._first = {}                    # person id -> first name, or its initial
        self._given = defaultdict(int)      # person id -> assists credited so far
        self._final = context.final_assists
        for row in records:
            name, team = row.get("playerName"), row.get("teamTricode")
            if not (isinstance(name, str) and name and isinstance(team, str)):
                continue
            pid = _person_id(row)
            self._add(team, name, pid)
            initialled = row.get("playerNameI")
            if isinstance(initialled, str) and ". " in initialled:
                self._first.setdefault(pid, name_key(initialled.split(". ", 1)[0]))
        for team, passers in context.passers.items():
            for pid, first, surname in passers:
                self._add(team, surname, pid)
                self._first[pid] = name_key(first)

    def _add(self, team, surname, pid):
        ids = self._ids[(team, name_key(surname))]
        if pid not in ids:
            ids.append(pid)

    def candidates(self, team, described):
        """Person ids on `team` that the described name can refer to."""
        prefix, surname = DESCRIBED_NAME_RE.match(described).groups()
        ids = self._ids.get((team, name_key(surname)), [])
        if prefix is None or len(ids) < 2:
            return ids
        wanted = prefix.lower()
        matching = [c for c in ids if self._starts_alike(self._first.get(c, ""), wanted)]
        return matching or ids

    @staticmethod
    def _starts_alike(first, prefix):
        # Either may be the shorter: a full first name against "Ja", or a bare
        # initial (all the play-by-play alone gives) against "Ja".
        return bool(first) and (first.startswith(prefix) or prefix.startswith(first))

    def passer_of(self, team, described, running_total, scorer_id):
        """The person who made this pass, or None; counts it against him.

        The scorer is never a candidate, even when he shares the surname.
        Among several, the description's number is the passer's running total
        for the game, so he stands one below it; of those, the one with the
        most box-score assists still unaccounted for - which keeps every
        player's total equal to the box score's where the feed leaves the
        choice open.
        """
        candidates = [c for c in self.candidates(team, described) if c != scorer_id]
        if not candidates:
            return None
        if len(candidates) > 1:
            one_below = [c for c in candidates if self._given[c] == running_total - 1]
            candidates = sorted(one_below or candidates,
                                key=lambda c: (self._given[c] - self._final.get(c, 0), str(c)))
        passer = candidates[0]
        self._given[passer] += 1
        return passer


# ---------------------------------------------------------------------------
#  One game
# ---------------------------------------------------------------------------
def compute_game_value(pbp, use_leverage=True, context=NO_CONTEXT, raw=False):
    """Per-player value for one game from its play-by-play frame.

    Returns {person_id: {name, team, value, <components>, events}}, plus
    RAW_FIELDS when `raw` is set. Rows without a person id (pre-2019 feeds)
    fall back to 'name:<surname>' keys.
    """
    if pbp is None or pbp.empty:
        return {}
    df = pbp.sort_values("actionNumber") if "actionNumber" in pbp.columns else pbp
    df = df.copy()
    df["_t"] = df["clock"].map(clock_seconds)
    df["_h"] = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0.0)
    df["_a"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0.0)
    df["_p"] = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)
    records = df.to_dict("records")

    passers = PasserIndex(records, context)
    players = {}

    def entry(pid, name, team):
        e = players.get(pid)
        if e is None:
            e = players[pid] = {"name": name, "team": team, "value": 0.0, "events": 0,
                                "weight_sum": 0.0, **{c: 0.0 for c in COMPONENTS},
                                **{r: 0.0 for r in RAW_FIELDS}}
        if team and not e["team"]:
            e["team"] = team
        return e

    last_miss_team = None
    for row in records:
        kind = row.get("actionType")
        desc = row.get("description") if isinstance(row.get("description"), str) else ""
        team = row.get("teamTricode") if isinstance(row.get("teamTricode"), str) else None
        is_miss = kind == "Missed Shot" or (kind == "Free Throw" and "MISS" in desc)
        name = row.get("playerName")
        if not isinstance(name, str) or not name:
            if is_miss:
                last_miss_team = team
            continue
        priced = price(row, last_miss_team)
        if is_miss:
            last_miss_team = team
        if priced is None:
            continue
        component, value = priced
        w = leverage(row["_p"], row["_t"], row["_h"] - row["_a"]) if use_leverage else 1.0
        credit = value * w
        scorer_id = _person_id(row)
        e = entry(scorer_id, name, team)
        e["events"] += 1
        e["weight_sum"] += w
        e["core"] += credit

        if component == "scoring" and team:
            _credit_passer(desc, team, credit, w, scorer_id, e, passers, entry)
        e[component] += credit
        e["value"] += credit

    _add_tracked_passing(players, context, use_leverage, entry)
    return {pid: _public(e, raw) for pid, e in players.items()}


def _credit_passer(desc, team, basket_credit, weight, scorer_id, scorer, passers, entry):
    """Pay the passer of an assisted basket his share, on top of the scorer's."""
    match = ASSIST_RE.search(desc)
    if match is None:
        return
    described, running_total = match.group(1), int(match.group(2))
    passer = passers.passer_of(team, described, running_total, scorer_id)
    if passer is None:
        return
    earned = ASSIST_SHARE * basket_credit
    p = entry(passer, DESCRIBED_NAME_RE.match(described).group(2), team)
    p["assists"] += earned
    p["value"] += earned
    p["assist_base"] += basket_credit
    p["assist_weight"] += weight
    p["assists_n"] += 1
    scorer["assisted_scoring"] += basket_credit


def _add_tracked_passing(players, context, use_leverage, entry):
    """Secondary and free-throw assists, which only the tracking table records.

    They are game totals with no clock, so in the leverage-weighted run they
    take the player's own mean leverage for the game.
    """
    for pid, tracked in context.tracking.items():
        e = entry(pid, tracked.surname, tracked.team)
        mean_weight = e["weight_sum"] / e["events"] if (use_leverage and e["events"]) else 1.0
        added = (SECONDARY_ASSIST * tracked.secondary + FT_ASSIST * tracked.free_throw) * mean_weight
        e["assists"] += added
        e["value"] += added
        e["leverage_mean"] = mean_weight
        e["secondary_assists"] = tracked.secondary
        e["ft_assists"] = tracked.free_throw


def _public(e, raw):
    out = {"name": e["name"], "team": e["team"], "value": e["value"], "events": e["events"],
           **{c: e[c] for c in COMPONENTS}}
    if raw:
        if not e["leverage_mean"]:
            e["leverage_mean"] = e["weight_sum"] / e["events"] if e["events"] else 1.0
        out.update({r: e[r] for r in RAW_FIELDS})
    return out


# ---------------------------------------------------------------------------
#  Seasons and the cache
# ---------------------------------------------------------------------------
def season_game_ids(season):
    games = pd.read_pickle(GAMES_PATH)
    return sorted(games.loc[games["season"] == season, "game_id"].astype(str).str.zfill(10))


def pbp_path(season, game_id):
    return os.path.join(DATA_DIR, season, game_id, "play_by_play", f"{game_id}pbp.csv")


def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def run_seasons(seasons, cache, verbose=True, cache_path=CACHE_PATH, use_leverage=True,
                refresh=False, raw=False):
    """Fill `cache` for the seasons and write it after each one.

    `refresh` recomputes games already cached - needed whenever the pricing
    changes, because a cached game does not say which pricing produced it.
    """
    t0 = time.time()
    for season in seasons:
        ids = season_game_ids(season)
        todo = ids if refresh else [g for g in ids if g not in cache]
        contexts = load_contexts(todo)
        missing = 0
        for n, gid in enumerate(todo, 1):
            path = pbp_path(season, gid)
            if not (os.path.exists(path) and os.path.getsize(path) > 64):
                missing += 1
                continue
            try:
                cache[gid] = compute_game_value(pd.read_csv(path, low_memory=False), use_leverage,
                                                contexts.get(gid, NO_CONTEXT), raw)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                missing += 1
            if verbose and n % 200 == 0:
                print(f"    {season}: {n}/{len(todo)} ({time.time() - t0:.0f} sn)", flush=True)
        if verbose:
            print(f"  {season}: {len(ids)} mac, {len(todo)} hesaplandi, {missing} dosyasiz "
                  f"({time.time() - t0:.0f} sn)", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    return cache


def player_frame(cache, game_ids, fields=COMPONENTS):
    rows = []
    for gid in game_ids:
        for pid, e in (cache.get(gid) or {}).items():
            rows.append({"game_id": gid, "player": pid, "name": e["name"], "team": e["team"],
                         "value": e["value"], **{c: e[c] for c in fields}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Evaluation against the impact score and on-court plus-minus
# ---------------------------------------------------------------------------
def box_minutes(game_ids):
    """Box-score lines for the games, with minutes parsed."""
    box = pd.read_pickle(BOX_PLAYER_PATH)
    box = box[box["game_id"].isin(game_ids)][["game_id", "player_id", "player_name", "min",
                                              "plus_minus", "pts", "reb", "ast", "to", "fga",
                                              "fta"]]
    parts = box["min"].astype("string").str.extract(r"^(\d+)(?:\.\d+)?(?::(\d+))?")
    box["minutes"] = (pd.to_numeric(parts[0], errors="coerce").fillna(0)
                      + pd.to_numeric(parts[1], errors="coerce").fillna(0) / 60.0)
    return box.rename(columns={"player_id": "player"})


def evaluate(season, cache, min_games=40, min_minutes=24.0):
    ids = season_game_ids(season)
    value = player_frame(cache, ids)
    value = value[~value["player"].astype(str).str.startswith("name:")].copy()
    value["player"] = value["player"].astype(int)

    with open(IMPACT_CACHE, "rb") as f:
        impact_cache = pickle.load(f)
    imp_rows = []
    for gid in ids:
        e = impact_cache.get(gid)
        for v in ((e or {}).get("players") or {}).values():
            if isinstance(v, dict) and v.get("person_id") is not None:
                imp_rows.append((gid, int(v["person_id"]), float(v.get("impact", 0.0))))
    impact = pd.DataFrame(imp_rows, columns=["game_id", "player", "impact"])

    m = (box_minutes(ids).merge(value, on=["game_id", "player"], how="inner")
         .merge(impact, on=["game_id", "player"], how="left"))
    m = m[m["minutes"] > 0]
    agg = (m.groupby("player_name")
           .agg(g=("value", "size"), minutes=("minutes", "mean"), value=("value", "mean"),
                impact=("impact", "mean"), pm=("plus_minus", "mean"), pts=("pts", "mean"),
                reb=("reb", "mean"), ast=("ast", "mean"), to=("to", "mean"),
                fga=("fga", "mean"), fta=("fta", "mean"),
                **{c: (c, "mean") for c in COMPONENTS}))
    agg = agg[(agg["g"] >= min_games) & (agg["minutes"] >= min_minutes)].copy()
    agg["ts"] = agg["pts"] / (2 * (agg["fga"] + 0.44 * agg["fta"]))
    for col in ("value", "impact", "pm"):
        agg[f"{col}36"] = agg[col] / agg["minutes"] * 36
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    agg["empty_index"] = z(agg["impact36"]) - z(agg["value36"])
    return agg


def print_evaluation(agg, season):
    print(f"\n=== {season}: {len(agg)} duzenli oyuncu (>=40 mac, >=24 dk) ===")
    print(f"  on-court +/- (36 dk) ile korelasyon:  impact {agg['impact36'].corr(agg['pm36']):.3f}"
          f"   value {agg['value36'].corr(agg['pm36']):.3f}")
    print(f"  mac basina:                           impact {agg['impact'].corr(agg['pm']):.3f}"
          f"   value {agg['value'].corr(agg['pm']):.3f}")
    cols = ["g", "minutes", "value", "impact", "pm", "pts", "reb", "ast", "to", "ts"]
    print("\n  DEGER'e gore ilk 15:")
    print(agg.sort_values("value", ascending=False).head(15)[cols].round(2).to_string())
    print("\n  en 'bos' istatistik (impact yuksek, deger dusuk):")
    print(agg.sort_values("empty_index", ascending=False).head(10)[cols + ["empty_index"]].round(2).to_string())
    print("\n  en 'sessiz' deger (deger yuksek, impact dusuk):")
    print(agg.sort_values("empty_index").head(8)[cols + ["empty_index"]].round(2).to_string())
    for name in ("Russell Westbrook", "Stephen Curry", "James Harden", "Kevin Durant",
                 "Andre Drummond", "Anthony Davis", "LeBron James"):
        if name in agg.index:
            r = agg.loc[name]
            comps = "  ".join(f"{c} {r[c]:+.1f}" for c in COMPONENTS)
            print(f"\n  {name}: value {r['value']:.1f}/mac (impact {r['impact']:.1f}), "
                  f"+/- {r['pm']:+.1f}, TS {r['ts']:.3f}\n    {comps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="veri setindeki her sezon")
    parser.add_argument("--eval", action="store_true", help="impact ve +/- ile karsilastir")
    parser.add_argument("--no-leverage", action="store_true",
                        help="oyun agirligi yok (her oyun 1.0) - modelin takim feature'lari icin")
    parser.add_argument("--cache", default=CACHE_PATH, help="onbellek dosyasi")
    parser.add_argument("--refresh", action="store_true",
                        help="onbellekteki maclari da yeniden hesapla (fiyatlama degistiyse sart)")
    parser.add_argument("--raw", action="store_true",
                        help="asist kuralini yeniden kurmaya yarayan ham alanlari da sakla")
    args = parser.parse_args()
    seasons = args.seasons or []
    if args.all:
        seasons = sorted(pd.read_pickle(GAMES_PATH)["season"].unique())
    if not seasons:
        seasons = ["2017_2018"]
    cache = load_cache(args.cache)
    cache = run_seasons(seasons, cache, cache_path=args.cache,
                        use_leverage=not args.no_leverage, refresh=args.refresh, raw=args.raw)
    print(f"onbellek: {len(cache):,} mac -> {args.cache}  (fiyatlama {PRICING_VERSION})")
    if args.eval:
        for season in seasons:
            print_evaluation(evaluate(season, cache), season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
