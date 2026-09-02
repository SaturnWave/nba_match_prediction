"""
Build two derived tables in phonedb: per-player game impact, and the
defender-to-shooter pairing that play-by-play carries but never states.

TABLE 1 â€” player_game_impact
    217,002 rows, one per (game, player). Today these live inside an 8.1 MB
    pickle, so every consumer that wants a player's history has to unpickle the
    whole thing and group it in pandas. As a table they join to box scores,
    tracking and matchups directly, and _build_player_history becomes a query.
    This is also the prerequisite for the award work: DPOY needs per-player
    defensive data joined to impact at season level.

TABLE 2 â€” pbp_defensive_event
    A block in play-by-play is its own row naming only the BLOCKER:
        #152  Missed Shot  Eason (HOU)                MISS Eason 2' Running Layup
        #152  (null)       Gilgeous-Alexander (OKC)   Gilgeous-Alexander BLOCK
    The two rows share an action_number â€” they are one event split in two â€” so
    the victim is recoverable by self-joining on (game_id, action_number) and
    taking the row from the other team. Verified on 60 games: 591 of 592 blocks
    are paired with a Missed Shot (the exception is a Heave), 0% of pairs are
    same-team, and the season's block count matches the box score total to
    within one event (11,906 against 11,907).

    Steals pair the same way against a Turnover row.

    This is finer-grained than box_matchups, which reports totals per
    offense/defense pair for a whole game. Here each stop is an event with a
    clock, a period and a shot distance.

LEAKAGE
    Both tables hold per-game FACTS, never trailing aggregates. Anything the
    model consumes has to be aggregated with a `game_date <` filter at query
    time. This project already carries a column correlating 0.746 with the
    outcome (impact_score_diff); a stored "how much did this defender suppress
    this scorer" column, fed directly, would be the same mistake with a
    friendlier name.

Run:  py prediction_engines/db_build_derived.py [--only impact|events] [--dry-run]
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
IMPACT_CACHE = os.path.join(PROJECT_ROOT, "game_impact_cache_v4.pkl")
INSERT_CHUNK = 5000

DDL_PLAYER_IMPACT = """
CREATE TABLE IF NOT EXISTS player_game_impact (
    game_id     CHAR(10)     NOT NULL,
    person_id   BIGINT       NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    team_abbr   VARCHAR(8)   NOT NULL,
    impact      DOUBLE       NOT NULL,
    PRIMARY KEY (game_id, person_id),
    KEY idx_pgi_person (person_id),
    KEY idx_pgi_name (player_name),
    KEY idx_pgi_team_game (team_abbr, game_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_DEFENSIVE_EVENT = """
CREATE TABLE IF NOT EXISTS pbp_defensive_event (
    game_id       CHAR(10)     NOT NULL,
    action_number BIGINT       NOT NULL,
    event_type    VARCHAR(8)   NOT NULL,
    defender      VARCHAR(255) NOT NULL,
    defender_team VARCHAR(8)   NULL,
    offender      VARCHAR(255) NULL,
    offender_team VARCHAR(8)   NULL,
    period        INT          NULL,
    clock         VARCHAR(32)  NULL,
    shot_distance INT          NULL,
    PRIMARY KEY (game_id, action_number, event_type, defender),
    KEY idx_pde_defender (defender),
    KEY idx_pde_offender (offender),
    KEY idx_pde_pair (offender, defender),
    KEY idx_pde_game (game_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# The victim row and the stop row share an action_number: they are one event
# the feed splits in two. Requiring different teams rejects the rare case where
# a single action_number carries two same-side rows.
POPULATE_EVENTS = """
INSERT INTO pbp_defensive_event
    (game_id, action_number, event_type, defender, defender_team,
     offender, offender_team, period, clock, shot_distance)
SELECT
    d.game_id,
    d.action_number,
    CASE WHEN d.description LIKE '%%BLOCK%%' THEN 'BLOCK' ELSE 'STEAL' END,
    d.player_name,
    d.team_tricode,
    o.player_name,
    o.team_tricode,
    d.period,
    d.clock,
    o.shot_distance
FROM play_by_play d
JOIN play_by_play o
  ON  o.game_id       = d.game_id
  AND o.action_number = d.action_number
  AND o.row_id       <> d.row_id
  AND o.team_tricode <> d.team_tricode
  AND o.player_name IS NOT NULL
WHERE d.player_name IS NOT NULL
  AND (d.description LIKE '%%BLOCK%%' OR d.description LIKE '%%STEAL%%')
  AND (   (d.description LIKE '%%BLOCK%%' AND o.action_type IN ('Missed Shot', 'Heave'))
       OR (d.description LIKE '%%STEAL%%' AND o.action_type = 'Turnover'))
ON DUPLICATE KEY UPDATE
    offender      = VALUES(offender),
    offender_team = VALUES(offender_team),
    shot_distance = VALUES(shot_distance)
"""


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def impact_rows(master):
    """(game_id, person_id, name, team, impact) from the v4 cache.

    Two kinds of row are dropped, counted separately because they mean
    different things:
      * no person_id — a pre-V3 play-by-play row the engine could only key by
        surname. It cannot be joined to a box score, so storing it would put an
        unidentifiable row in a table whose whole purpose is joining.
      * team matches neither side — a stale or wrong tricode, which would
        attribute a player to a roster that never fielded them.
    """
    with open(IMPACT_CACHE, "rb") as f:
        cache = pickle.load(f)
    sides = master.set_index("game_id")[["home_team", "away_team"]].to_dict("index")
    rows = []
    no_id = wrong_team = 0
    for gid, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        side = sides.get(gid)
        if side is None:
            continue
        for key, value in (entry.get("players") or {}).items():
            if not isinstance(value, dict):
                no_id += 1
                continue
            person_id = value.get("person_id")
            if person_id is None:
                no_id += 1
                continue
            team = value.get("team")
            if team not in (side["home_team"], side["away_team"]):
                wrong_team += 1
                continue
            name = (value.get("name") or str(key))[:255]
            rows.append((gid, int(person_id), name, team, float(value.get("impact", 0.0))))
    return rows, no_id, wrong_team


def build_impact_table(db, conn, dry_run=False):
    master = db.load_master_frame(verbose=False)
    rows, no_id, wrong_team = impact_rows(master)
    print(f"player_game_impact: {len(rows):,} satir hazir "
          f"({no_id:,} kimliksiz, {wrong_team:,} takim eslesmedi)")
    if dry_run:
        return

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS player_game_impact")
        cur.execute(DDL_PLAYER_IMPACT)
        sql = ("INSERT INTO player_game_impact "
               "(game_id, person_id, player_name, team_abbr, impact) "
               "VALUES (%s, %s, %s, %s, %s)")
        t0 = time.time()
        for start in range(0, len(rows), INSERT_CHUNK):
            cur.executemany(sql, rows[start:start + INSERT_CHUNK])
        conn.commit()
        print(f"  yazildi: {time.time() - t0:.1f} sn")
        cur.execute("SELECT COUNT(*) FROM player_game_impact")
        print(f"  tabloda: {cur.fetchone()[0]:,} satir")


def build_event_table(conn, dry_run=False):
    if dry_run:
        print("pbp_defensive_event: dry-run, DDL ve INSERT atlandi")
        return
    with conn.cursor() as cur:
        cur.execute(DDL_DEFENSIVE_EVENT)
        cur.execute("TRUNCATE TABLE pbp_defensive_event")
        t0 = time.time()
        cur.execute(POPULATE_EVENTS)
        conn.commit()
        print(f"pbp_defensive_event: {cur.rowcount:,} satir, {time.time() - t0:.1f} sn")


def verify(conn):
    print("\n=== DOGRULAMA ===")
    checks = {
        "player_game_impact satir": "SELECT COUNT(*) FROM player_game_impact",
        "  benzersiz oyuncu": "SELECT COUNT(DISTINCT player_name) FROM player_game_impact",
        "pbp_defensive_event satir": "SELECT COUNT(*) FROM pbp_defensive_event",
        "  blok": "SELECT COUNT(*) FROM pbp_defensive_event WHERE event_type='BLOCK'",
        "  steal": "SELECT COUNT(*) FROM pbp_defensive_event WHERE event_type='STEAL'",
        "  hedefi bos": "SELECT COUNT(*) FROM pbp_defensive_event WHERE offender IS NULL",
        "  ayni takim (olmamali)":
            "SELECT COUNT(*) FROM pbp_defensive_event WHERE offender_team = defender_team",
    }
    with conn.cursor() as cur:
        for label, sql in checks.items():
            cur.execute(sql)
            print(f"  {label:28} {cur.fetchone()[0]:,}")

        print("\n  box score ile blok karsilastirmasi (sezon bazli):")
        cur.execute("""
            SELECT g.season,
                   SUM(CASE WHEN e.event_type='BLOCK' THEN 1 ELSE 0 END) AS pbp_blok,
                   (SELECT SUM(b.blk) FROM box_team_traditional b
                    JOIN games g2 ON g2.game_id=b.game_id WHERE g2.season=g.season) AS box_blok
            FROM pbp_defensive_event e JOIN games g ON g.game_id=e.game_id
            GROUP BY g.season ORDER BY g.season
        """)
        for season, pbp_blk, box_blk in cur.fetchall():
            box_blk = box_blk or 0
            delta = pbp_blk - box_blk
            print(f"    {season}  pbp {pbp_blk:>6,}  box {box_blk:>6,}  fark {delta:+,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["impact", "events"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = _load_sibling("db_source")
    conn = db.connect()
    try:
        if args.only in (None, "impact"):
            build_impact_table(db, conn, args.dry_run)
        if args.only in (None, "events"):
            build_event_table(conn, args.dry_run)
        if not args.dry_run:
            verify(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
