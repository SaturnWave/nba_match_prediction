"""Read the phonedb MariaDB schema over a link that keeps breaking.

WHY A PAGED READER
    db_source.py used to issue one unpaged query per table. That was fine
    when the pull took 1.7 seconds. It is not fine now, and the reason is the
    transport rather than the schema:

      * the database lives in a Debian VM on a phone, reached over Tailscale;
      * the direct path runs through the phone's mobile uplink (measured
        2026-09-12: 583 ms round trip, and a DERP relay in Frankfurt whenever
        NAT traversal fails);
      * result sets die in transit. Measured against game_summary on that
        link: 100 rows arrived in 15 s, 1.000 rows was reset mid-transfer,
        3.000 rows arrived once and was reset once, 6.000 and 10.749 always
        reset. The failure is 'Lost connection during query', i.e. the server
        answered and the bytes did not survive the trip.

    A single SELECT over a 5.000.000-row table cannot be made to work here.
    What does work is asking for small ranges and keeping every range that
    arrives, so a reset costs one page instead of the whole pull.

HOW IT READS
    Keyset paging on the primary key: `WHERE key > last ORDER BY key LIMIT n`.
    Every table in this schema has a single-column primary key (`row_id` on
    the raw tables, `game_id` on the conformed ones) and an index on game_id,
    so each page is an index range scan rather than a growing OFFSET walk.

    The page size adapts. It halves on a reset and doubles after three clean
    pages, because the link's capacity changes by the minute - the same query
    that needed 15 s for 100 rows answered a 3.000-row page a minute later.
    A fixed page size is either too slow when the link is good or too big when
    it is not.

    Progress is written to disk as it arrives. A pull killed halfway resumes
    from the last page it kept instead of starting over, which for
    play_by_play (5 million rows) is the difference between an interrupted
    transfer and a lost evening.

WHAT IT RETURNS
    DataFrames with the database's own column names - no renaming. Renaming
    to the UPPER_SNAKE names FeatureEngineer expects stays with the callers
    that feed it, so this module only has to know the database.

WHO READS THROUGH IT
    db_source.load_master_frame, availability_features.load_absences and
    player_source.load_player_games all call load(), and each was checked
    against the output of the SQL it replaced: identical frames, column order
    and dtypes included. The play-by-play readers (clutch_features,
    db_source.load_pbp) still query the phone directly, because that table is
    deliberately left on the phone rather than cached.

CACHING
    A pulled table is written to phonedb_cache/<table>.pkl and read from there
    on the next call. The cache directory is deliberately NOT one of the paths
    auto_push.ps1 commits (nba_data, output, models, game_ids): these files are
    a local copy of someone else's database, they can be several hundred
    megabytes, and pushing them to GitHub every five minutes would be wrong on
    both counts.

Run:  py prediction_engines/phonedb_source.py --list
      py prediction_engines/phonedb_source.py --pull game_summary training_games
      py prediction_engines/phonedb_source.py --pull-all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass

import pandas as pd
import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:                 # so `py prediction_engines/...` and
    sys.path.insert(0, HERE)             # `import phonedb_source` both work
from db_source import connect, load_config, probe   # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "phonedb_cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")

# Paging bounds. The floor is low because the link genuinely does fail at
# 1.000 rows; the ceiling is not high because a page that dies has to be
# fetched again, and a big page dies more often than a small one.
MIN_PAGE_ROWS = 50
MAX_PAGE_ROWS = 4000
CLEAN_PAGES_BEFORE_GROWING = 3

# A reset costs one page, so retrying is cheap - but a server that is actually
# down should not be retried for an hour.
PAGE_ATTEMPTS = 6
PAGE_BACKOFF_SECONDS = (2, 5, 10, 20, 30)

# How often the partial pull is flushed to disk. Counted in rows rather
# than pages because the page size moves with the link: "every 20 pages"
# means 1.000 rows on a bad link and 80.000 on a good one, and it is the
# rows that have to be fetched again after an interruption.
CHECKPOINT_EVERY_ROWS = 25000


@dataclass(frozen=True)
class TableSpec:
    """One readable object in phonedb.

    key         single-column PRIMARY KEY, which is also the paging cursor
    key_is_text game_id is char(10) and orders as text; row_id is an integer
    has_season  whether the table carries `season` itself, or needs a join to
                games to be filtered by one
    known_rows  row count measured 2026-09-12, used only to show progress -
                the pull does not depend on it being current
    """
    name: str
    key: str
    key_is_text: bool
    has_season: bool
    known_rows: int
    layer: str


# The schema as it stands: 15 tables in three layers. team_form is a view over
# these and is excluded on purpose - it recomputes window functions over every
# row on each call, which on this link took 3,6 s for three rows and dropped
# the connection when asked for a season. Anything it offers can be computed
# locally from game_summary once that table is cached.
TABLES = {
    spec.name: spec for spec in (
        # conformed layer - one row per game
        TableSpec("games",                  "game_id", True,  True,     10749, "conformed"),
        TableSpec("game_dates",             "game_id", True,  False,    10749, "conformed"),
        TableSpec("game_summary",           "game_id", True,  True,     10749, "conformed"),
        # feature layer - the surface meant for training
        TableSpec("training_games",         "game_id", True,  True,     10749, "feature"),
        # raw layer - one row per team-game
        TableSpec("box_team_traditional",   "row_id",  False, False,    21498, "raw"),
        TableSpec("box_team_advanced",      "row_id",  False, False,    21498, "raw"),
        TableSpec("box_team_defensive",     "row_id",  False, False,    21450, "raw"),
        TableSpec("team_tracking",          "row_id",  False, False,    21498, "raw"),
        # raw layer - one row per player-game
        TableSpec("box_player_traditional", "row_id",  False, False,   277127, "raw"),
        TableSpec("box_player_advanced",    "row_id",  False, False,   277402, "raw"),
        TableSpec("box_player_defensive",   "row_id",  False, False,   228245, "raw"),
        TableSpec("player_tracking",        "row_id",  False, False,   277127, "raw"),
        # raw layer - the big ones
        TableSpec("box_matchups",           "row_id",  False, False,  1905721, "raw"),
        TableSpec("pbp_score",              "row_id",  False, False,  5396877, "raw"),
        TableSpec("play_by_play",           "row_id",  False, False,  5046447, "raw"),
    )
}

LINK_ERRORS = (pymysql.err.OperationalError, pymysql.err.InterfaceError)


class _Link:
    """One connection to phonedb, reopened whenever the link drops.

    Opening a connection costs several round trips, and at 583 ms each that is
    most of a second - too much to pay per page. So the connection is held and
    only replaced after a failure, which is the only time it is known bad.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._conn = None

    def query(self, sql, params):
        """(column_names, rows). Raises LINK_ERRORS for the caller to handle."""
        if self._conn is None:
            self._conn = connect(self._cfg)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return names, rows

    def drop(self):
        """Throw the connection away; the next query opens a fresh one."""
        if self._conn is not None:
            try:
                self._conn.close()
            except LINK_ERRORS:
                pass          # it is already broken; that is why we are here
            self._conn = None

    def close(self):
        self.drop()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _first_cursor(spec):
    """A key value that sorts before every real one."""
    return "" if spec.key_is_text else -1


def _page_query(spec, columns, seasons):
    """SQL and the leading params for one page, minus the cursor and limit.

    Tables that do not carry `season` are filtered through a join to games
    rather than by collecting ten thousand game_ids into an IN list - the IN
    list would be larger than the page it filters.
    """
    selected = ", ".join(f"t.`{c}`" for c in columns) if columns else "t.*"
    sql = f"SELECT {selected} FROM `{spec.name}` t"
    params = []
    if seasons and not spec.has_season:
        placeholders = ",".join(["%s"] * len(seasons))
        sql += f" JOIN games g ON g.game_id = t.game_id AND g.season IN ({placeholders})"
        params.extend(seasons)
    sql += f" WHERE t.`{spec.key}` > %s"
    if seasons and spec.has_season:
        placeholders = ",".join(["%s"] * len(seasons))
        sql += f" AND t.`season` IN ({placeholders})"
    sql += f" ORDER BY t.`{spec.key}` LIMIT %s"
    return sql, params


def _pull_key(table, seasons, columns):
    """What identifies one pull's checkpoint files.

    A checkpoint written for one question must never be resumed by another.
    Parts left by an interrupted one-season pull, picked up by a later pull of
    the whole table, would hand back a "whole" table that silently holds one
    season. So the key carries the season filter and the column list; a plain
    full pull keeps the bare table name.
    """
    key = _cache_name(table, seasons)[:-len(".pkl")]
    if columns:
        digest = hashlib.sha1(",".join(columns).encode("utf-8")).hexdigest()[:8]
        key += f"__cols_{digest}"
    return key


def _parts_dir(pull_key):
    return os.path.join(CACHE_DIR, f"{pull_key}.parts")


def _write_part(pull_key, index, frame, cursor):
    """One checkpoint: the rows since the last one, already framed.

    Parts hold DataFrames rather than the raw tuples they arrived as. Five
    million rows kept as Python tuples is several gigabytes of interpreter
    objects; the same rows in a frame are a fraction of that, and converting
    per checkpoint lets the tuples be freed as the pull goes rather than all
    at the end.
    """
    os.makedirs(_parts_dir(pull_key), exist_ok=True)
    path = os.path.join(_parts_dir(pull_key), f"{index:05d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"frame": frame, "cursor": cursor}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def _read_parts(pull_key):
    """(frames, cursor, next_index) from a previously interrupted pull.

    cursor is None when there is nothing to resume from, which is a different
    thing from a cursor that happens to sort first.
    """
    directory = _parts_dir(pull_key)
    if not os.path.isdir(directory):
        return [], None, 0
    names = sorted(n for n in os.listdir(directory) if n.endswith(".pkl"))
    frames, cursor = [], None
    for name in names:
        with open(os.path.join(directory, name), "rb") as f:
            part = pickle.load(f)
        frames.append(part["frame"])
        cursor = part["cursor"]
    return frames, cursor, len(names)


def _clear_parts(pull_key):
    directory = _parts_dir(pull_key)
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        os.remove(os.path.join(directory, name))
    os.rmdir(directory)


def _run_page(link, sql, params, table, verbose):
    """One page, retried across resets. (names, rows, rows_asked, hit_reset).

    Returns None when the link is down rather than merely slow - the caller
    then saves what it has instead of losing it to an exception.

    rows_asked comes back because a reset lowers it: the caller has to know
    how many rows this page actually asked for to tell a short final page from
    a full one, and whether the link misbehaved before it decides to ask for
    more next time.
    """
    rows_asked = params[-1]
    hit_reset = False
    for attempt in range(PAGE_ATTEMPTS):
        try:
            names, rows = link.query(sql, params)
            return names, rows, rows_asked, hit_reset
        except LINK_ERRORS as exc:
            link.drop()
            hit_reset = True
            rows_asked = max(MIN_PAGE_ROWS, rows_asked // 2)
            params[-1] = rows_asked
            wait = PAGE_BACKOFF_SECONDS[min(attempt, len(PAGE_BACKOFF_SECONDS) - 1)]
            if verbose:
                print(f"    {table}: sayfa koptu, {rows_asked} satira dusuldu, "
                      f"{wait} sn sonra tekrar ({attempt + 1}/{PAGE_ATTEMPTS}): "
                      f"{str(exc)[:50]}", flush=True)
            time.sleep(wait)
    return None


def _adjust_page_size(page_rows, clean_pages):
    """Grow the page only after the link has proved it can carry the current one."""
    if clean_pages >= CLEAN_PAGES_BEFORE_GROWING and page_rows < MAX_PAGE_ROWS:
        return min(MAX_PAGE_ROWS, page_rows * 2), 0
    return page_rows, clean_pages


def _report_progress(table, have, expected, started, page_rows):
    elapsed = max(time.time() - started, 0.001)
    rate = have / elapsed
    if expected and have < expected and rate > 0:
        eta = (expected - have) / rate
        tail = f", kalan ~{eta / 60:.0f} dk"
    else:
        tail = ""
    pct = f"{100 * have / expected:.0f}%" if expected else "?"
    print(f"    {table}: {have:,}/{expected:,} ({pct}) "
          f"{rate:.0f} satir/sn, sayfa {page_rows}{tail}", flush=True)


def fetch_table(table, columns=None, seasons=None, cfg=None, resume=True,
                verbose=True):
    """Pull one table page by page, keeping every page that arrives.

    The page size starts in the middle of what the link has been observed to
    carry and moves with it: halved after a reset, doubled after three clean
    pages. Pages already received are flushed to disk periodically, so killing
    this and running it again continues from there rather than from nothing.

    Checkpoints belong to one pull - this table, these seasons, these columns -
    and are left in place when the pull completes: the caller clears them
    once the result is safely stored, which is what load() does.
    """
    spec = TABLES[table]
    cfg = cfg or load_config()
    # The key is the paging cursor. A column list that leaves it out would
    # leave the pull no way to ask for the page after the first.
    if columns and spec.key not in columns:
        columns = [spec.key, *columns]
    pull_key = _pull_key(table, seasons, columns)

    if resume:
        frames, cursor, part_index = _read_parts(pull_key)
    else:
        _clear_parts(pull_key)
        frames, cursor, part_index = [], None, 0
    if cursor is not None and verbose:
        print(f"  {table}: {sum(len(f) for f in frames):,} satir onceki "
              f"cekimden devam ediyor", flush=True)
    if cursor is None:
        cursor = _first_cursor(spec)

    sql, base_params = _page_query(spec, columns, seasons)
    page_rows, clean_pages = 500, 0
    pending, columns_seen = [], None
    started = time.time()

    with _Link(cfg) as link:
        while True:
            params = list(base_params)
            params.append(cursor)
            if seasons and spec.has_season:
                params.extend(seasons)
            params.append(page_rows)

            page = _run_page(link, sql, params, table, verbose)
            if page is None:
                part_index = _flush(pull_key, part_index, columns_seen, pending,
                                    frames, cursor)
                raise RuntimeError(
                    f"{table}: baglanti {PAGE_ATTEMPTS} denemede kurulamadi. "
                    f"{sum(len(f) for f in frames):,} satir diske yazildi; "
                    f"ayni komut kaldigi yerden devam eder.")

            names, batch, rows_asked, hit_reset = page
            if not batch:
                break
            columns_seen = columns_seen or names
            pending.extend(batch)
            cursor = batch[-1][names.index(spec.key)]

            if len(pending) >= CHECKPOINT_EVERY_ROWS:
                part_index = _flush(pull_key, part_index, columns_seen, pending,
                                    frames, cursor)
                pending = []
            if verbose:
                have = sum(len(f) for f in frames) + len(pending)
                _report_progress(table, have, spec.known_rows, started, rows_asked)

            # Short page means the table ran out - but only against the size
            # this page actually asked for. Comparing against a size the next
            # page will use would end the pull early and silently.
            if len(batch) < rows_asked:
                break

            clean_pages = 0 if hit_reset else clean_pages + 1
            page_rows, clean_pages = _adjust_page_size(rows_asked, clean_pages)

    if pending:
        _flush(pull_key, part_index, columns_seen, pending, frames, cursor)
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=columns_seen))
    if verbose:
        elapsed = max(time.time() - started, 0.001)
        print(f"  {table}: {len(frame):,} satir, {elapsed:.0f} sn, "
              f"{len(frame) / elapsed:.0f} satir/sn", flush=True)
    return frame


def _flush(pull_key, part_index, columns, pending, frames, cursor):
    """Turn the rows received since the last checkpoint into a part on disk.

    Returns the next part index. Writing nothing when there is nothing keeps
    an interrupted pull from leaving an empty part behind that later reads
    back as a zero-row resume point.
    """
    if not pending:
        return part_index
    frame = pd.DataFrame.from_records(pending, columns=columns)
    frames.append(frame)
    _write_part(pull_key, part_index, frame, cursor)
    return part_index + 1


def _cache_name(table, seasons):
    """A season-filtered pull is not the table, and must not be cached as it.

    Serving a nine-season cache to a caller that asked for one season is
    merely wasteful; serving a one-season cache to a caller that asked for all
    of them is a wrong answer that looks like a right one.
    """
    if not seasons:
        return f"{table}.pkl"
    return f"{table}__{'_'.join(sorted(seasons))}.pkl"


def cache_path(table, seasons=None):
    return os.path.join(CACHE_DIR, _cache_name(table, seasons))


def _read_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(entry_key, entry):
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = _read_manifest()
    manifest[entry_key] = entry
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)


def load(table, seasons=None, refresh=False, cfg=None, verbose=True):
    """The table, from the local cache when it is there, from phonedb when not.

    Callers in this repo should go through here rather than fetch_table: on a
    bad day this link needs minutes to hours for a table, so repeating a pull
    that already succeeded is the most expensive mistake available.

    The cache does not notice when phonedb gains rows - a new game day stays
    invisible until the table is pulled again with refresh=True (`--refresh`
    on the command line). That is why a cache hit names the date of the pull
    it is serving: a stale copy should look stale.
    """
    path = cache_path(table, seasons)
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            frame = pickle.load(f)
        if verbose:
            entry = _read_manifest().get(_cache_name(table, seasons), {})
            pulled = entry.get("pulled_at", "tarihi bilinmiyor")
            print(f"  {table}: {len(frame):,} satir yerel cache'ten "
                  f"({pulled} cekimi)", flush=True)
        return frame

    frame = fetch_table(table, seasons=seasons, cfg=cfg, verbose=verbose)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
    _clear_parts(_pull_key(table, seasons, None))
    _write_manifest(_cache_name(table, seasons), {
        "table": table,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "seasons": sorted(seasons) if seasons else "all",
        "pulled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": (cfg or load_config())["NBA_DB_HOST"],
    })
    return frame


def cache_state():
    """What is on disk right now, interrupted pulls included."""
    manifest = _read_manifest()
    state = []
    for name, spec in TABLES.items():
        path = cache_path(name)
        entry = manifest.get(_cache_name(name, None))
        parts = _parts_dir(name)
        state.append({
            "table": name,
            "layer": spec.layer,
            "known_rows": spec.known_rows,
            "cached_rows": entry["rows"] if entry else None,
            "pulled_at": entry["pulled_at"] if entry else None,
            "size_mb": (round(os.path.getsize(path) / 1048576, 1)
                        if os.path.exists(path) else None),
            "partial_files": (len(os.listdir(parts)) if os.path.isdir(parts) else 0),
        })
    return state


def _print_state():
    print(f"phonedb: {load_config()['NBA_DB_HOST']}  -  {probe()[1]}\n")
    header = (f"{'tablo':<26}{'katman':<11}{'DB satir':>10}{'cache':>12}"
              f"{'MB':>7}  cekim")
    print(header)
    print("-" * len(header))
    for row in cache_state():
        cached = f"{row['cached_rows']:,}" if row["cached_rows"] is not None else "-"
        size = f"{row['size_mb']}" if row["size_mb"] is not None else "-"
        if row["pulled_at"]:
            when = row["pulled_at"]
        elif row["partial_files"]:
            when = f"yarim: {row['partial_files']} parca"
        else:
            when = "-"
        print(f"{row['table']:<26}{row['layer']:<11}{row['known_rows']:>10,}"
              f"{cached:>12}{size:>7}  {when}")


# Millions of rows each. At the throughput this link has shown these are a
# multi-hour pull on a bad day, so --pull-all leaves them out; they have to be
# asked for by name, deliberately.
HEAVY_TABLES = ("box_matchups", "pbp_score", "play_by_play")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="phonedb'den tablo cek ve yerel cache'e yaz.")
    parser.add_argument("--list", action="store_true",
                        help="tablolari ve cache durumunu goster")
    parser.add_argument("--pull", nargs="+", metavar="TABLO",
                        help="verilen tablolari cek")
    parser.add_argument("--pull-all", action="store_true",
                        help=f"su uc buyuk tablo disinda hepsini cek: "
                             f"{', '.join(HEAVY_TABLES)}")
    parser.add_argument("--seasons", nargs="+", metavar="SEZON",
                        help="orn. 2024_2025 2025_2026")
    parser.add_argument("--refresh", action="store_true",
                        help="cache dolu olsa da yeniden cek")
    args = parser.parse_args(argv)

    if args.list or not (args.pull or args.pull_all):
        _print_state()
        return 0

    tables = args.pull or [t for t in TABLES if t not in HEAVY_TABLES]
    unknown = [t for t in tables if t not in TABLES]
    if unknown:
        parser.error(f"bilinmeyen tablo: {unknown}. --list ile bakin.")

    reachable, detail = probe()
    if not reachable:
        print(f"phonedb yanit vermiyor: {detail}")
        return 1
    print(f"phonedb: {detail}\n")

    for table in tables:
        load(table, seasons=args.seasons, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
