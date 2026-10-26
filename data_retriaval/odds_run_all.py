"""
Run the whole odds chain in order; each step resumes where the last run stopped.

  1. odds_sbr.py          Sportsbook Reviews archive, 2017-18 .. 2023-01-16
  2. odds_espn_events.py  ESPN event id for every game date from 2023-01-01
  3. odds_espn_lines.py   ESPN lines per event, median over books
  4. odds_build.py        both matched to NBA game ids -> odds_data/odds_history.csv

Run:  py data_retriaval/odds_run_all.py
"""
import time

import odds_build
import odds_espn_events
import odds_espn_lines
import odds_sbr

STEPS = (("Sportsbook Reviews arsivi", odds_sbr.main), ("ESPN mac listesi", odds_espn_events.main),
         ("ESPN oranlari", odds_espn_lines.main), ("birlestirme", odds_build.main))


def main():
    for name, step in STEPS:
        started = time.time()
        print(f"== {name}", flush=True)
        step()
        print(f"   ({(time.time() - started) / 60:.1f} dk)", flush=True)


if __name__ == "__main__":
    main()
