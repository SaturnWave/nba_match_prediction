"""
What the odds scripts share: where files go, ESPN's team codes, and one polite way to fetch.

Every fetched response is cached on disk first (odds_data/raw/, ignored by git),
so a run that stops can be started again and only fetches what is missing.
Requests go out one at a time with a pause between them. A 403 or 429 means the
site does not want more requests: the run stops there instead of retrying or
working around it.
"""
import json
import os
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_DIR = os.path.join(PROJECT_ROOT, "odds_data")
RAW_DIR = os.path.join(ODDS_DIR, "raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba_match_prediction personal research)"}
TIMEOUT = 30
DELAY_SECONDS = 1.0
RETRY_WAITS = (5, 15, 45)          # timeouts and server errors only
REFUSED = (403, 429)

# ESPN abbreviations that differ from the NBA's tricodes
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


class Refused(RuntimeError):
    """The site answered 403 or 429; the run stops rather than pressing on."""


def nba_code(espn_abbreviation):
    return ESPN_TO_NBA.get(espn_abbreviation, espn_abbreviation)


def cached_json(url, cache_path, params=None):
    """The JSON at `url`, from `cache_path` when it was fetched before."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    for attempt, wait in enumerate((0, *RETRY_WAITS)):
        time.sleep(wait or DELAY_SECONDS)
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(f"{url}: {len(RETRY_WAITS) + 1} denemede de yanit yok") from exc
            continue
        if response.status_code in REFUSED:
            raise Refused(f"{url}: HTTP {response.status_code} - site istekleri reddediyor, durduruldu")
        if response.status_code >= 500 and attempt < len(RETRY_WAITS):
            continue
        response.raise_for_status()
        payload = response.json()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        partial = cache_path + ".part"
        with open(partial, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(partial, cache_path)
        return payload
    raise RuntimeError(f"{url}: sunucu hatasi surdu")
