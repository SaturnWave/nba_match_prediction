"""ESPN odds provider/coverage for older games (before the Dec 1, 2025 DraftKings switch). Read-only."""
import requests
H = {"User-Agent": "Mozilla/5.0"}
for day in ("20251105", "20250115", "20240115", "20221115", "20191115", "20171115"):
    sb = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day}",
                      headers=H, timeout=30).json()
    events = sb.get("events", [])
    if not events:
        print(day, "no events"); continue
    eid = events[0]["id"]
    r = requests.get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}"
                     f"/competitions/{eid}/odds", headers=H, timeout=30)
    items = r.json().get("items", []) if r.ok else []
    summary = []
    for it in items:
        p = it.get("provider", {})
        has = [k for k in ("open", "close", "current") if k in it.get("homeTeamOdds", {})]
        summary.append(f"{p.get('id')}:{p.get('name')} spread={it.get('spread')} total={it.get('overUnder')} phases={has}")
    print(day, eid, events[0]["shortName"], "HTTP", r.status_code, "| providers:", summary if summary else "none")
