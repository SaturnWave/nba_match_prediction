"""What does ESPN's public (undocumented) API expose for NBA odds? Read-only probe."""
import json, sys, requests
out = sys.argv[1]
H = {"User-Agent": "Mozilla/5.0"}
def get(url):
    r = requests.get(url, headers=H, timeout=30)
    print("GET", url.split("?")[0][-90:], r.status_code, flush=True)
    return r.json() if r.ok else None
for day in ("20261024", "20260115"):
    sb = get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day}")
    if not sb or not sb.get("events"):
        continue
    event = sb["events"][0]
    comp = event["competitions"][0]
    print(day, event["id"], event["name"], "| scoreboard odds:",
          json.dumps(comp.get("odds", [])[:1])[:600], flush=True)
    odds = get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event['id']}"
               f"/competitions/{event['id']}/odds")
    if odds:
        json.dump(odds, open(f"{out}/odds_{day}.json", "w"), indent=1)
        for item in odds.get("items", []):
            prov = item.get("provider", {})
            print("  provider:", prov.get("id"), prov.get("name"), "| keys:", sorted(item.keys())[:40], flush=True)
            for side in ("homeTeamOdds",):
                block = item.get(side, {})
                print("   ", side, "keys:", sorted(block.keys()), flush=True)
                for phase in ("open", "close", "current"):
                    if phase in block:
                        print("     ", phase, json.dumps(block[phase])[:300], flush=True)
