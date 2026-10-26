import sys, requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
out_dir = sys.argv[1]
for path in ["liveData/boxscore/boxscore_0022600001.json", "liveData/playbyplay/playbyplay_0022600001.json",
             "liveData/scoreboard/todaysScoreboard_00.json", "staticData/scheduleLeagueV2.json",
             "liveData/odds/odds_todaysGames.json"]:
    r = requests.get("https://cdn.nba.com/static/json/" + path, headers=headers, timeout=30)
    print(path, r.status_code, len(r.content), r.headers.get("Last-Modified"), r.headers.get("Cache-Control"))
    if r.ok:
        open(out_dir + "/" + path.split("/")[-1], "wb").write(r.content)
