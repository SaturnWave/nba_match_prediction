import pandas as pd, numpy as np
bp = pd.read_pickle("phonedb_cache/box_player_traditional.pkl")
def to_min(s):
    s = s.astype("string")
    parts = s.str.extract(r"^(\d+)(?::(\d+))?")
    m = pd.to_numeric(parts[0], errors="coerce").fillna(0) + pd.to_numeric(parts[1], errors="coerce").fillna(0) / 60.0
    return m.where(s.notna(), np.nan)
bp["minutes"] = to_min(bp["min"])
print("rows", len(bp), "games", bp.game_id.nunique(), "players w/ minutes>0 share %.3f" % (bp.minutes > 0).mean())
print("min column samples:", bp["min"].dropna().sample(8, random_state=1).tolist())
played = bp[bp.minutes > 0]
per = played.groupby(["game_id", "team_abbreviation"]).size()
print("players with minutes>0 per team-game: mean %.2f sd %.2f min %d max %d" % (per.mean(), per.std(), per.min(), per.max()))
listed = bp.groupby(["game_id", "team_abbreviation"]).size()
print("listed players per team-game (incl DNP): mean %.2f sd %.2f" % (listed.mean(), listed.std()))
# team minutes sum sanity
tm = played.groupby(["game_id", "team_abbreviation"]).minutes.sum()
print("team minutes per game: mean %.1f (240 expected; OT more)" % tm.mean())
# compare with impact cache participation
import pickle
cache = pickle.load(open("game_impact_cache_v4.pkl", "rb"))
gid = "0022400058"
imp_players = {int(v["person_id"]) for v in cache[gid]["players"].values() if isinstance(v, dict) and v.get("person_id") is not None}
box_players = set(bp[(bp.game_id == gid) & (bp.minutes > 0)].player_id)
print(f"game {gid}: impact-cache players {len(imp_players)}, box minutes>0 {len(box_players)}, overlap {len(imp_players & box_players)}")
