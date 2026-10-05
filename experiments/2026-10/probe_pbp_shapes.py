"""What real play-by-play rows look like, and the league rates a generator has to reproduce."""
import collections, sys
import numpy as np, pandas as pd
sys.path.insert(0, "prediction_engines")
import value_engine as ve
season = "2025_2026"
ids = ve.season_game_ids(season)[::6][:200]
c = collections.Counter(); kinds = collections.Counter(); tov = collections.Counter(); fouls = collections.Counter()
examples = collections.defaultdict(list)
for gid in ids:
    d = pd.read_csv(ve.pbp_path(season, gid), low_memory=False).sort_values("actionNumber")
    last_miss = None; prev = None
    for r in d.itertuples(index=False):
        desc = r.description if isinstance(r.description, str) else ""
        kind = r.actionType if isinstance(r.actionType, str) else ""
        sub = r.subType if isinstance(r.subType, str) else ""
        named = isinstance(r.playerName, str) and bool(r.playerName)
        if "STEAL" in desc: kinds[("STEAL", kind, sub)] += 1; c["steal"] += 1
        if "BLOCK" in desc: kinds[("BLOCK", kind, sub)] += 1; c["block"] += 1
        if kind == "Made Shot":
            v = int(r.shotValue); c[f"made{v}"] += 1
            if " AST)" in desc: c[f"ast{v}"] += 1
            if len(examples["made"]) < 3: examples["made"].append(desc)
        elif kind == "Missed Shot":
            v = int(r.shotValue); c[f"miss{v}"] += 1
            if len(examples["miss"]) < 3: examples["miss"].append(desc)
        elif kind == "Free Throw":
            m = ve.FREE_THROW_RE.search(desc); k, n = m.groups()
            c[f"ft_{k or 'reg'}_{n or '-'}"] += 1
            if "MISS" in desc: c["ft_miss"] += 1
        elif kind == "Rebound":
            c["reb_player" if named else "reb_team"] += 1
            if named and last_miss is not None:
                c["oreb" if r.teamTricode == last_miss[0] else "dreb"] += 1
                c[("oreb_" if r.teamTricode == last_miss[0] else "dreb_") + last_miss[1]] += 1
            if not named and len(examples["teamreb"]) < 3: examples["teamreb"].append((desc, sub))
        elif kind == "Turnover":
            c["tov"] += 1; c["tov_player" if named else "tov_team"] += 1
            tov[sub or desc.split(" Turnover")[0][-18:]] += 1
            if len(examples["tov"]) < 4: examples["tov"].append((desc, sub))
        elif kind == "Foul":
            fouls[sub] += 1
            if len(examples["foul"]) < 4: examples["foul"].append((desc, sub))
        elif kind in ("period", "Jump Ball", "Timeout", "Substitution", "Violation", "Instant Replay", ""):
            c["other_" + (kind or "blank")] += 1
            if kind == "period" and len(examples["period"]) < 3: examples["period"].append((desc, sub, r.clock, r.period))
        if kind == "Missed Shot": last_miss = (r.teamTricode, "fg")
        elif kind == "Free Throw" and "MISS" in desc: last_miss = (r.teamTricode, "ft")
g = len(ids)
print(f"{g} mac")
print("STEAL/BLOCK satir turleri:", dict(kinds))
for k, v in examples.items(): print(" ", k, v)
print(f"asistli isabet: 2'lik {c['ast2']/c['made2']:.3f}  3'luk {c['ast3']/c['made3']:.3f}")
print(f"blok / kacan 2'lik: {c['block']/c['miss2']:.3f}   top calma / top kaybi: {c['steal']/c['tov']:.3f}   takim top kaybi payi: {c['tov_team']/c['tov']:.3f}")
print(f"takim ribaundu payi: {c['reb_team']/(c['reb_team']+c['reb_player']):.3f}   hucum ribaundu orani: saha {c['oreb_fg']/(c['oreb_fg']+c['dreb_fg']):.3f}  serbest atis {c['oreb_ft']/max(1,c['oreb_ft']+c['dreb_ft']):.3f}")
ft = {k: v for k, v in c.items() if k.startswith("ft_")}; print("serbest atis:", ft)
print("top kaybi turleri:", tov.most_common(12))
print("faul turleri:", fouls.most_common(10))
print("mac basina (iki takim):", {k: round(v / g, 1) for k, v in c.items() if k in ("made2", "miss2", "made3", "miss3", "tov", "steal", "block", "reb_player", "reb_team")})
