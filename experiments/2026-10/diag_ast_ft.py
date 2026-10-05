"""How well does the play-by-play recover assists and free throws, per player-game?"""
import os, re, collections
import pandas as pd, numpy as np

AST = re.compile(r"\(([^()]+?) (\d+) AST\)")
FT = re.compile(r"Free Throw\s*(Technical|Flagrant|Clear Path)?\s*(?:(\d) of (\d))?")
games = pd.read_pickle("phonedb_cache/games.pkl")
box = pd.read_pickle("phonedb_cache/box_player_traditional.pkl")
trk = pd.read_pickle("phonedb_cache/player_tracking.pkl")

def run(season, n):
    ids = sorted(games.loc[games.season == season, "game_id"].astype(str).str.zfill(10))[:n]
    ast_rows, ft_rows, shapes = [], [], collections.Counter()
    lost_unseen = lost_collision = total_ast = 0
    for gid in ids:
        path = f"nba_data/{season}/{gid}/play_by_play/{gid}pbp.csv"
        if not os.path.exists(path): continue
        d = pd.read_csv(path, low_memory=False).sort_values("actionNumber")
        named = d[d.playerName.notna() & (d.personId.fillna(0) != 0)]
        full = collections.defaultdict(set)
        for t, nme, pid in zip(named.teamTricode, named.playerName, named.personId):
            if isinstance(t, str): full[(t, nme)].add(int(pid))
        seen = collections.defaultdict(set)
        ast, ftm, fta = collections.Counter(), collections.Counter(), collections.Counter()
        for r in d.itertuples(index=False):
            desc = r.description if isinstance(r.description, str) else ""
            team = r.teamTricode if isinstance(r.teamTricode, str) else None
            if isinstance(r.playerName, str) and r.playerName and team and r.personId:
                seen[(team, r.playerName)].add(int(r.personId))
            if r.actionType == "Made Shot":
                m = AST.search(desc)
                if m and team:
                    total_ast += 1
                    key = (team, m.group(1))
                    if len(seen.get(key, ())) == 1:
                        ast[(gid, next(iter(seen[key])))] += 1
                    elif len(full.get(key, ())) == 1: lost_unseen += 1
                    else: lost_collision += 1
            if r.actionType == "Free Throw" and r.personId:
                m = FT.search(desc)
                shapes[(m.group(1) or "regular", m.group(3) or "-") if m else ("?", desc[:30])] += 1
                fta[(gid, int(r.personId))] += 1
                if "MISS" not in desc: ftm[(gid, int(r.personId))] += 1
        b = box[box.game_id == gid]
        for pid, a, fm, fa in zip(b.player_id, b.ast.fillna(0), b.ftm.fillna(0), b.fta.fillna(0)):
            ast_rows.append((a, ast.get((gid, int(pid)), 0)))
            ft_rows.append((fm, ftm.get((gid, int(pid)), 0), fa, fta.get((gid, int(pid)), 0)))
    a = np.array(ast_rows); f = np.array(ft_rows)
    print(f"\n{season}: {len(ids)} mac")
    print(f"  asist: kutu {int(a[:,0].sum())}, pbp'de {total_ast}, v5'in oyuncuya yazabildigi {int(a[:,1].sum())} "
          f"({a[:,1].sum()/a[:,0].sum():.1%}); kayip: henuz gorulmemis oyuncu {lost_unseen}, ayni takimda ayni soyad {lost_collision}")
    print(f"  serbest atis: isabet kutu {int(f[:,0].sum())} / pbp {int(f[:,1].sum())}; deneme kutu {int(f[:,2].sum())} / pbp {int(f[:,3].sum())}; "
          f"oyuncu-mac birebir {np.mean((f[:,0]==f[:,1])&(f[:,2]==f[:,3])):.4f}")
    print("  serbest atis turleri:", dict(shapes.most_common(12)))

run("2025_2026", 150)
run("2017_2018", 150)
print("\ntracking (sezon bazinda ortalama / bos orani):")
t = trk.merge(games[["game_id", "season"]], on="game_id")
print(t.groupby("season").agg(rows=("sast", "size"), sast_mean=("sast", "mean"), sast_null=("sast", lambda s: s.isna().mean()),
      ftast_mean=("ftast", "mean"), pass_mean=("pass", "mean"), ast_mean=("ast", "mean")).round(3).to_string())
