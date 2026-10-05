import pickle, time, pandas as pd, numpy as np
t0 = time.time()
with open("matchup_cache_v1.pkl", "rb") as f:
    c = pickle.load(f)
df = c["long_df"]
print(f"loaded in {time.time()-t0:.1f}s; rows {len(df):,}; cols {list(df.columns)}; mem {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
print("games:", df.game_id.nunique(), "seasons:", df.groupby("season").game_id.nunique().to_dict())
print("person_id_off zero share: %.4f  person_id_def zero share: %.4f" % ((df.person_id_off == 0).mean(), (df.person_id_def == 0).mean()))
print(df.head(3).to_string())
# per (game, def_team) distinct defenders
nd = df.groupby(["game_id", "def_team"]).person_id_def.nunique()
print("defenders per team-game: mean %.2f sd %.2f" % (nd.mean(), nd.std()))
