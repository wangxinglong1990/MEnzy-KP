"""Rank-based composite scoring"""
import pandas as pd, numpy as np

df = pd.read_csv("datafor/ncbi5000_composite.csv")

df["rank_kcat"] = df["Pred_kcat_over_Km"].rank(ascending=False)
df["rank_sim"] = df["Similarity"].rank(ascending=False)
df["combined_rank"] = df["rank_sim"] * 0.6 + df["rank_kcat"] * 0.4
df = df.sort_values("combined_rank").reset_index(drop=True)

n_total = len(df)
n_top = int(n_total * 0.10)

six_shorts = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
print("Rank-based composite:", flush=True)
for ss in six_shorts:
    m = df[df["Entry"].str.contains(ss, na=False, regex=False)]
    if len(m) > 0:
        r = int(m.index[0]) + 1
        v = float(m["Pred_kcat_over_Km"].iloc[0])
        s = float(m["Similarity"].iloc[0])
        print("  %s: #%d/%d  kcat/Km=%.0f  sim=%.3f  %s" % (ss, r, n_total, v, s, "GREEN" if r <= n_top else "gray"), flush=True)

# Also try: just rank by sim (sixdata should be #1 since sim=1.0)
print("\nJust similarity ranking:", flush=True)
df2 = df.sort_values("Similarity", ascending=False).reset_index(drop=True)
for ss in six_shorts:
    m = df2[df2["Entry"].str.contains(ss, na=False, regex=False)]
    if len(m) > 0:
        r = int(m.index[0]) + 1
        print("  %s: #%d" % (ss, r), flush=True)

df.to_csv("datafor/ncbi5000_composite.csv", index=False)
print("DONE", flush=True)
