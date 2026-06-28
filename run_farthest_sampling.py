#!/usr/bin/env python3
"""
farthest-point sampling in ESMC embedding space
直接挑覆盖最广的 N 条序列，不需要聚类
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / "outputs" / "run_5000"

# ═══ 1. Load cached ESMC embeddings ═══
emb_path = PROJECT / "outputs" / "run_esmc" / "embeddings.npy"
if not emb_path.exists():
    # Try remote path
    emb_path = Path("/home/yons/ESKin/outputs/run_esmc/embeddings.npy")

embeddings = np.load(str(emb_path))
embeddings = StandardScaler().fit_transform(embeddings)
n = len(embeddings)

# Load sequence IDs & predictions
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip(): continue
    p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
    if len(p) == 2:
        six.append({"Sequence_ID": p[0].strip(), "Sequence": p[1].strip()})
df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
six_rows = [{"Sequence_ID": e["Sequence_ID"], "Sequence": e["Sequence"]} for e in six]
df = pd.concat([df, pd.DataFrame(six_rows)], ignore_index=True)
ids = df["Sequence_ID"].tolist()
seqs = df["Sequence"].tolist()
n = len(ids)

# ═══ 2. Farthest-point sampling ═══
# Start from the sequence with highest Pred_kcat_over_Km
scores = df["Pred_kcat_over_Km"].values if "Pred_kcat_over_Km" in df.columns else np.zeros(n)
first_idx = int(np.argmax(scores))
print(f"Start from: {ids[first_idx]} (Pred_kcat_over_Km={scores[first_idx]:.1f})", flush=True)

N_SAMPLE = 60  # pick 60 representatives
selected = [first_idx]
selected_set = {first_idx}
dists = np.full(n, np.inf)

for i in range(1, N_SAMPLE):
    # Compute distance from latest selected point to all others
    latest = embeddings[selected[-1]]
    d = np.linalg.norm(embeddings - latest, axis=1)
    dists = np.minimum(dists, d)  # distance to nearest selected
    # Pick farthest
    while True:
        farthest = int(np.argmax(dists))
        if farthest not in selected_set and dists[farthest] > 1e-10:
            break
        dists[farthest] = -1  # mark as done
    selected.append(farthest)
    selected_set.add(farthest)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{N_SAMPLE} selected", flush=True)

print(f"\nSelected {len(selected)} sequences", flush=True)

# ═══ 3. Save ═══
sel_df = pd.DataFrame({
    "id": [ids[i] for i in selected],
    "sequence": [seqs[i] for i in selected],
    "Pred_kcat_over_Km": [scores[i] for i in selected],
    "rank": range(1, len(selected)+1),
})
sel_df.to_csv(OUT / "farthest_sampling_60.csv", index=False)
print(f"✅ {OUT / 'farthest_sampling_60.csv'}", flush=True)

# FASTA
fasta_path = OUT / "farthest_sampling_60.fasta"
with open(fasta_path, "w") as f:
    for i in selected:
        f.write(f">{ids[i]} | Pred_kcat_over_Km={scores[i]:.1f}\n{seqs[i]}\n")
print(f"✅ {fasta_path}", flush=True)

# ═══ 4. UMAP plot ═══
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

um = umap.UMAP(n_neighbors=25, min_dist=0.08, metric="cosine", random_state=42)
xy = um.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")

# All points in grey
ax.scatter(xy[:, 0], xy[:, 1], c="#d0d0d0", s=3, alpha=0.3, edgecolors="none", rasterized=True)

# Selected points highlighted
ax.scatter(xy[selected, 0], xy[selected, 1], c="#e74c3c", s=30, alpha=0.8,
           edgecolors="white", linewidths=0.5, label=f"Selected ({N_SAMPLE})")

# Sixdata targets
tgt_idx = list(range(n - 6, n))
names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
for ti, nm in zip(tgt_idx, names):
    c = "#2ecc71" if ti in selected_set else "#e74c3c"
    ax.scatter(xy[ti, 0], xy[ti, 1], c=c, s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ax.annotate(nm, (xy[ti, 0], xy[ti, 1]),
                xytext=(xy[ti, 0]+20, xy[ti, 1]+20),
                fontsize=11, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5), zorder=11)

ax.set_title(f"Farthest-point Sampling in ESMC Space  |  60 reps from {n} seqs",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, markerscale=2)
ax.grid(alpha=0.08)
plt.tight_layout()
sp = OUT / "farthest_sampling_60.png"
plt.savefig(sp, dpi=250, bbox_inches="tight"); plt.close()
print(f"✅ {sp}", flush=True)

print("\nDONE ✅", flush=True)
