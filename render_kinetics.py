"""Ajinomoto 493 — Threshold-based grouping by kcat/Km"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

df = pd.read_csv("datafor/ajinomoto_493.csv")
df = df.sort_values("Pred_kcat_over_Km", ascending=False).head(200)
km = df["Pred_Km"].values
kcat = df["Pred_kcat"].values
ratio = df["Pred_kcat_over_Km"].values

# Threshold-based groups
labels = np.zeros(200, dtype=int)
labels[ratio > 10] = 2
labels[(ratio > 5) & (ratio <= 10)] = 1
labels[ratio <= 5] = 0

pal = ["#2563eb", "#f59e0b", "#dc2626"]
names = ["Moderate (<=5)", "High (5-10)", "Ultra-high (>10)"]
symbols = ["o", "s", "D"]

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor("#f8fafc")

# Left: KM vs kcat
ax = axes[0]; ax.set_facecolor("#f8fafc")
for c in range(3):
    m = labels == c
    ax.scatter(km[m], kcat[m], c=pal[c], s=35, alpha=0.85, marker=symbols[c],
               edgecolors="white", linewidths=0.3,
               label=f"{names[c]} (n={m.sum()})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Predicted KM (M)", fontsize=14)
ax.set_ylabel("Predicted kcat (s$^{-1}$)", fontsize=14)
ax.set_title("Ajinomoto Top200: KM vs kcat (Grouped by Catalytic Efficiency)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.12)
kr = np.logspace(-3, 0.5, 100)
for r, ls in [(5,"--"),(10,"-")]:
    ax.plot(kr, r*kr, "gray", linestyle=ls, alpha=0.4, lw=2)
    ax.text(kr[-1]*1.05, r*kr[-1]*0.9, f"kcat/Km={r}", fontsize=9, color="gray")

# Right: kcat/Km distribution
ax = axes[1]; ax.set_facecolor("#f8fafc")
bins = np.linspace(ratio.min()-0.5, ratio.max()+0.5, 25)
for c in range(3):
    m = labels == c
    ax.hist(ratio[m], bins=bins, color=pal[c], alpha=0.55, edgecolor="white")
ax.axvline(5, color="gray", linestyle="--", lw=2, alpha=0.4)
ax.axvline(10, color="gray", linestyle="-", lw=2, alpha=0.4)
ax.set_xlabel("kcat/Km", fontsize=14); ax.set_ylabel("Count", fontsize=14)
ax.set_title("kcat/Km Distribution", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.12)

for c in range(3):
    m = labels == c
    if m.sum() > 0:
        print(f"{names[c]}: n={m.sum()} ratio=[{ratio[m].min():.1f}~{ratio[m].max():.1f}]")

plt.tight_layout(pad=3)
out = "outputs/run_best/cluster_by_kinetics.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"DONE: {out}")
