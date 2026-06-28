#!/usr/bin/env python3
"""
从 3×2 聚类结果中，按 Pred_kcat_over_Km 从每个子簇选代表序列
输出：① FASTA ② 汇总表 ③ 子簇统计
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

OUT = PROJECT / "outputs" / "run_5000"
TOPK = 10  # 每簇选多少条

# ═══ 1. Load data ═══
print("=" * 60, flush=True)
print("Load cluster assignments & predictions", flush=True)

# Cluster assignments (ESMC 3×2)
clusters = pd.read_csv(OUT / "seq_to_cluster_hier3x2.csv")

# The 5000 sequences come from final_submission_5000_sequences.csv (has Pred_kcat_over_Km)
seqs_5000 = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")

# ═══ 2. Merge cluster assignments with sequences ═══
print(f"Cluster file: {len(clusters)} seqs", flush=True)
print(f"5000 seqs file: {len(seqs_5000)} seqs", flush=True)

# Merge to get sequences + predictions
merged = clusters.merge(
    seqs_5000[["Sequence_ID", "Sequence", "Pred_kcat_over_Km"]],
    left_on="id", right_on="Sequence_ID", how="left"
)

# For sixdata targets that might not be in seqs_5000
sixdata = clusters[~clusters["id"].isin(seqs_5000["Sequence_ID"])]
if len(sixdata) > 0:
    # Load sixdata sequences
    txt = open(PROJECT / "textdocs" / "sixdata.text").read()
    for ln in txt.strip().split("\n"):
        if not ln.strip(): continue
        p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
        if len(p) == 2:
            sid = p[0].strip()
            if sid in sixdata["id"].values:
                merged.loc[merged["id"] == sid, "Sequence"] = p[1].strip()

# ═══ 3. Print cluster stats ═══
print("\n" + "=" * 60, flush=True)
print("6 sub-clusters overview", flush=True)
print(f"{'L1':>4} {'L2':>4} {'Count':>6} {'%':>6}", flush=True)
print("-" * 24, flush=True)
for l1 in range(3):
    for l2 in sorted(clusters[clusters["l1_cluster"]==l1]["l2_sub_cluster"].unique()):
        cnt = len(clusters[clusters["l2_sub_cluster"]==l2])
        print(f"{l1:>4} {l2:>4} {cnt:>6} {cnt/len(clusters)*100:>5.1f}%", flush=True)

# ═══ 4. Select top N from each sub-cluster ═══
print("\n" + "=" * 60, flush=True)
print(f"Select TOP {TOPK} sequences from each sub-cluster")
print("-" * 60, flush=True)

selected = []
for sc in sorted(clusters["l2_sub_cluster"].unique()):
    mask = merged["l2_sub_cluster"] == sc
    pool = merged[mask].copy()

    # Sort & pick top
    pool = pool.sort_values("Pred_kcat_over_Km", ascending=False).head(TOPK)
    selected.append(pool)

    print(f"  L2-C{sc} ({mask.sum()} seqs):", flush=True)
    for _, row in pool.iterrows():
        score = row.get("Pred_kcat_over_Km", 0)
        tag = " ★" if pd.notna(row.get("is_target")) else ""
        print(f"    {row['id']:>10s}  score={score:.3f}{tag}", flush=True)

# ═══ 5. Export FASTA ═══
df_selected = pd.concat(selected, ignore_index=True)

fasta_path = OUT / "selected_hier3x2.fasta"
with open(fasta_path, "w") as f:
    for _, row in df_selected.iterrows():
        seq = str(row.get("Sequence", ""))
        if seq:
            f.write(f">{row['id']} | L2-C{row['l2_sub_cluster']} | Pred_kcat_over_Km={row.get('Pred_kcat_over_Km', 0):.3f}\n{seq}\n")
print(f"\n  ✅ FASTA: {fasta_path} ({len(df_selected)} seqs)", flush=True)

# ═══ 6. Export summary table ═══
summary_path = OUT / "selected_hier3x2.csv"
df_selected.to_csv(summary_path, index=False)
print(f"  ✅ Table: {summary_path}", flush=True)

# Per-cluster stats
stats = []
for sc in sorted(clusters["l2_sub_cluster"].unique()):
    n = len(clusters[clusters["l2_sub_cluster"]==sc])
    n_tgt = len(clusters[(clusters["l2_sub_cluster"]==sc) & clusters["is_target"].notna()])
    stats.append({"sub_cluster": sc, "total": n, "targets_in_cluster": n_tgt})
pd.DataFrame(stats).to_csv(OUT / "hier3x2_cluster_stats.csv", index=False)

print(f"\n{'='*60}", flush=True)
print("DONE ✅", flush=True)
print(f"\nNext step: use {fasta_path} for docking", flush=True)
