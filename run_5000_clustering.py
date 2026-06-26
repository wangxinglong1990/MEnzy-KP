#!/usr/bin/env python3
"""
sixdata 6条嵌入 5000序列 → 聚类 → UMAP散点图
k-mer+TF-IDF+KMeans (K=5) + UMAP

用法: python3 -u run_5000_clustering.py
"""
import os, sys, math
from pathlib import Path

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

fp = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
fm.fontManager.addfont(fp)
plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()

PROJECT = Path(__file__).resolve().parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
from src.clustering.kmer import KmerFeatureExtractor
from src.clustering.cluster import ClusterAnalyzer

OUT = PROJECT / "outputs" / "run_5000"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════ 1. DATA ═══════════════
print("Step 1: Build dataset", flush=True)
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split('\n'):
    if not ln.strip(): continue
    p = ln.split('：',1) if '：' in ln else ln.split(':',1)
    if len(p)==2: six.append({"Sequence_ID":p[0].strip(),"Full_Header":p[0].strip(),"Sequence":p[1].strip(),"Length":len(p[1].strip())})
df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
df = pd.concat([df, pd.DataFrame(six)], ignore_index=True)
print(f"  {len(df)} seqs ({len(df)-6} unique + 6 targets)", flush=True)
n = len(df)
ids, seqs = df["Sequence_ID"].tolist(), df["Sequence"].tolist()
tgt = list(range(n-6, n))
names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]

# ═══════════════ 2. K-MER ═══════════════
print("Step 2: k-mer (k=3) + TF-IDF", flush=True)
ext = KmerFeatureExtractor(k=3)
X, ic, vc, sc = ext.build_matrix_from_lists(ids, seqs)
Xtf = ext.tfidf_transform(X)
if hasattr(Xtf,'toarray'): Xtf = Xtf.toarray()
Xs = StandardScaler().fit_transform(Xtf)
print(f"  vocab={len(vc)} shape={Xs.shape}", flush=True)

# ═══════════════ 3. CLUSTER ═══════════════
print("Step 3: KMeans (K=5)", flush=True)
az = ClusterAnalyzer(n_clusters=5)
labels = az.fit(Xs)
k = az.chosen_k_
disp = az.dispersion_report(Xs)
g = disp["global"]
print(f"  K={k} Sil={g['silhouette_score']:.4f} DB={g['davies_bouldin_score']:.2f} CH={g['calinski_harabasz_score']:.2f}", flush=True)
for c in disp["per_cluster"]:
    print(f"  C{c['cluster']}: n={c['n_members']} rad={c['radius']:.4f}", flush=True)
ClusterAnalyzer.export_stats(disp, str(OUT))
pd.DataFrame({"id":ic,"cluster":labels}).to_csv(OUT/"seq_to_cluster.csv",index=False)
for nm,ti in zip(names,tgt):
    print(f"  ★ {nm} → C{labels[ti]}", flush=True)

# ═══════════════ 4. UMAP ═══════════════
print("Step 4: UMAP → scatter", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.08,
               metric='cosine', random_state=42, verbose=True)
xy = um.fit_transform(Xs)
print(f"  UMAP shape={xy.shape}", flush=True)

# Plot
fig, ax = plt.subplots(figsize=(16,14))
fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
pal = ['#2563eb','#ea580c','#16a34a','#7c3aed','#d946ef','#0d9488','#dc2626','#f59e0b']
for c in range(k):
    m = labels==c
    ax.scatter(xy[m,0], xy[m,1], c=pal[c], s=5, alpha=0.4,
               label=f'C{c} (n={m.sum()})', edgecolors='none', rasterized=True)
for ti,nm in zip(tgt,names):
    ax.scatter(xy[ti,0], xy[ti,1], c='red', s=350, marker='*',
               edgecolors='black', linewidths=2, zorder=10)
    ox=(xy[:,0].max()-xy[:,0].min())*0.035
    oy=(xy[:,1].max()-xy[:,1].min())*0.035
    ax.annotate(f'{nm}(C{labels[ti]})', (xy[ti,0],xy[ti,1]),
                xytext=(xy[ti,0]+ox,xy[ti,1]+oy), fontsize=11,
                fontweight='bold', color='#dc2626',
                bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#dc2626',alpha=0.92),
                arrowprops=dict(arrowstyle='->',color='#dc2626',lw=1.5), zorder=11)
ax.set_xlabel('UMAP-1',fontsize=14); ax.set_ylabel('UMAP-2',fontsize=14)
ax.set_title(f'k-mer+TF-IDF+KMeans(K={k})  |  n={n}  |  ★ = sixdata',fontsize=17,fontweight='bold',pad=20)
ax.legend(loc='upper right',fontsize=9,framealpha=0.9,markerscale=3)
ax.grid(alpha=0.12,linestyle='--')
cs=" | ".join([f"C{c}={(labels==c).sum()}" for c in range(k)])
ts=", ".join([f"{nm}(C{labels[ti]})" for nm,ti in zip(names,tgt)])
ax.text(0.02,0.02,f"K={k} Sil={g['silhouette_score']:.4f} DB={g['davies_bouldin_score']:.2f}\n{cs}\n{ts}",
        transform=ax.transAxes,fontsize=9,va='bottom',family='monospace',
        bbox=dict(boxstyle='round,pad=0.5',fc='white',ec='#ccc',alpha=0.9))
plt.tight_layout()
sp = OUT/"cluster_scatter.png"
plt.savefig(sp,dpi=250,bbox_inches='tight',facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ {sp}  ({xy.shape[1]}dim)", flush=True)
print("DONE", flush=True)
