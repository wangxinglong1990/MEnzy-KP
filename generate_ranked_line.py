#!/usr/bin/env python3
"""
排序折线图 (Figure G) — 全序列预测催化效率排名
用法: python3 generate_ranked_line.py --prepare  (Mac准备数据)
      python3 generate_ranked_line.py --predict  (GPU跑预测)
      python3 generate_ranked_line.py --plot     (出图)
"""

import os, sys, argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
OUT_DIR = PROJECT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── prepare ──
def prepare_data():
    import pandas as pd
    import numpy as np
    df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
    txt = (PROJECT / "textdocs" / "sixdata.text").read_text()
    six = []
    for ln in txt.strip().split('\n'):
        if not ln.strip(): continue
        sep = '：' if '：' in ln else ':'
        p = ln.split(sep, 1)
        if len(p) == 2:
            nm = p[0].strip()
            seq = "".join(c for c in p[1].strip().upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            six.append({"Entry": nm, "Enzyme": seq})
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    sseq = {e["Enzyme"] for e in six}
    df = df[~df["Sequence"].apply(lambda s: "".join(
        c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")).isin(sseq)].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        rows.append({"Entry": str(row["Sequence_ID"]),
                     "Enzyme": "".join(c for c in str(row["Sequence"]).upper() if c in "ACDEFGHIKLMNPQRSTVWY"),
                     "Substrates": "COC(=O)CC[NH3+]",
                     "Products": "C1=C(NC=N1)C[C@@H](C(=O)O)NC(=O)CCN"})
    for e in six:
        rows.append({"Entry": e["Entry"], "Enzyme": e["Enzyme"],
                     "Substrates": "COC(=O)CC[NH3+]",
                     "Products": "C1=C(NC=N1)C[C@@H](C(=O)O)NC(=O)CCN"})
    pred_input = pd.DataFrame(rows)
    pred_input.to_csv(OUT_DIR / "prediction_input_5000.csv", index=False)
    six_names = [e["Entry"] for e in six]
    pd.Series(six_names).to_csv(OUT_DIR / "sixdata_names.csv", index=False)
    print(f"Prepared: {len(pred_input)} seqs ({len(six)} targets at end)")

# ── predict ──
def run_prediction():
    import pandas as pd
    import numpy as np
    input_path = OUT_DIR / "prediction_input_5000.csv"
    df = pd.read_csv(input_path)
    print(f"Predicting {len(df)} sequences...")
    from config import KM_MODEL_PATH, KCAT_MODEL_PATH
    from src.features.extractor import extract_joint_features
    import joblib
    sequences = df["Enzyme"].astype(str).str.strip().tolist()
    smiles_list = df["Substrates"].astype(str).str.strip().tolist()
    km_model = joblib.load(str(KM_MODEL_PATH))
    kcat_model = joblib.load(str(KCAT_MODEL_PATH))
    features = extract_joint_features(smiles_list, sequences).astype(np.float32)
    PROTEIN_DIM = 960
    features = np.concatenate([features[:, PROTEIN_DIM:], features[:, :PROTEIN_DIM]], axis=1)
    log10_km = km_model.predict(features)
    log10_kcat = kcat_model.predict(features)
    df["Pred_log10_kcat_over_Km"] = log10_kcat - log10_km
    df.to_csv(OUT_DIR / "prediction_output_5000.csv", index=False)
    scores = df["Pred_log10_kcat_over_Km"].values
    print(f"Done: min={scores.min():.2f} max={scores.max():.2f} median={np.median(scores):.2f}")
    six_names = pd.read_csv(OUT_DIR / "sixdata_names.csv").iloc[:,0].tolist()
    for nm in six_names:
        row = df[df["Entry"] == nm]
        if len(row)>0: print(f"  {nm}: {row['Pred_log10_kcat_over_Km'].values[0]:.3f}")

# ── plot ──
def make_plot():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.ticker as ticker

    _FP = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
    if os.path.exists(_FP): fm.fontManager.addfont(_FP)

    df = pd.read_csv(OUT_DIR / "prediction_output_5000.csv")
    df_sorted = df.sort_values("Pred_log10_kcat_over_Km", ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1
    n = len(df_sorted)
    scores = df_sorted["Pred_log10_kcat_over_Km"].values
    top200_cutoff = scores[min(199, n-1)]

    six_names = pd.read_csv(OUT_DIR / "sixdata_names.csv").iloc[:,0].tolist()
    name_map = {}
    for nm in six_names:
        short = nm.split('.')[1] if '.' in nm else nm.split('：')[1] if '：' in nm else nm
        name_map[nm] = short if short else nm
    target_data = []
    for nm in six_names:
        row = df_sorted[df_sorted["Entry"] == nm]
        if len(row) > 0:
            target_data.append({"name": name_map.get(nm, nm),
                                "rank": int(row["Rank"].values[0]),
                                "score": float(row["Pred_log10_kcat_over_Km"].values[0])})

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafbfc')

    # Top200 绿色填充
    ax.fill_between(df_sorted["Rank"], 0, scores, where=df_sorted["Rank"] <= 200,
                    color='#bae4bc', alpha=0.35, label='Top 200 Candidate Pool')

    # 降采样曲线
    step = max(1, n // 5000)
    xp = df_sorted["Rank"].values[::step]
    yp = scores[::step]
    if xp[-1] != n: xp, yp = np.append(xp, n), np.append(yp, scores[-1])
    ax.plot(xp, yp, color='#2c7bb6', linewidth=1.8, alpha=0.9, label='Ranked Score Curve')

    # Top200 截断线
    ax.axvline(x=200, color='#e41a1c', linestyle='--', linewidth=2.0, alpha=0.8,
               label=f'Top-200 Cutoff (score >= {top200_cutoff:.2f})')

    # 主图：目标序列只画三角
    for td in target_data:
        ax.scatter(td["rank"], td["score"], color='#e41a1c', s=160, marker='^',
                   edgecolors='#333333', linewidths=1.0, zorder=10)

    ax.set_xlabel('Ranked Sequences (by predicted catalytic efficiency)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Predicted log10(Kcat/Km)', fontsize=15, fontweight='bold')
    ax.set_title(f'Global Score Ranking of {n:,} Enzyme Sequences', fontsize=17, fontweight='bold', pad=16)
    ax.set_xlim(-n*0.01, n*1.03)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.92, edgecolor='#cccccc')
    ax.grid(alpha=0.08, linestyle='--', linewidth=0.4)

    # 统计条（底部）
    stats = (f"n = {n:,}  |  Top-200 cutoff = {top200_cutoff:.3f}  |  "
             f"Max = {scores.max():.3f}  |  Median = {np.median(scores):.3f}  |  Min = {scores.min():.3f}")
    ax.text(0.5, -0.06, stats, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', style='italic', color='#666666')

    # ── 放大插图 ──
    inset_x0 = min(td["rank"] for td in target_data) - 120
    inset_x1 = n + 10
    inset_y0 = min(td["score"] for td in target_data) - 0.12
    inset_y1 = max(td["score"] for td in target_data) + 0.25

    ax_inset = inset_axes(ax, width="48%", height="42%",
                          bbox_to_anchor=(0.50, 0.16, 0.48, 0.48),
                          bbox_transform=ax.transAxes, loc='lower left')
    ax_inset.set_facecolor('#fefefe')

    mask = (df_sorted["Rank"] >= inset_x0) & (df_sorted["Rank"] <= inset_x1)
    ax_inset.plot(df_sorted["Rank"][mask], df_sorted["Pred_log10_kcat_over_Km"][mask],
                  color='#2c7bb6', linewidth=2.0, alpha=0.95)

    tgt_sorted = sorted(target_data, key=lambda x: x["rank"])
    # 手动偏移：(x偏移占x范围比例, y偏移绝对值)
    manual_offsets = [
        (-0.05,  0.18),   # MsAP: 左上
        ( 0.04,  0.15),   # MnAP: 右上
        (-0.06, -0.16),   # EaAP: 左下
        ( 0.05, -0.19),   # KoAP: 右下
        (-0.03, -0.08),   # AcAP: 左中下
        ( 0.06,  0.08),   # TcAP: 右中上
    ]
    x_span = inset_x1 - inset_x0
    for i, td in enumerate(tgt_sorted):
        ax_inset.scatter(td["rank"], td["score"], color='#e41a1c', s=200, marker='^',
                         edgecolors='#333333', linewidths=1.5, zorder=10)
        ox, oy = manual_offsets[i]
        ax_inset.annotate(f"{td['name']} (#{td['rank']})",
                          (td["rank"], td["score"]),
                          xytext=(td["rank"] + ox * x_span, td["score"] + oy),
                          fontsize=9, fontweight='bold', color='#c4121c',
                          ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                    edgecolor='#c4121c', alpha=0.93, linewidth=1.0),
                          arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8,
                                          connectionstyle='arc3,rad=0.15'),
                          zorder=11)

    ax_inset.set_xlim(inset_x0, inset_x1)
    ax_inset.set_ylim(inset_y0 - 0.05, inset_y1 + 0.05)
    ax_inset.set_title('Detail: Six Target Candidates (bottom 15%)', fontsize=11,
                       fontweight='bold', color='#c4121c', pad=6)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(alpha=0.10, linestyle='--', linewidth=0.3)

    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc='none', ec='#888888',
               lw=1.5, linestyle='--', alpha=0.6)

    fig.text(0.5, 0.01,
             "Six target aminopeptidases all rank in the bottom 15%, suggesting substrate-specificity divergence from the training set.",
             ha='center', fontsize=9, style='italic', color='#999999')

    fig.subplots_adjust(bottom=0.10, top=0.94, left=0.07, right=0.97)
    fig_path = OUT_DIR / "fig_g_ranked_line.png"
    fig_path_600 = OUT_DIR / "fig_g_ranked_line_600dpi.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(fig_path_600, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.prepare: prepare_data()
    if args.predict: run_prediction()
    if args.plot: make_plot()
    if not any([args.prepare, args.predict, args.plot]): ap.print_help()
