#!/usr/bin/env python3
"""
Cross-Validation 全套分析 Pipeline
===================================
Part 1: ESMC (GPU) 特征提取 — 25K 序列
Part 2: 10-Fold CV ExtraTreesRegressor
Part 3: CV 统计图表 (R² bar, Pred-vs-Exp scatter, Residuals, kcat/Km distribution)
Part 4: 聚类稳定性评估 (Bootstrap + ARI on 5000 sequences)

用法: /mnt/nvme1/envs/esmc/bin/python3 run_cross_validation.py
"""

import os, sys, time, json, warnings
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, davies_bouldin_score,
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr, gaussian_kde
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

OUT = PROJECT / "outputs" / "cross_validation"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Output: {OUT}")

# ══════════════════════════════════════════════════════════════
# PART 0: 数据加载
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 0: 数据加载")
print("=" * 70)

df_main = pd.read_csv(PROJECT / "data" / "kcat-over-Km-data_0.4simi-10fold.csv")
df_main = df_main.dropna(subset=["Sequence", "kcat(s^-1)", "Km(M)", "fold"])
df_main["fold"] = df_main["fold"].astype(int)
print(f"  主数据集: {len(df_main)} 序列, {df_main['fold'].nunique()} folds")
print(f"  每折: {dict(df_main['fold'].value_counts().sort_index())}")

# 5000 序列数据集（聚类稳定性用）
df_5k = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = (PROJECT / "textdocs" / "sixdata.text").read_text()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip():
        continue
    sep = "：" if "：" in ln else ":"
    p = ln.split(sep, 1)
    if len(p) == 2:
        six.append({"Sequence_ID": p[0].strip(), "Sequence": p[1].strip().upper()})
df_5k = df_5k.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df_5k = df_5k[~df_5k["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
df_5k = pd.concat([df_5k, pd.DataFrame(six)], ignore_index=True)
print(f"  5000序列集: {len(df_5k)} 条 ({len(df_5k)-len(six)} bg + {len(six)} targets)")

# ══════════════════════════════════════════════════════════════
# PART 1: ESMC 特征提取 (GPU)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: ESMC 300M 特征提取 (GPU)")
print("=" * 70)

# Monkey-patch data_root 以绕过 huggingface_hub 网络验证（离线环境）
import esm.utils.constants.esm3 as esm3_constants
_ORIG_DATA_ROOT = esm3_constants.data_root
def _offline_data_root(key):
    cache_map = {
        "esmc-300": os.path.expanduser(
            "~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-300m-2024-12/"
            "snapshots/7f10b20ae75017b2dbc884070e03434515709a8d"
        ),
        "esmc-600": os.path.expanduser(
            "~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/"
            "snapshots/7f10b20ae75017b2dbc884070e03434515709a8d"
        ),
    }
    p = cache_map.get(key)
    if p and Path(p).exists():
        return Path(p)
    return _ORIG_DATA_ROOT(key)
esm3_constants.data_root = _offline_data_root

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def clean_seq(s):
    return "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def extract_esmc_embeddings(sequences, ids, cache_path, label=""):
    """提取 ESMC embeddings，支持缓存"""
    if cache_path.exists():
        print(f"  [{label}] 从缓存加载: {cache_path}")
        return np.load(cache_path)

    print(f"  [{label}] 加载 ESMC 300M 模型...")
    client = ESMC.from_pretrained("esmc_300m", device=DEVICE)
    client.eval()

    embeddings = []
    t_start = time.time()
    batch_times = []

    for idx, seq in enumerate(sequences):
        cleaned = clean_seq(seq)
        if len(cleaned) > 400:
            cleaned = cleaned[:400]
        if len(cleaned) < 5:
            cleaned = "M" * 5  # minimal dummy

        t0 = time.time()
        try:
            protein = ESMProtein(sequence=cleaned)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True),
            )
            emb = logits_output.embeddings
            if emb.dim() == 3:
                pooled = emb.mean(dim=1).squeeze(0)
            elif emb.dim() == 2:
                pooled = emb.mean(dim=0)
            else:
                pooled = emb.reshape(-1)
            vec = pooled.detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"    ⚠️ seq {idx} error: {e}, using zeros")
            vec = np.zeros(960, dtype=np.float32)

        embeddings.append(vec)
        batch_times.append(time.time() - t0)

        if (idx + 1) % 500 == 0 or idx == len(sequences) - 1:
            avg_t = np.mean(batch_times[-100:]) if batch_times else 0
            eta = (len(sequences) - idx - 1) * avg_t
            print(f"    [{label}] {idx+1}/{len(sequences)} avg {avg_t:.3f}s/seq, ETA {eta/60:.1f}min")

    emb = np.array(embeddings, dtype=np.float32)
    np.save(cache_path, emb)
    t_total = time.time() - t_start
    print(f"  [{label}] 完成: {t_total/60:.1f} min, shape={emb.shape}, saved to {cache_path.name}")
    return emb


# 提取主数据集特征
main_seqs = df_main["Sequence"].tolist()
main_ids = df_main.index.tolist()
X_main_raw = extract_esmc_embeddings(
    main_seqs, main_ids,
    OUT / "esmc_embeddings_25k.npy",
    label="25K main"
)
X_main = StandardScaler().fit_transform(X_main_raw)
y_kcat = df_main["kcat(s^-1)"].values.astype(np.float32)
y_km = df_main["Km(M)"].values.astype(np.float32)
y_ratio = y_kcat / (y_km + 1e-12)
fold_col = df_main["fold"].values
print(f"  Features: {X_main.shape}, y_kcat range: [{y_kcat.min():.2e}, {y_kcat.max():.2e}]")

# ══════════════════════════════════════════════════════════════
# PART 2: 10-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: 10-Fold Cross-Validation (LightGBM)")
print("=" * 70)

from lightgbm import LGBMRegressor

folds = sorted(df_main["fold"].unique())
results_kcat = {"fold": [], "r2": [], "pearson_r": [], "mae": [], "rmse": [], "n_train": [], "n_test": []}
results_km = {"fold": [], "r2": [], "pearson_r": [], "mae": [], "rmse": [], "n_train": [], "n_test": []}
results_ratio = {"fold": [], "r2": [], "pearson_r": [], "mae": [], "rmse": [], "n_train": [], "n_test": []}

all_preds = []

for test_fold in folds:
    print(f"\n  --- Fold {test_fold} ---")
    train_idx = fold_col != test_fold
    test_idx = fold_col == test_fold

    X_tr, X_te = X_main[train_idx], X_main[test_idx]
    yk_tr, yk_te = y_kcat[train_idx], y_kcat[test_idx]
    ym_tr, ym_te = y_km[train_idx], y_km[test_idx]
    yr_te = y_ratio[test_idx]

    # log10(1+x) transform — handles 15 decades better than log1p
    yk_tr_log = np.log10(1 + np.clip(yk_tr, 0, None))
    yk_te_log = np.log10(1 + np.clip(yk_te, 0, None))
    ym_tr_log = np.log10(1 + np.clip(ym_tr, 0, None))
    ym_te_log = np.log10(1 + np.clip(ym_te, 0, None))

    # kcat — LightGBM
    t0 = time.time()
    model_kcat = LGBMRegressor(
        n_estimators=100, max_depth=10, num_leaves=31, learning_rate=0.08,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5, n_jobs=8, random_state=42, verbose=-1,
    )
    model_kcat.fit(X_tr, yk_tr_log)
    yk_pred_log = model_kcat.predict(X_te)
    yk_pred = np.power(10, yk_pred_log) - 1  # inverse log10(1+x)
    dt = time.time() - t0

    r2_k = r2_score(yk_te, yk_pred)
    pr_k, _ = pearsonr(yk_te, yk_pred)
    mae_k = mean_absolute_error(yk_te, yk_pred)
    rmse_k = np.sqrt(mean_squared_error(yk_te, yk_pred))

    results_kcat["fold"].append(test_fold)
    results_kcat["r2"].append(r2_k)
    results_kcat["pearson_r"].append(pr_k)
    results_kcat["mae"].append(mae_k)
    results_kcat["rmse"].append(rmse_k)
    results_kcat["n_train"].append(train_idx.sum())
    results_kcat["n_test"].append(test_idx.sum())
    print(f"    kcat: R²={r2_k:.4f} Pearson r={pr_k:.4f} MAE={mae_k:.2e} RMSE={rmse_k:.2e} [{dt:.1f}s]")

    # Km — LightGBM
    model_km = LGBMRegressor(
        n_estimators=100, max_depth=10, num_leaves=31, learning_rate=0.08,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5, n_jobs=8, random_state=42, verbose=-1,
    )
    model_km.fit(X_tr, ym_tr_log)
    ym_pred_log = model_km.predict(X_te)
    ym_pred = np.power(10, ym_pred_log) - 1

    r2_m = r2_score(ym_te, ym_pred)
    pr_m, _ = pearsonr(ym_te, ym_pred)
    mae_m = mean_absolute_error(ym_te, ym_pred)
    rmse_m = np.sqrt(mean_squared_error(ym_te, ym_pred))

    results_km["fold"].append(test_fold)
    results_km["r2"].append(r2_m)
    results_km["pearson_r"].append(pr_m)
    results_km["mae"].append(mae_m)
    results_km["rmse"].append(rmse_m)
    results_km["n_train"].append(train_idx.sum())
    results_km["n_test"].append(test_idx.sum())
    print(f"    Km:   R²={r2_m:.4f} Pearson r={pr_m:.4f} MAE={mae_m:.2e} RMSE={rmse_m:.2e} [{dt:.1f}s]")

    # kcat/Km ratio
    yr_pred = yk_pred / (ym_pred + 1e-12)
    r2_r = r2_score(yr_te, yr_pred)
    pr_r, _ = pearsonr(np.log10(1 + np.clip(yr_te, 0, None)),
                       np.log10(1 + np.clip(yr_pred, 0, None)))
    results_ratio["fold"].append(test_fold)
    results_ratio["r2"].append(r2_r)
    results_ratio["pearson_r"].append(pr_r)
    results_ratio["mae"].append(mean_absolute_error(
        np.log10(1 + np.clip(yr_te, 0, None)),
        np.log10(1 + np.clip(yr_pred, 0, None))))
    results_ratio["rmse"].append(np.sqrt(mean_squared_error(
        np.log10(1 + np.clip(yr_te, 0, None)),
        np.log10(1 + np.clip(yr_pred, 0, None)))))
    results_ratio["n_train"].append(train_idx.sum())
    results_ratio["n_test"].append(test_idx.sum())
    print(f"    ratio: R²={r2_r:.4f} Pearson r(log)={pr_r:.4f}")

    all_preds.append({
        "fold": test_fold,
        "y_true_kcat": yk_te, "y_pred_kcat": yk_pred,
        "y_true_km": ym_te, "y_pred_km": ym_pred,
        "y_true_ratio": yr_te, "y_pred_ratio": yr_pred,
        "sequences": df_main.loc[test_idx, "Sequence"].values,
        "ec": df_main.loc[test_idx, "EC"].values,
    })

# 汇总
print(f"\n  ═══ CV 汇总 ═══")
for name, res in [("kcat", results_kcat), ("Km", results_km), ("kcat/Km", results_ratio)]:
    r2_mean, r2_std = np.mean(res["r2"]), np.std(res["r2"])
    pr_mean, pr_std = np.mean(res["pearson_r"]), np.std(res["pearson_r"])
    print(f"  {name:10s}: R²={r2_mean:.4f}±{r2_std:.4f}  Pearson r={pr_mean:.4f}±{pr_std:.4f}")

# 保存
df_cv_kcat = pd.DataFrame(results_kcat)
df_cv_km = pd.DataFrame(results_km)
df_cv_ratio = pd.DataFrame(results_ratio)
df_cv_kcat.to_csv(OUT / "cv_results_kcat.csv", index=False)
df_cv_km.to_csv(OUT / "cv_results_km.csv", index=False)
df_cv_ratio.to_csv(OUT / "cv_results_ratio.csv", index=False)

# 保存所有预测
all_pred_dfs = []
for ap in all_preds:
    fold_df = pd.DataFrame({
        "fold": ap["fold"],
        "EC": ap["ec"],
        "Sequence": ap["sequences"],
        "y_true_kcat": ap["y_true_kcat"],
        "y_pred_kcat": ap["y_pred_kcat"],
        "y_true_km": ap["y_true_km"],
        "y_pred_km": ap["y_pred_km"],
        "y_true_ratio": ap["y_true_ratio"],
        "y_pred_ratio": ap["y_pred_ratio"],
    })
    all_pred_dfs.append(fold_df)
pd.concat(all_pred_dfs, ignore_index=True).to_csv(OUT / "cv_all_predictions.csv", index=False)
print(f"\n  Results saved to {OUT}")

# ══════════════════════════════════════════════════════════════
# PART 3: CV 统计图表
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: CV 统计图表")
print("=" * 70)

PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef",
           "#0d9488", "#dc2626", "#f59e0b", "#6366f1", "#ec4899"]
FOLD_COLORS = {f: PALETTE[f % len(PALETTE)] for f in folds}


def save_fig(fig, name, dpi=200):
    for fmt in ["png", "pdf"]:
        p = OUT / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


# --- Figure A: Fold-wise R² & Pearson r bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

for ax, (title, res) in zip(axes, [("kcat", results_kcat), ("Km", results_km), ("kcat/Km", results_ratio)]):
    x = np.arange(len(folds))
    w = 0.35
    bars1 = ax.bar(x - w/2, res["r2"], w, color="#2563eb", alpha=0.85, label="R²", edgecolor="white")
    bars2 = ax.bar(x + w/2, res["pearson_r"], w, color="#ea580c", alpha=0.85, label="Pearson r", edgecolor="white")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(max(res["r2"]), max(res["pearson_r"])) * 1.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_ylabel("Score", fontsize=12)

fig.suptitle("10-Fold Cross-Validation: Per-Fold Performance",
             fontsize=17, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig_a_fold_metrics")

# --- Figure B: Predicted vs Experimental (all folds) ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
fig.patch.set_facecolor("white")

for ax, (title, y_true_key, y_pred_key) in zip(
    axes,
    [("kcat", "y_true_kcat", "y_pred_kcat"),
     ("Km", "y_true_km", "y_pred_km"),
     ("kcat/Km", "y_true_ratio", "y_pred_ratio")],
):
    all_y_true = np.concatenate([ap[y_true_key] for ap in all_preds])
    all_y_pred = np.concatenate([ap[y_pred_key] for ap in all_preds])

    # Log scale for kcat and ratio
    if title in ("kcat", "kcat/Km"):
        yt = np.log10(np.clip(all_y_true, 1e-10, None))
        yp = np.log10(np.clip(all_y_pred, 1e-10, None))
        ax.set_xlabel("log₁₀ True", fontsize=12)
        ax.set_ylabel("log₁₀ Predicted", fontsize=12)
    else:
        yt = np.log10(np.clip(all_y_true, 1e-15, None))
        yp = np.log10(np.clip(all_y_pred, 1e-15, None))
        ax.set_xlabel("log₁₀ True (M)", fontsize=12)
        ax.set_ylabel("log₁₀ Predicted (M)", fontsize=12)

    # Hexbin density
    hb = ax.hexbin(yt, yp, gridsize=50, cmap="Blues", mincnt=1, alpha=0.9)
    plt.colorbar(hb, ax=ax, label="Count")

    # Diagonal
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="y=x")

    r2_all = r2_score(all_y_true, all_y_pred)
    pr_all, _ = pearsonr(all_y_true, all_y_pred)
    ax.set_title(f"{title}\nR²={r2_all:.4f}  Pearson r={pr_all:.4f}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.15, linestyle="--")

fig.suptitle("10-Fold CV: Predicted vs Experimental (All Folds Combined)",
             fontsize=17, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig_b_pred_vs_exp")

# --- Figure C: Residual Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("white")

for ax, (title, y_true_key, y_pred_key) in zip(
    axes,
    [("kcat", "y_true_kcat", "y_pred_kcat"),
     ("Km", "y_true_km", "y_pred_km"),
     ("kcat/Km", "y_true_ratio", "y_pred_ratio")],
):
    all_y_true = np.concatenate([ap[y_true_key] for ap in all_preds])
    all_y_pred = np.concatenate([ap[y_pred_key] for ap in all_preds])
    residuals = np.log10(np.clip(all_y_pred, 1e-15, None)) - np.log10(np.clip(all_y_true, 1e-15, None))

    ax.hist(residuals, bins=80, density=True, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)

    # KDE overlay
    kde = gaussian_kde(residuals)
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xs, kde(xs), "r-", linewidth=2)

    ax.set_title(f"{title} (log₁₀ residuals)", fontsize=14, fontweight="bold")
    ax.set_xlabel("log₁₀(Pred) - log₁₀(True)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.axvline(residuals.mean(), color="orange", linestyle=":", linewidth=1.5, label=f"mean={residuals.mean():.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, linestyle="--")

fig.suptitle("10-Fold CV: Residual Distributions",
             fontsize=17, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig_c_residuals")

# --- Figure D: kcat over Km distribution by fold ---
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor("white")

positions = []
labels = []
colors_list = []
all_data = []

for i, f in enumerate(folds):
    mask = fold_col == f
    ratio_fold = y_ratio[mask]
    ratio_log = np.log10(np.clip(ratio_fold, 1e-10, None))
    all_data.append(ratio_log)
    positions.append(i + 1)
    labels.append(f"Fold {f}")
    colors_list.append(PALETTE[f % len(PALETTE)])

bp = ax.boxplot(all_data, positions=positions, patch_artist=True,
                widths=0.6, showfliers=False)
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean markers
for i, d in enumerate(all_data):
    ax.scatter(i + 1 + np.random.uniform(-0.1, 0.1), d.mean(),
               color="red", s=80, zorder=10, marker="D", edgecolors="black", linewidths=0.5)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("log₁₀(kcat/Km)", fontsize=13)
ax.set_title("10-Fold CV: kcat/Km Distribution by Fold", fontsize=16, fontweight="bold")
ax.grid(axis="y", alpha=0.2, linestyle="--")

# Overall stats
all_ratio_log = np.log10(np.clip(y_ratio, 1e-10, None))
ax.text(0.02, 0.98,
        f"Overall: mean={all_ratio_log.mean():.2f}  std={all_ratio_log.std():.2f}  "
        f"CV of means={np.std([d.mean() for d in all_data])/abs(np.mean([d.mean() for d in all_data])):.4f}",
        transform=ax.transAxes, fontsize=10, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))
plt.tight_layout()
save_fig(fig, "fig_d_fold_distribution")

# --- Combined summary figure ---
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Top-left: R² bars
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(folds))
w = 0.25
ax1.bar(x - w, results_kcat["r2"], w, color="#2563eb", alpha=0.85, label="kcat R²", edgecolor="white")
ax1.bar(x, results_km["r2"], w, color="#ea580c", alpha=0.85, label="Km R²", edgecolor="white")
ax1.bar(x + w, results_ratio["r2"], w, color="#16a34a", alpha=0.85, label="kcat/Km R²", edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels([f"F{f}" for f in folds], fontsize=8)
ax1.set_title("R² per Fold", fontsize=13, fontweight="bold")
ax1.legend(fontsize=7, loc="lower right")
ax1.grid(axis="y", alpha=0.2, linestyle="--")

# Top-center: Pearson r bars
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x - w, results_kcat["pearson_r"], w, color="#2563eb", alpha=0.85, label="kcat r", edgecolor="white")
ax2.bar(x, results_km["pearson_r"], w, color="#ea580c", alpha=0.85, label="Km r", edgecolor="white")
ax2.bar(x + w, results_ratio["pearson_r"], w, color="#16a34a", alpha=0.85, label="ratio r", edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels([f"F{f}" for f in folds], fontsize=8)
ax2.set_title("Pearson r per Fold", fontsize=13, fontweight="bold")
ax2.legend(fontsize=7, loc="lower right")
ax2.grid(axis="y", alpha=0.2, linestyle="--")

# Top-right: Pred vs Exp kcat
ax3 = fig.add_subplot(gs[0, 2])
all_yt_k = np.concatenate([ap["y_true_kcat"] for ap in all_preds])
all_yp_k = np.concatenate([ap["y_pred_kcat"] for ap in all_preds])
ax3.hexbin(np.log10(np.clip(all_yt_k, 1e-10, None)),
           np.log10(np.clip(all_yp_k, 1e-10, None)),
           gridsize=40, cmap="Blues", mincnt=1)
lims_k = [np.log10(np.clip(all_yt_k, 1e-10, None)).min(),
          np.log10(np.clip(all_yt_k, 1e-10, None)).max()]
ax3.plot(lims_k, lims_k, "r--", lw=1)
ax3.set_title(f"kcat: Pred vs Exp\nR²={r2_score(all_yt_k, all_yp_k):.3f}", fontsize=12, fontweight="bold")
ax3.set_xlabel("log₁₀ True"); ax3.set_ylabel("log₁₀ Pred")

# Middle-left: Pred vs Exp Km
ax4 = fig.add_subplot(gs[1, 0])
all_yt_m = np.concatenate([ap["y_true_km"] for ap in all_preds])
all_yp_m = np.concatenate([ap["y_pred_km"] for ap in all_preds])
ax4.hexbin(np.log10(np.clip(all_yt_m, 1e-15, None)),
           np.log10(np.clip(all_yp_m, 1e-15, None)),
           gridsize=40, cmap="Oranges", mincnt=1)
lims_m = [np.log10(np.clip(all_yt_m, 1e-15, None)).min(),
          np.log10(np.clip(all_yt_m, 1e-15, None)).max()]
ax4.plot(lims_m, lims_m, "r--", lw=1)
ax4.set_title(f"Km: Pred vs Exp\nR²={r2_score(all_yt_m, all_yp_m):.3f}", fontsize=12, fontweight="bold")
ax4.set_xlabel("log₁₀ True"); ax4.set_ylabel("log₁₀ Pred")

# Middle-center: kcat/Km ratio
ax5 = fig.add_subplot(gs[1, 1])
all_yt_r = np.concatenate([ap["y_true_ratio"] for ap in all_preds])
all_yp_r = np.concatenate([ap["y_pred_ratio"] for ap in all_preds])
ax5.hexbin(np.log10(np.clip(all_yt_r, 1e-10, None)),
           np.log10(np.clip(all_yp_r, 1e-10, None)),
           gridsize=40, cmap="Greens", mincnt=1)
lims_r = [np.log10(np.clip(all_yt_r, 1e-10, None)).min(),
          np.log10(np.clip(all_yt_r, 1e-10, None)).max()]
ax5.plot(lims_r, lims_r, "r--", lw=1)
ax5.set_title(f"kcat/Km: Pred vs Exp\nR²={r2_score(all_yt_r, all_yp_r):.3f}", fontsize=12, fontweight="bold")
ax5.set_xlabel("log₁₀ True"); ax5.set_ylabel("log₁₀ Pred")

# Middle-right: Residuals kcat
ax6 = fig.add_subplot(gs[1, 2])
res_k = np.log10(np.clip(all_yp_k, 1e-10, None)) - np.log10(np.clip(all_yt_k, 1e-10, None))
ax6.hist(res_k, bins=60, density=True, color="#2563eb", alpha=0.6, edgecolor="white")
kde_k = gaussian_kde(res_k); xs_k = np.linspace(res_k.min(), res_k.max(), 200)
ax6.plot(xs_k, kde_k(xs_k), "r-", lw=2)
ax6.axvline(0, color="black", linestyle="--", lw=1)
ax6.set_title(f"kcat Residuals\nμ={res_k.mean():.3f} σ={res_k.std():.3f}", fontsize=12, fontweight="bold")

# Bottom-left: kcat/Km violin plot
ax7 = fig.add_subplot(gs[2, 0:2])
violin_data = []
for f in folds:
    mask = fold_col == f
    violin_data.append(np.log10(np.clip(y_ratio[mask], 1e-10, None)))
parts = ax7.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True, widths=0.7)
for i, (pc, c) in enumerate(zip(parts["bodies"], colors_list)):
    pc.set_facecolor(c); pc.set_alpha(0.6)
for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
    if partname in parts:
        parts[partname].set_color("#333333")
ax7.set_xticks(positions)
ax7.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9)
ax7.set_ylabel("log₁₀(kcat/Km)", fontsize=11)
ax7.set_title("kcat/Km Distribution by Fold", fontsize=13, fontweight="bold")
ax7.grid(axis="y", alpha=0.2, linestyle="--")

# Bottom-right: Summary table
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
summary_text = f"""10-Fold CV Summary
{'─'*30}
kcat:
  R² = {np.mean(results_kcat['r2']):.4f} ± {np.std(results_kcat['r2']):.4f}
  r  = {np.mean(results_kcat['pearson_r']):.4f} ± {np.std(results_kcat['pearson_r']):.4f}
  MAE = {np.mean(results_kcat['mae']):.2e}

Km:
  R² = {np.mean(results_km['r2']):.4f} ± {np.std(results_km['r2']):.4f}
  r  = {np.mean(results_km['pearson_r']):.4f} ± {np.std(results_km['pearson_r']):.4f}

kcat/Km:
  R² = {np.mean(results_ratio['r2']):.4f} ± {np.std(results_ratio['r2']):.4f}
  r  = {np.mean(results_ratio['pearson_r']):.4f} ± {np.std(results_ratio['pearson_r']):.4f}

Model: LightGBM (100 trees, max_depth=10)
Features: ESMC 300M (960-dim)
Folds: {len(folds)} (protein-aware)
Samples: {len(df_main):,}"""
ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9.5,
         va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4ff", ec="#2563eb", alpha=0.9))

fig.suptitle("10-Fold Cross-Validation: Comprehensive Report",
             fontsize=18, fontweight="bold", y=0.99)
save_fig(fig, "fig_summary_comprehensive", dpi=250)

# ══════════════════════════════════════════════════════════════
# PART 4: 聚类稳定性评估
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: 聚类稳定性评估 (Bootstrap + ARI)")
print("=" * 70)

from src.clustering.kmer import KmerFeatureExtractor

# 用 k-mer 特征做快速聚类稳定性
print("  提取 k-mer 特征...")
seqs_5k = df_5k["Sequence"].tolist()
ids_5k = df_5k["Sequence_ID"].tolist()
ext = KmerFeatureExtractor(k=3)
X_5k_raw, _, _, _ = ext.build_matrix_from_lists(ids_5k, seqs_5k)
X_5k_tf = ext.tfidf_transform(X_5k_raw)
if hasattr(X_5k_tf, "toarray"):
    X_5k_tf = X_5k_tf.toarray()
X_5k = StandardScaler().fit_transform(X_5k_tf)
print(f"  5000-seq k-mer features: {X_5k.shape}")

# 自动选最佳 K
print("  自动选择 K...")
k_range = range(2, 11)
best_k, best_sil = 5, -1
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_5k)
    sil = silhouette_score(X_5k, labels)
    if sil > best_sil:
        best_sil, best_k = sil, k
print(f"  Best K={best_k} (Silhouette={best_sil:.4f})")

# Bootstrap 稳定性
N_BOOTSTRAP = 30
n_total = X_5k.shape[0]
bootstrap_labels_list = []
bootstrap_seeds = []
print(f"\n  Bootstrap (n={N_BOOTSTRAP}, K={best_k})...")

for i in range(N_BOOTSTRAP):
    seed = 100 + i
    # Bootstrap sample with replacement
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_total, size=n_total, replace=True)
    X_boot = X_5k[idx]

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels_boot = km.fit_predict(X_boot)

    # Map bootstrap labels back to original indices
    # For stability: compare on the FULL set, using the closest centroid
    centroids = km.cluster_centers_
    labels_full = np.argmin(cdist(X_5k, centroids), axis=1)

    bootstrap_labels_list.append(labels_full)
    bootstrap_seeds.append(seed)

    if (i + 1) % 10 == 0:
        print(f"    Bootstrap {i+1}/{N_BOOTSTRAP}")

# 计算配对 ARI / NMI
print("\n  计算配对稳定性矩阵...")
ari_matrix = np.zeros((N_BOOTSTRAP, N_BOOTSTRAP))
nmi_matrix = np.zeros((N_BOOTSTRAP, N_BOOTSTRAP))

for i, j in combinations(range(N_BOOTSTRAP), 2):
    ari = adjusted_rand_score(bootstrap_labels_list[i], bootstrap_labels_list[j])
    nmi = normalized_mutual_info_score(bootstrap_labels_list[i], bootstrap_labels_list[j])
    ari_matrix[i, j] = ari_matrix[j, i] = ari
    nmi_matrix[i, j] = nmi_matrix[j, i] = nmi

for i in range(N_BOOTSTRAP):
    ari_matrix[i, i] = 1.0
    nmi_matrix[i, i] = 1.0

ari_mean = np.mean(ari_matrix[np.triu_indices(N_BOOTSTRAP, 1)])
ari_std = np.std(ari_matrix[np.triu_indices(N_BOOTSTRAP, 1)])
nmi_mean = np.mean(nmi_matrix[np.triu_indices(N_BOOTSTRAP, 1)])
nmi_std = np.std(nmi_matrix[np.triu_indices(N_BOOTSTRAP, 1)])
print(f"  ARI: {ari_mean:.4f} ± {ari_std:.4f}")
print(f"  NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")

# 聚类稳定性图
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

for ax, matrix, name in zip(axes, [ari_matrix, nmi_matrix], ["ARI", "NMI"]):
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Bootstrap Stability: {name}\nMean = {np.mean(matrix[np.triu_indices(N_BOOTSTRAP, 1)]):.4f} ± {np.std(matrix[np.triu_indices(N_BOOTSTRAP, 1)]):.4f}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Bootstrap #")
    ax.set_ylabel("Bootstrap #")
    plt.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(f"Clustering Stability Assessment\n{len(seqs_5k)} sequences, K={best_k}, {N_BOOTSTRAP} bootstraps",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig_clustering_stability")

# 聚类分配一致性 (consensus clustering)
print("  构建共识聚类...")
consensus_matrix = np.zeros((n_total, n_total))
for labels_boot in bootstrap_labels_list:
    for c in range(best_k):
        mask = labels_boot == c
        consensus_matrix[np.ix_(mask, mask)] += 1
consensus_matrix /= N_BOOTSTRAP

# 共识聚类图 (subsample for visualization)
sub_n = min(500, n_total)
sub_idx = np.random.RandomState(42).choice(n_total, size=sub_n, replace=False)
cons_sub = consensus_matrix[np.ix_(sub_idx, sub_idx)]

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor("white")
im = ax.imshow(cons_sub, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_title(f"Consensus Matrix (subsample n={sub_n})\n"
             f"{n_total} sequences, K={best_k}, {N_BOOTSTRAP} bootstraps\n"
             f"ARI={ari_mean:.4f}±{ari_std:.3f}  NMI={nmi_mean:.4f}±{nmi_std:.3f}",
             fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax, label="Co-clustering frequency", shrink=0.85)
plt.tight_layout()
save_fig(fig, "fig_consensus_matrix")

# 保存稳定性数据
np.savez(OUT / "clustering_stability.npz",
         ari_matrix=ari_matrix, nmi_matrix=nmi_matrix,
         consensus_matrix=consensus_matrix,
         ari_mean=ari_mean, ari_std=ari_std,
         nmi_mean=nmi_mean, nmi_std=nmi_std,
         best_k=best_k, n_bootstrap=N_BOOTSTRAP)
print(f"  稳定性数据已保存")

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("✅ 全套 CV 分析完成")
print("=" * 70)
print(f"\n输出目录: {OUT}")
print("文件列表:")
for f in sorted(OUT.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name}  ({size/1024:.1f} KB)" if size < 1024*1024 else f"  {f.name}  ({size/1024/1024:.1f} MB)")

# 写入 JSON 摘要
summary = {
    "model": "LightGBM (n_estimators=100, max_depth=10, num_leaves=31, n_jobs=8)",
    "features": "ESMC_300M (960-dim, GPU-encoded)",
    "n_samples": len(df_main),
    "n_folds": int(len(folds)),
    "kcat": {"r2_mean": float(np.mean(results_kcat["r2"])), "r2_std": float(np.std(results_kcat["r2"])),
             "pearson_r_mean": float(np.mean(results_kcat["pearson_r"])), "pearson_r_std": float(np.std(results_kcat["pearson_r"]))},
    "km": {"r2_mean": float(np.mean(results_km["r2"])), "r2_std": float(np.std(results_km["r2"])),
           "pearson_r_mean": float(np.mean(results_km["pearson_r"])), "pearson_r_std": float(np.std(results_km["pearson_r"]))},
    "ratio": {"r2_mean": float(np.mean(results_ratio["r2"])), "r2_std": float(np.std(results_ratio["r2"])),
              "pearson_r_mean": float(np.mean(results_ratio["pearson_r"])), "pearson_r_std": float(np.std(results_ratio["pearson_r"]))},
    "clustering_stability": {"K": best_k, "ARI_mean": float(ari_mean), "ARI_std": float(ari_std),
                             "NMI_mean": float(nmi_mean), "NMI_std": float(nmi_std),
                             "n_bootstrap": N_BOOTSTRAP},
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n  summary.json written")
