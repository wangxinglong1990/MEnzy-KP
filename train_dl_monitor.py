#!/usr/bin/env python3
"""
Train a deep learning regressor on ESMC protein embeddings → kcat/Km,
capturing per-epoch training/validation metrics for monitoring curves.

Outputs:
  - figures/Fig_training_curves.png  (publication-quality plot)
  - training_history.json            (raw metrics)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Style ────────────────────────────────────────────────
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import _get_protein_encoder


# ── Data ─────────────────────────────────────────────────
def load_data(csv_path: str, n_samples: int = 3000):
    """Load sequences and ESKin-predicted targets."""
    df = pd.read_csv(csv_path)
    df = df.head(n_samples).copy()
    sequences = df["Sequence"].astype(str).str.strip().tolist()
    targets = df["Pred_kcat_over_Km"].values.astype(np.float32)
    # Log10-transform (enzyme kinetics convention)
    targets = np.log10(np.clip(targets, 1e-6, None))
    return sequences, targets


# ── Model ─────────────────────────────────────────────────
class EnzymeMLP(nn.Module):
    """3-layer MLP for enzyme kinetics regression."""
    def __init__(self, input_dim=960, hidden_dims=(512, 256, 128), dropout=0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(y)
        n += len(y)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    yp = np.concatenate(preds)
    yt = np.concatenate(trues)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(yt - yp))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    pearson = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else 0.0
    result = {"loss": total_loss / max(n, 1), "r2": float(r2), "mae": float(mae),
              "rmse": rmse, "pearson_r": pearson}
    if return_preds:
        result["y_pred"] = yp
        result["y_true"] = yt
    return result


# ── Main ──────────────────────────────────────────────────
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    csv_path = PROJECT_ROOT / "seqdump" / "seqdump(1).csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run prediction first.")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────
    print("Loading sequences...")
    sequences, targets = load_data(str(csv_path), n_samples=3000)
    print(f"Loaded {len(sequences)} samples, target range: [{targets.min():.2f}, {targets.max():.2f}]")

    # ── Extract ESMC embeddings ──────────────────────────
    print("Extracting ESMC-300M embeddings (this takes 2-5 min)...")
    encoder = _get_protein_encoder()
    embeddings = encoder.encode(sequences).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Normalize ────────────────────────────────────────
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(embeddings)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(targets.reshape(-1, 1)).ravel()

    # ── Train/val/test split ──────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── DataLoaders ──────────────────────────────────────
    B = 64
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=B, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=B, shuffle=False)

    # ── Model ────────────────────────────────────────────
    model = EnzymeMLP(input_dim=960).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    # ── Training loop with metric logging ────────────────
    EPOCHS = 200
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_r2": [], "val_mae": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nTraining {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_mae"].append(val_metrics["mae"])
        history["lr"].append(current_lr)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | val_r2={val_metrics['r2']:.4f} | lr={current_lr:.2e}")

    # ── Final test eval ───────────────────────────────────
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=B, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion, device, return_preds=True)
    y_test_pred = test_metrics.pop("y_pred")
    y_test_true = test_metrics.pop("y_true")
    residuals = y_test_true - y_test_pred

    # Also get train predictions for accuracy assessment
    train_loader_eval = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=B, shuffle=False)
    train_metrics = evaluate(model, train_loader_eval, criterion, device, return_preds=True)
    y_train_pred = train_metrics.pop("y_pred")
    y_train_true = train_metrics.pop("y_true")

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Train: R2={train_metrics['r2']:.4f}, MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}")
    print(f"Test:  R2={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson r={test_metrics['pearson_r']:.4f}")

    # ── Save history ──────────────────────────────────────
    out_dir = PROJECT_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    history["test_r2"] = test_metrics["r2"]
    history["test_mae"] = test_metrics["mae"]
    history["test_rmse"] = test_metrics["rmse"]
    history["test_pearson_r"] = test_metrics["pearson_r"]
    history["train_r2"] = train_metrics["r2"]

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)

    # ── PLOT: 2×3 layout ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    epochs = history["epoch"]

    # (A) Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2171b5", linewidth=1.2, alpha=0.85, label="Training Loss")
    ax.plot(epochs, history["val_loss"], color="#e6550d", linewidth=1.8, label="Validation Loss")
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate(f"Best: epoch {best_epoch}", xy=(best_epoch, history["val_loss"][best_epoch-1]),
                xytext=(best_epoch+15, history["val_loss"][best_epoch-1]*1.25),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("A  Training & Validation Loss")
    ax.legend(frameon=True, fancybox=False, edgecolor="lightgray", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (B) Validation R²
    ax = axes[0, 1]
    ax.plot(epochs, history["val_r2"], color="#238b45", linewidth=1.8, label="Validation R2")
    ax.fill_between(epochs, 0, history["val_r2"], alpha=0.12, color="#238b45")
    ax.axhline(y=test_metrics["r2"], color="#d94801", linestyle=":", linewidth=1.2, alpha=0.7,
              label=f"Test R2 = {test_metrics['r2']:.3f}")
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2 Score")
    ax.set_title("B  Validation R2")
    ax.legend(frameon=True, fancybox=False, edgecolor="lightgray", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (C) MAE + RMSE
    ax = axes[0, 2]
    ax.plot(epochs, history["val_mae"], color="#8c6bb1", linewidth=1.8, label="Validation MAE")
    ax.axhline(y=test_metrics["mae"], color="#d94801", linestyle=":", linewidth=1.2, alpha=0.7,
              label=f"Test MAE = {test_metrics['mae']:.3f}")
    ax.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (log10 scale)")
    ax.set_title("C  Validation MAE")
    ax.legend(frameon=True, fancybox=False, edgecolor="lightgray", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (D) Learning rate schedule
    ax = axes[1, 0]
    ax.plot(epochs, history["lr"], color="#d94801", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("D  Learning Rate Schedule (ReduceLROnPlateau)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (E) Parity Plot — Predicted vs Actual (Test Set)
    ax = axes[1, 1]
    # Convert back from standardized to log10 scale for interpretability
    yt_orig = scaler_y.inverse_transform(y_test_true.reshape(-1, 1)).ravel()
    yp_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
    ax.scatter(yt_orig, yp_orig, c="#2171b5", alpha=0.45, s=18, edgecolors="none",
              label=f"Test set (n={len(yt_orig)})")
    # Identity line
    lims = [min(yt_orig.min(), yp_orig.min()) - 0.1, max(yt_orig.max(), yp_orig.max()) + 0.1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1.0, alpha=0.7, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("ESKin-predicted log10(kcat/Km)")
    ax.set_ylabel("MLP-predicted log10(kcat/Km)")
    ax.set_title("E  Parity Plot — Prediction Accuracy")
    # Annotate metrics
    textstr = f"R2 = {test_metrics['r2']:.3f}\nMAE = {test_metrics['mae']:.3f}\nRMSE = {test_metrics['rmse']:.3f}\nPearson r = {test_metrics['pearson_r']:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9))
    ax.legend(frameon=True, fancybox=False, edgecolor="lightgray", fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (F) Residual Distribution
    ax = axes[1, 2]
    ax.hist(residuals, bins=40, color="#8c6bb1", alpha=0.7, edgecolor="white", linewidth=0.3,
            label=f"Test residuals\n(mean={residuals.mean():.4f}, std={residuals.std():.4f})")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    # Overlay KDE-like smooth curve
    from scipy import stats as sp_stats
    kde_x = np.linspace(residuals.min(), residuals.max(), 200)
    kde = sp_stats.gaussian_kde(residuals)
    ax_twin = ax.twinx()
    ax_twin.plot(kde_x, kde(kde_x), color="#d94801", linewidth=1.8, alpha=0.8)
    ax_twin.set_ylabel("Density", color="#d94801", alpha=0.6)
    ax_twin.tick_params(axis="y", colors="#d94801", alpha=0.6)
    ax_twin.set_yticks([])
    ax.set_xlabel("Residual (True − Predicted) [standardized]")
    ax.set_ylabel("Count")
    ax.set_title("F  Residual Distribution")
    ax.legend(frameon=True, fancybox=False, edgecolor="lightgray", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("EnzymeMLP — Deep Learning Regression Training Monitor & Accuracy Assessment\n"
                 f"ESMC-300M embeddings (960-dim) to log10(kcat/Km) | "
                 f"Params: {n_params:,} | "
                 f"Train R2={train_metrics['r2']:.3f} | Test R2={test_metrics['r2']:.3f}",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    fig_path = out_dir / "Fig_training_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {fig_path}")

    pdf_path = out_dir / "Fig_training_curves.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"PDF saved:   {pdf_path}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
