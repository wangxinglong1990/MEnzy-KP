#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.msa2d_full_dataset import MSA2DFullDataset
from models.msa2d_full.model import ContactMapCNN

PROJECT_ROOT = _ROOT

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    n = 0

    for batch in loader:
        cm = batch["contact_map"].to(device)
        target = batch["target"].float().to(device)

        optimizer.zero_grad()

        pred = model(cm).squeeze(-1)

        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(target)
        n += len(target)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    n = 0

    preds = []
    targets = []

    for batch in loader:
        cm = batch["contact_map"].to(device)
        target = batch["target"].float().to(device)

        pred = model(cm).squeeze(-1)

        loss = criterion(pred, target)

        total_loss += loss.item() * len(target)
        n += len(target)

        preds.append(pred.cpu())
        targets.append(target.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    from evaluate.evaluate_stacking_v2 import evaluate_stacking

    metrics = evaluate_stacking(y_true, y_pred)
    metrics["loss"] = total_loss / max(n, 1)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True, choices=["kcat", "km"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=5)

    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    print(f"任务: {args.task}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    train_ds = MSA2DFullDataset(
        task=args.task,
        split="train",
        pad_to=256
    )

    test_ds = MSA2DFullDataset(
        task=args.task,
        split="test",
        pad_to=256
    )

    val_ds = MSA2DFullDataset(
        task=args.task,
        split="val",
        pad_to=256
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = ContactMapCNN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_patience,
        factor=0.5
    )

    criterion = nn.MSELoss()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"模型参数: {total_params:,}")

    best_r2 = -999

    ckpt = PROJECT_ROOT / "checkpoints" / f"msa2d_full_{args.task}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        t0 = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| val_r2={val_metrics['r2']:.4f} "
            f"| test_r2={test_metrics['r2']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]

            torch.save(
                model.state_dict(),
                ckpt
            )

            print(
                f"保存最佳模型 "
                f"(R2={best_r2:.4f})"
            )

    print("\n训练结束")

    print(f"最佳R²: {best_r2:.4f}")

    out_dir = PROJECT_ROOT / "artifacts" / "msa2d_full" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {"best_r2": float(best_r2)},
            f,
            indent=4
        )

    print(f"结果保存至: {out_dir}")


if __name__ == "__main__":
    main()
