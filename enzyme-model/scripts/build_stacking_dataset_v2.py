#!/usr/bin/env python3
"""Stacking V2 — Auto-discover model predictions and build meta-feature dataset.

Scans artifacts/*/{task}/predictions.csv for any model that exports
the standard schema (sample_id, split, y_true, y_pred) and merges
them into a single meta-feature matrix.

Usage:
    python scripts/build_stacking_dataset_v2.py --task kcat
    python scripts/build_stacking_dataset_v2.py --task km

Output:
    artifacts/stacking_v2/{task}/meta_features.csv
"""
import argparse, json, sys, glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
STACKING_DIR = ARTIFACTS / "stacking_v2"

REQUIRED_COLS = {"sample_id", "split", "y_true", "y_pred"}


def discover_models(task: str) -> list[dict]:
    """Scan artifacts/ for models with valid predictions.csv.

    Returns list of {name, path, df} sorted alphabetically.
    """
    discovered = []
    for pred_path in sorted(ARTIFACTS.rglob(f"{task}/predictions.csv")):
        # Skip stacking directories themselves
        if "stacking" in pred_path.parts and "stacking" != pred_path.parents[1].name:
            continue
        try:
            df = pd.read_csv(pred_path)
            if REQUIRED_COLS.issubset(set(df.columns)):
                model_name = pred_path.parents[1].name  # e.g. "baseline"
                discovered.append({"name": model_name, "path": pred_path, "df": df})
            else:
                print(f"  ⚠️  Skipping {pred_path}: missing cols {REQUIRED_COLS - set(df.columns)}")
        except Exception as e:
            print(f"  ⚠️  Skipping {pred_path}: {e}")
    return discovered


def main():
    parser = argparse.ArgumentParser(description="Build StackingV2 meta-feature dataset")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"])
    args = parser.parse_args()

    out_dir = STACKING_DIR / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover models
    models = discover_models(args.task)
    print(f"任务: {args.task}")
    print(f"发现 {len(models)} 个模型:\n")
    for m in models:
        print(f"  {m['name']:15s} {len(m['df']):>5} rows  cols={list(m['df'].columns)}")

    if len(models) < 2:
        print("❌ 至少需要 2 个模型才能构建 Stacking")
        sys.exit(1)

    # Merge on sample_id + y_true
    base = models[0]["df"][["sample_id", "split", "y_true"]].copy()
    base.rename(columns={}, inplace=True)

    for m in models:
        col_name = f"y_{m['name']}"
        sub = m["df"][["sample_id", "y_pred"]].rename(columns={"y_pred": col_name})
        base = base.merge(sub, on="sample_id", how="inner")

    # Validate
    n_dup = base["sample_id"].duplicated().sum()
    n_null = base.isna().sum().sum()
    n_rows = len(base)
    n_models = len(models)
    splits = base["split"].unique().tolist()

    report = {
        "task": args.task,
        "models_discovered": [m["name"] for m in models],
        "meta_features_rows": n_rows,
        "feature_columns": [c for c in base.columns if c.startswith("y_")],
        "split_distribution": base["split"].value_counts().to_dict(),
        "duplicate_sample_ids": int(n_dup),
        "null_values": int(n_null),
    }

    print(f"\n合并结果:")
    print(f"  Rows: {n_rows}")
    print(f"  Features: {report['feature_columns']}")
    print(f"  Splits: {splits}")
    print(f"  Duplicates: {n_dup} {'✅' if n_dup==0 else '❌'}")
    print(f"  Nulls: {n_null} {'✅' if n_null==0 else '❌'}")

    if n_dup > 0 or n_null > 0:
        print("❌ 质量检查失败")
        sys.exit(1)

    # Write
    meta_path = out_dir / "meta_features.csv"
    base.to_csv(meta_path, index=False)
    print(f"\nMeta features saved: {meta_path}")

    report_path = out_dir / "meta_build_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")
    print("✅ StackingV2 dataset ready")


if __name__ == "__main__":
    main()
