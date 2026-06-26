#!/usr/bin/env python3
"""Build Stacking Dataset — merge predictions from Baseline, Condition, MSA1D.

Usage:
    python scripts/build_stacking_dataset.py --task kcat
    python scripts/build_stacking_dataset.py --task km

Output:
    artifacts/stacking/{task}/meta_features.csv
    artifacts/stacking/{task}/meta_dataset_report.json
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    parser = argparse.ArgumentParser(description="Build Stacking meta-feature dataset")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"])
    args = parser.parse_args()
    task = args.task

    pred_paths = {
        "baseline":   ARTIFACTS / "baseline" / task / "predictions.csv",
        "condition":  ARTIFACTS / "condition" / task / "predictions.csv",
        "msa1d":      ARTIFACTS / "msa1d" / task / "predictions.csv",
    }

    out_dir = ARTIFACTS / "stacking" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check all prediction files exist
    missing = [k for k, p in pred_paths.items() if not p.exists()]
    if missing:
        print(f"❌ Missing prediction files: {missing}")
        print(f"   Run training first: python train/train_{missing[0]}.py --task {task}")
        sys.exit(1)

    # Load and merge
    dfs = {}
    for name, path in pred_paths.items():
        df = pd.read_csv(path)
        df = df.rename(columns={"y_pred": f"y_{name}"})
        df = df[["sample_id", f"y_{name}", "y_true"]]
        dfs[name] = df
        print(f"  {name:12s} {path.name}: {len(df)} rows")

    # Merge on sample_id
    meta = dfs["baseline"]
    for name in ["condition", "msa1d"]:
        meta = meta.merge(dfs[name], on=["sample_id", "y_true"], how="inner")

    meta = meta.rename(columns={"y_true": "y_true"})
    cols = ["sample_id", "y_baseline", "y_condition", "y_msa1d", "y_true"]
    meta = meta[cols]

    # Validation
    n_dup = meta["sample_id"].duplicated().sum()
    n_null = meta.isna().sum().sum()
    n_align = len(meta)
    total_rows = {k: len(v) for k, v in dfs.items()}

    report = {
        "task": task,
        "meta_features_rows": n_align,
        "duplicate_sample_ids": int(n_dup),
        "null_values": int(n_null),
        "input_rows": total_rows,
        "alignment_status": "FULL" if n_align == min(total_rows.values()) else "PARTIAL",
    }

    meta.to_csv(out_dir / "meta_features.csv", index=False)
    print(f"\n  Meta features: {out_dir / 'meta_features.csv'} ({n_align} rows, {len(cols)-1} features + target)")

    with open(out_dir / "meta_dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Validation:")
    print(f"    Duplicate sample_ids: {n_dup} {'✅' if n_dup==0 else '❌'}")
    print(f"    Null values:          {n_null} {'✅' if n_null==0 else '❌'}")
    print(f"    Alignment:            {n_align}/{min(total_rows.values())} rows {'✅' if n_align==min(total_rows.values()) else '⚠️'}")

    if n_dup > 0 or n_null > 0:
        print("❌ FAILED: Quality checks failed")
        sys.exit(1)
    print("✅ Stacking dataset ready")


if __name__ == "__main__":
    main()
