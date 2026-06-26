#!/usr/bin/env python3
"""Build benchmark table comparing all models across all metrics.

Auto-discovers artifacts/*/{task}/*metrics.json files.

Usage:
    python scripts/build_benchmark_table.py --task kcat
    python scripts/build_benchmark_table.py --task km

Output:
    reports/BENCHMARK_SUMMARY.csv
"""
import argparse, json, glob
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def discover_metrics(task: str) -> list[dict]:
    """Discover all metrics.json files for a given task."""
    results = []
    skip_dirs = {"stacking", "stacking_v2"}

    for path in sorted(ARTIFACTS.rglob(f"{task}/*metrics.json")):
        model_dir = path.parents[1].name
        model_name = path.parents[1].name

        with open(path) as f:
            data = json.load(f)

        # Normalize metric names
        entry = {"model": model_name}
        entry["r2"] = data.get("r2") or data.get("test_r2")
        entry["mae"] = data.get("mae") or data.get("test_mae")
        mse = data.get("mse") or data.get("test_mse")
        if mse is not None:
            entry["rmse"] = round(float(mse) ** 0.5, 4)
        else:
            entry["rmse"] = data.get("rmse")
        entry["pearson_r"] = data.get("pearson_r")
        entry["spearman_r"] = data.get("spearman_r")

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="Build benchmark comparison table")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"])
    args = parser.parse_args()

    models = discover_metrics(args.task)
    if not models:
        print(f"No metrics found for {args.task}")
        return

    df = pd.DataFrame(models)
    # Sort by R² descending
    df = df.sort_values("r2", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # rank starting at 1

    cols = ["model", "r2", "mae", "rmse", "pearson_r", "spearman_r"]
    display_cols = [c for c in cols if c in df.columns]
    display = df[display_cols].copy()

    print(f"\nBenchmark Summary — {args.task}")
    print("=" * 70)
    print(display.to_string(index=True, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))

    out_path = REPORTS / f"BENCHMARK_SUMMARY_{args.task}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
