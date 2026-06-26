#!/usr/bin/env python3
"""
Kinora-main 主流程编排

完整端到端工作流：训练 → 预测 → Top200 → 聚类 → 分子对接

用法:
    python pipeline.py full                           # 运行完整流程
    python pipeline.py train                          # 仅训练
    python pipeline.py predict                        # 仅批量预测
    python pipeline.py top200                         # 仅 Top200 筛选
    python pipeline.py cluster                        # 仅聚类
    python pipeline.py docking                        # 仅分子对接
    python pipeline.py predict --input my_data.csv    # 指定输入文件
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a pipeline step, return True on success."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return False
    print(f"\n  COMPLETED: {name}")
    return True


def step_train(args) -> bool:
    cmd = [PYTHON, "train.py", "--task", args.task or "both"]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    return run_step("训练模型 (sklearn ExtraTreesRegressor)", cmd)


def step_predict(args) -> bool:
    input_csv = args.input or "blast500.csv"
    output_csv = args.predict_output or input_csv
    cmd = [
        PYTHON, "predict_experiment_csv.py",
        "--input", input_csv,
        "--output", output_csv,
    ]
    if args.seq_col:
        cmd += ["--seq-col", args.seq_col]
    if args.smiles_col:
        cmd += ["--smiles-col", args.smiles_col]
    return run_step(f"批量预测 ({input_csv})", cmd)


def step_top200(args) -> bool:
    input_csv = args.input or "blast500.csv"
    output_csv = args.top200_output or "top200_kcat_over_km.csv"
    cmd = [
        PYTHON, "report_top200_kcat_over_km.py",
        "--input", input_csv,
        "--top200-output", output_csv,
    ]
    return run_step(f"Top-200 筛选 ({input_csv} → {output_csv})", cmd)


def step_cluster(args) -> bool:
    input_csv = args.input or "blast500.csv"
    n_clusters = args.clusters or 3
    n_sample = args.sample or 20
    out_dir = args.cluster_out or f"outputs/pipeline_{_timestamp()}"
    cmd = [
        PYTHON, "kmer_cluster_pipeline.py",
        "--csv-input", input_csv,
        "--seq-col", args.seq_col or "Enzyme",
        "--id-col", args.id_col or "Entry",
        "--prefilter-score", input_csv,
        "--score-col", args.score_col or "Pred_kcat_over_Km",
        "--prefilter", str(args.prefilter or 200),
        "--clusters", str(n_clusters),
        "--sample", str(n_sample),
        "--sample-strategy", args.sample_strategy or "closest",
        "--out", out_dir,
    ]
    return run_step(f"k-mer 聚类 (K={n_clusters}, sample={n_sample})", cmd)


def step_docking(args) -> bool:
    csv_input = args.docking_csv or "outputs/pipeline_latest/extracted/all_extracted.fasta"
    out_dir = args.docking_out or "docking_results"
    cmd = [PYTHON, "run_docking.py", "--csv", csv_input, "--output-dir", out_dir]
    if args.skip_folding:
        cmd.append("--skip-folding")
    return run_step(f"分子对接 ({csv_input})", cmd)


def step_full(args) -> bool:
    """Run complete pipeline end-to-end."""
    steps = [
        ("训练", step_train),
        ("预测", step_predict),
        ("Top200", step_top200),
        ("聚类", step_cluster),
        ("对接", step_docking),
    ]
    for name, fn in steps:
        if not fn(args):
            print(f"\nPipeline stopped at: {name}")
            return False
    print(f"\n{'='*60}")
    print("  ALL STEPS COMPLETED")
    print(f"{'='*60}")
    return True


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Kinora-main 主流程编排 — 酶动力学预测全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py full                                  # 完整流程
  python pipeline.py train --task both                     # 仅训练
  python pipeline.py predict --input blast500.csv          # 仅预测
  python pipeline.py cluster --clusters 3 --sample 20      # 仅聚类
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline step")

    # full
    p_full = sub.add_parser("full", help="运行完整流程 (train→predict→top200→cluster→docking)")

    # train
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--task", type=str, default="both", choices=["km", "kcat", "both"])
    p_train.add_argument("--dataset", type=str, default=None)

    # predict
    p_pred = sub.add_parser("predict", help="批量预测")
    p_pred.add_argument("--input", type=str, default="blast500.csv")
    p_pred.add_argument("--predict-output", type=str, default=None)
    p_pred.add_argument("--seq-col", type=str, default=None)
    p_pred.add_argument("--smiles-col", type=str, default=None)

    # top200
    p_top = sub.add_parser("top200", help="Top-200 筛选")
    p_top.add_argument("--input", type=str, default="blast500.csv")
    p_top.add_argument("--top200-output", type=str, default=None)

    # cluster
    p_cluster = sub.add_parser("cluster", help="k-mer 聚类")
    p_cluster.add_argument("--input", type=str, default="blast500.csv")
    p_cluster.add_argument("--clusters", type=int, default=3)
    p_cluster.add_argument("--prefilter", type=int, default=200)
    p_cluster.add_argument("--sample", type=int, default=20)
    p_cluster.add_argument("--sample-strategy", type=str, default="closest")
    p_cluster.add_argument("--seq-col", type=str, default="Enzyme")
    p_cluster.add_argument("--id-col", type=str, default="Entry")
    p_cluster.add_argument("--score-col", type=str, default="Pred_kcat_over_Km")
    p_cluster.add_argument("--cluster-out", type=str, default=None)

    # docking
    p_dock = sub.add_parser("docking", help="分子对接")
    p_dock.add_argument("--docking-csv", type=str, default=None)
    p_dock.add_argument("--docking-out", type=str, default="docking_results")
    p_dock.add_argument("--skip-folding", action="store_true")

    # Shared args for full
    for p in [p_full, p_train, p_pred, p_top, p_cluster, p_dock]:
        if p is p_train:
            continue
        if p is p_dock:
            continue

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    steps_map = {
        "full": step_full,
        "train": step_train,
        "predict": step_predict,
        "top200": step_top200,
        "cluster": step_cluster,
        "docking": step_docking,
    }

    fn = steps_map[args.command]
    success = fn(args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
