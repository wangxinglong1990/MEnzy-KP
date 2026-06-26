#!/usr/bin/env python3
"""统一训练入口。

用法:
    python train/train_baseline.py --task kcat
    python train/train_baseline.py --task km
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from core.shared_smiles.vocab_builder import WordVocab  # noqa: F401, E402 — required by pickle


def main():
    parser = argparse.ArgumentParser(description='Baseline模型统一训练入口')
    parser.add_argument('--task', type=str, required=True, choices=['kcat', 'km'],
                        help='训练任务: kcat 或 km')
    args = parser.parse_args()

    # 在调用子任务前清理 sys.argv (子任务有自己的 argparse)
    import copy
    saved_argv = copy.deepcopy(sys.argv)
    sys.argv = [sys.argv[0]]

    if args.task == 'kcat':
        from train.train_kcat import main as kcat_main
        kcat_main()
    elif args.task == 'km':
        from train.train_km import main as km_main
        km_main()

    sys.argv = saved_argv


if __name__ == '__main__':
    main()
