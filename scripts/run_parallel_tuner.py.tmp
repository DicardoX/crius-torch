#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripts/test_parallel_tuner.py with configurable test shapes."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Global batch size used by test_parallel_tuner.py. Default: 16.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length used by test_parallel_tuner.py. Default: 1024.",
    )
    parser.add_argument(
        "unittest_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to test_parallel_tuner.py after '--'.",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def main() -> int:
    args = parse_args()
    require_positive("--batch-size", args.batch_size)
    require_positive("--seq-len", args.seq_len)

    script_dir = Path(__file__).resolve().parent
    test_script = script_dir / "test_parallel_tuner.py"
    env = os.environ.copy()
    env["CRIUS_TEST_BATCH_SIZE"] = str(args.batch_size)
    env["CRIUS_TEST_SEQ_LEN"] = str(args.seq_len)

    forwarded_args = args.unittest_args
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]

    command = [sys.executable, str(test_script), *forwarded_args]
    print(
        "Running test_parallel_tuner.py "
        f"with batch_size={args.batch_size}, seq_len={args.seq_len}"
    )
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
