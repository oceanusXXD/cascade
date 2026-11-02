from __future__ import annotations
import argparse
from argparse import Namespace
from algorithm.commands import prepare, step1, step2, step3
from algorithm.settings import DEFAULT_STEP1_BATCH_SIZE, DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE

def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("document", "chunk"), required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--embedding-model", required=True)

def _add_step1_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--step1-chat-model", required=True)
    parser.add_argument("--step1-chat-base-url", required=True)
    parser.add_argument("--step1-chat-local-path", default="")
    parser.add_argument("--step1-max-parallel-requests", type=int, default=0)
    parser.add_argument(
        "--step1-chunk-score-temperature",
        type=float,
        default=DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE,
    )

def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--step1-batch-size", type=int, default=DEFAULT_STEP1_BATCH_SIZE)

def _add_step2_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--step2-chat-model", default="")
    parser.add_argument("--step2-chat-base-url", default="")
    parser.add_argument("--step2-chat-local-path", default="")
    parser.add_argument("--step2-max-parallel-requests", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--delta-w", type=float, default=0.05)
    parser.add_argument("--calibration-ratio", type=float, default=0.0)
    parser.add_argument("--combiner-source", choices=("source1", "source2"), default="source1")
    parser.add_argument("--combiner-theta-grid", default="")

def _add_step3_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--step3-chat-model", default="")
    parser.add_argument("--step3-chat-base-url", default="")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="algorithm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    _add_prepare_arguments(prepare_parser)
    step1_parser = subparsers.add_parser("step1")
    _add_step1_arguments(step1_parser)
    step2_parser = subparsers.add_parser("step2")
    _add_step2_arguments(step2_parser)
    step3_parser = subparsers.add_parser("step3")
    _add_step3_arguments(step3_parser)
    return parser

def run_from_args(args: Namespace) -> int:
    command = str(args.command or "").strip().lower()
    if command == "prepare":
        prepare.run_from_args(args)
        return 0
    if command == "step1":
        step1.run_from_args(args)
        return 0
    if command == "step2":
        step2.run_from_args(args)
        return 0
    if command == "step3":
        step3.run_from_args(args)
        return 0
    raise ValueError(f"unsupported algorithm command: {command!r}")

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)
if __name__ == "__main__":
    raise SystemExit(main())
