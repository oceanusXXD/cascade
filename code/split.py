"""Sample calibration records without replacement."""

import argparse
import math
import random
from pathlib import Path

from betting import zero_error_min_size
from common import read_json, read_rows, write_json, write_rows


def sample_calibration(rows, *, size, seed=42):
    count = len(rows)
    if not count:
        raise ValueError("The population is empty")
    if not 1 <= size <= count:
        raise ValueError("Calibration size must be between 1 and the population size")
    indices = random.Random(seed).sample(range(count), size)
    return [rows[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prepared", required=True, help="Frozen candidate family from wsr.py prepare")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, help="Override the automatic sample size before drawing records")
    parser.add_argument("--tau", type=float, default=0.9, help="Target Oracle agreement; use the same value in wsr.py fit")
    parser.add_argument("--delta", type=float, default=0.05, help="Failure budget; use the same value in wsr.py fit")
    parser.add_argument("--seed", type=int, default=42, help="Seed for one uniform ordered draw without replacement")
    args = parser.parse_args()
    rows = read_rows(args.input)
    population_size = len(rows)
    prepared = read_json(args.prepared)
    if prepared["population_ids"] != [row["id"] for row in rows]:
        parser.error("input IDs and order must match the prepared population")
    if not 0 < args.tau < 1 or not 0 < args.delta < 1:
        parser.error("tau and delta must be in (0, 1)")
    tracks = len(prepared["lambda1_grid"])
    if tracks < 1:
        parser.error("the prepared Stage-1 grid must be nonempty")
    epsilon = 1 - args.tau
    boundary = math.log(tracks / args.delta)
    minimum_size = zero_error_min_size(population_size, epsilon, boundary)
    nominal = (population_size + 50) // 100
    default_size = min(population_size, max(nominal, minimum_size if minimum_size is not None else population_size))
    size = args.size if args.size is not None else default_size
    calibration = sample_calibration(rows, size=size, seed=args.seed)
    root = Path(args.output_dir)
    write_rows(root / "calibration.jsonl", calibration)
    write_json(root / "split.json", {"population_size": population_size, "calibration_size": len(calibration),
                                    "seed": args.seed, "tau": args.tau, "delta": args.delta, "tracks": tracks,
                                    "nominal_one_percent": nominal, "zero_error_min_size": minimum_size,
                                    "size_overridden": args.size is not None})
    print(f"Population {len(rows)}; calibration {len(calibration)} -> {root}")


if __name__ == "__main__":
    main()
