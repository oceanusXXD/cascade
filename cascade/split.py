"""Draw an ordered calibration sample without replacement from the population."""

import argparse
import math
import random
from pathlib import Path

from .data import read_rows, write_json, write_rows


def split_dataset(rows, *, size=None, seed=42):
    count = len(rows)
    if not count:
        raise ValueError("The population is empty")
    size = min(count, max(500, math.ceil(0.01 * count))) if size is None else size
    if not 1 <= size <= count:
        raise ValueError("Calibration size must be between 1 and the population size")
    indices = random.Random(seed).sample(range(count), size)
    chosen = set(indices)
    return [rows[i] for i in indices], [row for i, row in enumerate(rows) if i not in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_rows(args.input)
    calibration, remaining = split_dataset(rows, size=args.size, seed=args.seed)
    root = Path(args.output_dir)
    write_rows(root / "population.jsonl", rows)
    write_rows(root / "calibration.jsonl", calibration)
    write_rows(root / "remaining.jsonl", remaining)
    write_json(root / "split.json", {"population_size": len(rows), "calibration_size": len(calibration), "seed": args.seed})
    print(f"Population {len(rows)}; calibration {len(calibration)}; remaining {len(remaining)}")


if __name__ == "__main__":
    main()
