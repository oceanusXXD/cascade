"""Sample calibration records without replacement."""

import argparse
import json
import math
import random
from pathlib import Path


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: each record needs a unique, nonempty string id")
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
