"""Frozen, label-free neighbor support and quantile grids for each SLM stage."""

import argparse
from pathlib import Path

import numpy as np

from .data import read_rows, write_json, write_rows
from .models import Embedder


def neighbor_support(vectors, predictions, *, k=20):
    import faiss

    size = len(vectors)
    support = np.full(size, 0.5)
    mass = np.zeros(size)
    if size < 2:
        return support, mass
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # Exact cosine neighbors, in blocks so the N-by-N matrix is never materialized.
    for start in range(0, size, 256):
        similarities, neighbors = index.search(vectors[start:start + 256], min(size, k + 1))
        for offset, (weights, indices) in enumerate(zip(similarities, neighbors)):
            current = start + offset
            keep = (indices != current) & (indices >= 0)
            weights = np.maximum(weights[keep][:k], 0)
            indices = indices[keep][:k]
            valid = np.array([predictions[j] in {"yes", "no"} for j in indices])
            weights *= valid
            mass[current] = weights.sum()
            if mass[current] > 0 and predictions[current] in {"yes", "no"}:
                agrees = np.array([predictions[j] == predictions[current] for j in indices])
                support[current] = float(weights @ agrees / mass[current])
    return support, mass


def score_rows(rows, predictions, vectors, *, k=20, gamma=1, epsilon=0.001):
    by_id = {row["id"]: row for row in predictions}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Prediction IDs must match the stage input IDs")
    if not rows:
        return []
    labels = [by_id[row["id"]]["prediction"] for row in rows]
    support, mass = neighbor_support(vectors, labels, k=k)
    sigma_sq = float(np.median(mass))
    rho = np.divide(mass, mass + sigma_sq, out=np.zeros_like(mass), where=(mass + sigma_sq) > 0)
    shrunk = rho * support + (1 - rho) * 0.5
    results = []
    for i, row in enumerate(rows):
        prediction = by_id[row["id"]]
        base = prediction.get("base_confidence")
        score = None
        if labels[i] in {"yes", "no"} and base is not None:
            score = float(np.log((base + epsilon) / (1 - base + epsilon)) + gamma * (2 * shrunk[i] - 1))
        results.append({**prediction, "neighbor_support": float(support[i]), "neighbor_mass": float(mass[i]),
                        "shrunk_support": float(shrunk[i]), "score": score})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embedding = commands.add_parser("embed", help="Prepare both views before starting the SLM server",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    embedding.add_argument("--input", required=True)
    embedding.add_argument("--evidence-output", required=True)
    embedding.add_argument("--document-output", required=True)
    embedding.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B", help="Model ID or local model directory")
    embedding.add_argument("--device", default="cuda", help="Embedding device")
    embedding.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    embedding.add_argument("--max-length", type=int, default=32768, help="Maximum embedding input length")
    scoring = commands.add_parser("score", help="Compute NS and freeze one stage's quantile grid",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    scoring.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--vectors", required=True)
    scoring.add_argument("--output", required=True)
    scoring.add_argument("--grid", required=True)
    scoring.add_argument("--stage2-input", help="Required for stage1: writes the maximal Stage-2 pool")
    scoring.add_argument("--k", type=int, default=20, help="Number of nearest neighbors, excluding self")
    scoring.add_argument("--gamma", type=float, default=1, help="Weight of the neighbor adjustment")
    scoring.add_argument("--epsilon", type=float, default=0.001, help="Smoothing constant for confidence log odds")
    scoring.add_argument("--grid-size", type=int, help="Quantile points; defaults to 32 for stage1, 64 for stage2")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.command == "embed":
        if not rows:
            raise ValueError("The population is empty")
        if min(args.batch_size, args.max_length) < 1:
            parser.error("batch-size and max-length must be positive")
        embedder = Embedder(args.embedding_model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
        for field, output in (("evidence", args.evidence_output), ("document", args.document_output)):
            texts = [f"Query: {row['query']}\nDocument:\n{row[field]}" for row in rows]
            vectors = embedder.encode(texts)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with Path(output).open("wb") as stream:
                np.savez(stream, ids=np.array([row["id"] for row in rows]), vectors=vectors)
        print(f"Prepared evidence and document vectors for {len(rows)} records")
        return
    if args.stage == "stage1" and not args.stage2_input:
        parser.error("stage1 requires --stage2-input")
    predictions = read_rows(args.predictions)
    with np.load(args.vectors) as saved:
        positions = {key: i for i, key in enumerate(saved["ids"].tolist())}
        vectors = saved["vectors"][[positions[row["id"]] for row in rows]]
    if args.k < 1 or args.gamma < 0 or args.epsilon <= 0:
        parser.error("k and epsilon must be positive; gamma must be nonnegative")
    scored = score_rows(rows, predictions, vectors, k=args.k, gamma=args.gamma, epsilon=args.epsilon)
    finite = [row["score"] for row in scored if row["score"] is not None]
    points = args.grid_size if args.grid_size is not None else (32 if args.stage == "stage1" else 64)
    if points < 1:
        raise ValueError("Grid size must be positive")
    thresholds = np.unique(np.quantile(finite, np.linspace(0, 1, points))).tolist() if finite else [0.0]
    write_rows(args.output, scored)
    write_json(args.grid, {"stage": args.stage, "pool_size": len(rows), "thresholds": thresholds})
    if args.stage == "stage1":
        scores = {row["id"]: row["score"] for row in scored}
        maximal_pool = [row for row in rows if scores[row["id"]] is None or scores[row["id"]] < max(thresholds)]
        write_rows(args.stage2_input, maximal_pool)
    print(f"{args.stage}: {len(scored)} scores, {len(thresholds)} thresholds")


if __name__ == "__main__":
    main()
