"""Rerank and merge evidence."""

import argparse
import json
from pathlib import Path

import numpy as np


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


class Reranker:
    def __init__(self, model, *, device="cuda", batch_size=8, max_length=8192):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model, device=device, max_length=max_length,
        )
        self.batch_size = batch_size

    def score(self, query, chunks):
        import torch

        return self.model.predict(
            [(query, chunk) for chunk in chunks], batch_size=self.batch_size,
            activation_fn=torch.nn.Sigmoid(), show_progress_bar=False,
        ).tolist()


def select_evidence(row, reranker, *, max_chunks=5):
    candidates = row["candidates"]
    if not candidates:
        raise ValueError(f"{row['id']}: no candidate windows")
    scores = np.asarray(reranker.score(row["query"], [chunk["text"] for chunk in candidates]), dtype=float).reshape(-1)
    if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
        raise ValueError(f"{row['id']}: invalid reranker scores")
    order = np.argsort(-scores, kind="stable")
    gaps = np.diff(-scores[order])
    count = int(np.argmax(gaps)) + 1 if len(gaps) and np.max(gaps) > 0 else len(order)
    selected = order[:min(count, max_chunks)]
    sentence_indices = sorted({j for i in selected for j in candidates[i]["sentence_indices"]})
    return {**{key: value for key, value in row.items() if key not in {"sentences", "candidates"}},
            "evidence": " ".join(row["sentences"][j] for j in sentence_indices),
            "selected_sentence_indices": sentence_indices,
            "selected_chunks": [{**candidates[i], "rerank_score": float(scores[i])} for i in selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B", help="Model ID or local model directory")
    parser.add_argument("--device", default="cuda", help="Reranker device")
    parser.add_argument("--batch-size", type=int, default=8, help="Reranker batch size")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum reranker input length")
    parser.add_argument("--max-chunks", type=int, default=5, help="Cap on selected windows")
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.max_chunks) < 1:
        parser.error("batch-size, max-length, and max-chunks must be positive")
    reranker = Reranker(args.reranker_model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    rows = read_rows(args.input)
    write_rows(args.output, (select_evidence(row, reranker, max_chunks=args.max_chunks) for row in rows))
    print(f"Evidence: {len(rows)} records -> {args.output}")


if __name__ == "__main__":
    main()
