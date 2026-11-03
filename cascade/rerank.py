"""Rerank RRF candidates, keep largest-gap prefix (cap 5), merge sentences."""

import argparse

import numpy as np

from .data import read_json, read_rows, write_rows
from .models import Reranker


def select_evidence(row, reranker, *, max_chunks=5):
    candidates = row["candidates"]
    scores = np.asarray(reranker.score(row["query"], [chunk["text"] for chunk in candidates]))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_json(args.config)
    reranker = Reranker(config["reranker"])
    rows = read_rows(args.input)
    write_rows(args.output, (select_evidence(row, reranker, max_chunks=config["max_chunks"]) for row in rows))
    print(f"Evidence: {len(rows)} records -> {args.output}")


if __name__ == "__main__":
    main()
