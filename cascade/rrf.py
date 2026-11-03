"""Three-sentence windows -> BM25 + embedding ranks -> largest gap + 5."""

import argparse
import math
import re
from collections import Counter

import numpy as np
from blingfire import text_to_sentences

from .data import read_json, read_rows, write_rows
from .models import Embedder


def bm25_scores(query, chunks):
    tokens = [re.findall(r"\w+", text.lower()) for text in chunks]
    frequencies = [Counter(words) for words in tokens]
    document_frequency = Counter(word for words in tokens for word in set(words))
    average_length = sum(map(len, tokens)) / len(tokens) or 1
    scores = np.zeros(len(chunks))
    for word in re.findall(r"\w+", query.lower()):
        df = document_frequency[word]
        idf = math.log1p((len(chunks) - df + 0.5) / (df + 0.5))
        for i, counts in enumerate(frequencies):
            tf = counts[word]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(tokens[i]) / average_length))
    return scores


def rank_positions(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def retrieve(row, embedder, *, window_size=3, rrf_k=60, buffer=5):
    sentences = text_to_sentences(row["document"]).splitlines()
    sentences = [text.strip() for text in sentences if text.strip()]
    if not sentences:
        raise ValueError(f"{row['id']}: document is empty")
    windows = [list(range(start, min(start + window_size, len(sentences))))
               for start in range(max(1, len(sentences) - window_size + 1))]
    chunks = [" ".join(sentences[i] for i in indices) for indices in windows]
    lexical = bm25_scores(row["query"], chunks)
    query_vector = embedder.encode([row["query"]], query=True)[0]
    dense = embedder.encode(chunks) @ query_vector
    scores = 1 / (rrf_k + rank_positions(lexical)) + 1 / (rrf_k + rank_positions(dense))
    order = np.argsort(-scores, kind="stable")
    gaps = np.diff(-scores[order])
    count = int(np.argmax(gaps)) + 1 + buffer if len(gaps) and np.max(gaps) > 0 else len(order)
    candidates = [{"sentence_indices": windows[i], "text": chunks[i],
                   "bm25_score": float(lexical[i]), "embedding_score": float(dense[i]),
                   "rrf_score": float(scores[i])} for i in order[:count]]
    return {**row, "sentences": sentences, "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_json(args.config)
    embedder = Embedder(config["embedding"])
    rows = read_rows(args.input)
    write_rows(args.output, (retrieve(row, embedder, **config["retrieval"]) for row in rows))
    print(f"RRF: {len(rows)} records -> {args.output}")


if __name__ == "__main__":
    main()
