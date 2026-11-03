"""BM25 and embedding rank fusion."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from blingfire import text_to_sentences


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


class Embedder:
    def __init__(self, model, *, device="cuda", batch_size=16, max_length=32768):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def encode(self, texts, *, query=False):
        prompt = "Instruct: Given a predicate, retrieve relevant evidence from the document\nQuery: " if query else ""
        return np.asarray(self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B", help="Model ID or local model directory")
    parser.add_argument("--device", default="cuda", help="Embedding device, e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=32768, help="Maximum embedding input length")
    parser.add_argument("--window-size", type=int, default=3, help="Sentences per window; stride is 1")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank denominator constant")
    parser.add_argument("--buffer", type=int, default=5, help="Extra windows after the largest RRF gap")
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.window_size, args.rrf_k) < 1 or args.buffer < 0:
        parser.error("sizes and rrf-k must be positive; buffer must be nonnegative")
    embedder = Embedder(args.embedding_model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    rows = read_rows(args.input)
    write_rows(args.output, (retrieve(row, embedder, window_size=args.window_size, rrf_k=args.rrf_k,
                                     buffer=args.buffer) for row in rows))
    print(f"RRF: {len(rows)} records -> {args.output}")


if __name__ == "__main__":
    main()
