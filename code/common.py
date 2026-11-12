"""JSONL artifacts and the shared, normalized embedding model."""

import json
from pathlib import Path


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Embedder:
    def __init__(self, model, *, device="cuda", batch_size=16, max_length=32768):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def encode(self, texts, *, query=False):
        import numpy as np

        prompt = "Instruct: Given a predicate, retrieve relevant evidence from the document\nQuery: " if query else ""
        vectors = np.asarray(self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding vectors")
        return vectors
