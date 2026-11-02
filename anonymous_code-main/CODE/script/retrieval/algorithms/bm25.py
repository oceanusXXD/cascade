from __future__ import annotations
import math
import re
from typing import Any, Sequence
from script.retrieval.utils.rows import StrategySample
from script.retrieval.utils.text import effective_query_text, lexical_queries
from script.retrieval.utils.text import sentence_windows
from script.text import split_into_sentences
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
BM25_CANDIDATE_STRATEGY = "bm25_sentence_window"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]

class BM25Index:
    def __init__(self, texts: Sequence[str]) -> None:
        self._token_rows = [_tokenize(text) for text in texts]
        self._doc_count = len(self._token_rows)
        self._avg_len = (
            sum(len(tokens) for tokens in self._token_rows) / float(self._doc_count)
            if self._doc_count
            else 0.0
        )
        self._doc_freq: dict[str, int] = {}
        for tokens in self._token_rows:
            for token in set(tokens):
                self._doc_freq[token] = int(self._doc_freq.get(token, 0)) + 1
    def _idf(self, token: str) -> float:
        doc_freq = int(self._doc_freq.get(token, 0))
        return math.log(1.0 + (self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
    def scores(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [0.0 for _ in self._token_rows]
        scores: list[float] = []
        for tokens in self._token_rows:
            doc_len = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = int(counts.get(token, 0)) + 1
            score = 0.0
            for token in query_tokens:
                freq = float(counts.get(token, 0))
                if freq <= 0.0:
                    continue
                numerator = freq * (k1 + 1.0)
                denominator = freq + k1 * (1.0 - b + b * (doc_len / self._avg_len if self._avg_len else 1.0))
                score += self._idf(token) * (numerator / denominator)
            scores.append(float(score))
        return scores
    def top(self, query: str) -> tuple[int, float]:
        scored = self.scores(query)
        if not scored:
            return -1, 0.0
        best_index = max(range(len(scored)), key=lambda index: scored[index])
        return int(best_index), float(scored[best_index])

def build_bm25_candidate_row(
    sample: StrategySample,
    *,
    window_size: int,
    rrf_k: int,
    selector_version: str,
) -> dict[str, Any]:
    normalized_document = str(sample.document_text or "").strip()
    sentences = split_into_sentences(normalized_document) or ([normalized_document] if normalized_document else [])
    candidates = [
        candidate
        for candidate in sentence_windows(sentences, window_size=int(window_size))
        if str(candidate.text or "").strip()
    ]
    if candidates:
        index = BM25Index([candidate.text for candidate in candidates])
        lexical_scores = [0.0 for _ in candidates]
        for _, query_variant in lexical_queries(sample):
            scores = index.scores(query_variant)
            for idx, score in enumerate(scores):
                lexical_scores[idx] = max(float(lexical_scores[idx]), float(score))
    else:
        lexical_scores = []
    ordered = sorted(range(len(candidates)), key=lambda idx: (-float(lexical_scores[idx]), idx))
    chunks = [
        {
            "rank": int(rank),
            "candidate_index": int(candidate_index),
            "sentence_indices": list(candidates[int(candidate_index)].sentence_indices),
            "text": str(candidates[int(candidate_index)].text or ""),
            "lexical_score": float(lexical_scores[int(candidate_index)]),
            "bm25_score": float(lexical_scores[int(candidate_index)]),
        }
        for rank, candidate_index in enumerate(ordered, start=1)
    ]
    return {
        "sample_id": str(sample.sample_id),
        "query_id": str(sample.query_id),
        "document_id": str(sample.document_id),
        "query_text": str(sample.query_text),
        "raw_query_text": str(sample.raw_query_text or sample.query_text),
        "claim": str(sample.claim_text or sample.query_text),
        "claim_text": str(sample.claim_text or sample.query_text),
        "prompt_style": str(sample.prompt_style or "query_document"),
        "prompt_task_label": str(sample.prompt_task_label),
        "prompt_evidence_label": str(sample.prompt_evidence_label),
        "prompt_task_text": str(sample.prompt_task_text),
        "prompt_instruction_text": str(sample.prompt_instruction_text),
        "prompt_evidence_text": str(sample.prompt_evidence_text),
        "embedding_task_text": str(sample.embedding_task_text),
        "embedding_instruction_text": str(sample.embedding_instruction_text),
        "embedding_evidence_text": str(sample.embedding_evidence_text),
        "document_text": str(sample.document_text),
        "retrieval_query_text": str(effective_query_text(sample)),
        "strategy": BM25_CANDIDATE_STRATEGY,
        "selector_version": str(selector_version),
        "label": str(sample.label or ""),
        "query_rewrites": [str(item) for item in list(sample.query_rewrites or ())],
        "hyde_hypotheses": [str(item) for item in list(sample.hyde_hypotheses or ())],
        "selected_query_variant": str(effective_query_text(sample)),
        "selected_query_source": "bm25",
        "sentence_count": int(len(sentences)),
        "retrieval_candidate_count": int(len(candidates)),
        "retrieval_topk_count": int(len(chunks)),
        "window_size": int(window_size),
        "rrf_k": int(rrf_k),
        "chunks": chunks,
    }
__all__ = ["BM25_CANDIDATE_STRATEGY", "BM25Index", "build_bm25_candidate_row"]
