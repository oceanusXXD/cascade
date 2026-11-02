from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from script.retrieval.utils.artifacts import chunk_list, validate_retrieval_candidate_row
from script.chunk_artifacts import validate_selected_chunk_row
from script.retrieval.utils.text import resolve_row_effective_query_text
from script.task_schema import resolve_default_task_text, resolve_embedding_payload, resolve_prompt_payload

@dataclass(frozen=True)

class StrategySample:
    sample_id: str
    query_id: str
    query_text: str
    document_id: str
    document_text: str
    raw_query_text: str = ""
    claim_text: str = ""
    prompt_style: str = "query_document"
    prompt_task_label: str = "Query"
    prompt_evidence_label: str = "Document"
    prompt_task_text: str = ""
    prompt_instruction_text: str = ""
    prompt_evidence_text: str = ""
    embedding_task_text: str = ""
    embedding_instruction_text: str = ""
    embedding_evidence_text: str = ""
    label: str = ""
    query_rewrites: tuple[str, ...] = ()
    hyde_hypotheses: tuple[str, ...] = ()

def build_strategy_sample(
    row: Mapping[str, Any],
    *,
    query_rewrites: tuple[str, ...] = (),
    hyde_hypotheses: tuple[str, ...] = (),
) -> StrategySample:
    query_text = str(row["query_text"])
    claim_text = str(row.get("claim_text", query_text) or query_text)
    default_task_text = resolve_row_effective_query_text(row)
    prompt_payload = resolve_prompt_payload(
        row,
        default_task_text=default_task_text,
        default_evidence_text=str(row["document_text"]),
    )
    embedding_payload = resolve_embedding_payload(
        row,
        default_task_text=default_task_text,
        default_evidence_text=str(row["document_text"]),
    )
    return StrategySample(
        sample_id=str(row["sample_id"]),
        query_id=str(row["query_id"]),
        query_text=query_text,
        document_id=str(row["document_id"]),
        document_text=str(row["document_text"]),
        raw_query_text=str(row.get("raw_query_text", query_text) or query_text),
        claim_text=claim_text,
        prompt_style=str(row.get("prompt_style", "query_document")),
        prompt_task_label=prompt_payload["task_label"],
        prompt_evidence_label=prompt_payload["evidence_label"],
        prompt_task_text=prompt_payload["task_text"],
        prompt_instruction_text=prompt_payload["instruction_text"],
        prompt_evidence_text=prompt_payload["evidence_text"],
        embedding_task_text=embedding_payload["task_text"],
        embedding_instruction_text=embedding_payload["instruction_text"],
        embedding_evidence_text=embedding_payload["evidence_text"],
        label=str(row.get("label", "")),
        query_rewrites=tuple(str(item) for item in query_rewrites),
        hyde_hypotheses=tuple(str(item) for item in hyde_hypotheses),
    )

def build_selected_chunk_row(
    sample: StrategySample,
    *,
    strategy_name: str,
    selector_version: str,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection is None:
        raise ValueError("selection is required when building a selected chunk row")
    selected_chunk_text = str(selection.get("selected_chunk_text", "") or "")
    selected_chunk_indices = [int(index) for index in list(selection.get("selected_chunk_indices", []) or [])]
    default_task_text = resolve_default_task_text(sample, fallback=sample.query_text)
    prompt_payload = resolve_prompt_payload(
        sample,
        default_task_text=default_task_text,
        default_evidence_text=sample.document_text,
    )
    embedding_payload = resolve_embedding_payload(
        sample,
        default_task_text=default_task_text,
        default_evidence_text=sample.document_text,
    )
    row = {
        "sample_id": str(sample.sample_id),
        "query_id": str(sample.query_id),
        "document_id": str(sample.document_id),
        "query_text": str(sample.query_text),
        "raw_query_text": str(sample.raw_query_text or sample.query_text),
        "claim": str(sample.claim_text or sample.query_text),
        "claim_text": str(sample.claim_text or sample.query_text),
        "prompt_style": str(sample.prompt_style or "query_document"),
        "prompt_task_label": prompt_payload["task_label"],
        "prompt_evidence_label": prompt_payload["evidence_label"],
        "prompt_task_text": prompt_payload["task_text"],
        "prompt_instruction_text": prompt_payload["instruction_text"],
        "prompt_evidence_text": prompt_payload["evidence_text"],
        "embedding_task_text": embedding_payload["task_text"],
        "embedding_instruction_text": embedding_payload["instruction_text"],
        "embedding_evidence_text": embedding_payload["evidence_text"],
        "document_text": str(sample.document_text),
        "selected_chunk_text": selected_chunk_text,
        "selected_chunk_indices": selected_chunk_indices,
        "selector_version": str(selector_version),
        "strategy": str(strategy_name),
        "label": str(sample.label or ""),
        "query_rewrites": [str(item) for item in list(sample.query_rewrites or ())],
        "hyde_hypotheses": [str(item) for item in list(sample.hyde_hypotheses or ())],
        "selected_query_variant": str(selection.get("selected_query_variant", "") or ""),
        "selected_query_source": str(selection.get("selected_query_source", "") or ""),
        "rerank_score": float(selection.get("rerank_score", 0.0) or 0.0),
        "rerank_scores": [float(score) for score in list(selection.get("rerank_scores", []) or [])],
        "lexical_score": float(selection.get("lexical_score", 0.0) or 0.0),
        "vector_score": float(selection.get("vector_score", 0.0) or 0.0),
        "rrf_score": float(selection.get("rrf_score", 0.0) or 0.0),
        "sentence_count": int(selection.get("sentence_count", 0) or 0),
        "retrieval_k": int(selection.get("retrieval_k", 0) or 0),
        "rerank_m": int(selection.get("rerank_m", 0) or 0),
        "retrieval_candidate_count": int(selection.get("retrieval_candidate_count", 0) or 0),
        "rerank_candidate_count": int(selection.get("rerank_candidate_count", 0) or 0),
        "embedding_model_call_count": int(selection.get("embedding_model_call_count", 0) or 0),
        "embedding_model_request_count": int(selection.get("embedding_model_request_count", 0) or 0),
        "embedding_model_input_count": int(selection.get("embedding_model_input_count", 0) or 0),
        "embedding_prompt_tokens": int(selection.get("embedding_prompt_tokens", 0) or 0),
        "rerank_model_call_count": int(selection.get("rerank_model_call_count", 0) or 0),
        "rerank_model_request_count": int(selection.get("rerank_model_request_count", 0) or 0),
        "rerank_model_input_count": int(selection.get("rerank_model_input_count", 0) or 0),
        "rerank_prompt_tokens": int(selection.get("rerank_prompt_tokens", 0) or 0),
        "rerank_completion_tokens": int(selection.get("rerank_completion_tokens", 0) or 0),
        "full_rerank_candidate_count": int(selection.get("full_rerank_candidate_count", 0) or 0),
        "rerank_saved_candidate_count": int(selection.get("rerank_saved_candidate_count", 0) or 0),
        "rerank_reduction_ratio": float(selection.get("rerank_reduction_ratio", 0.0) or 0.0),
    }
    validate_selected_chunk_row(row)
    return row

def build_retrieval_candidate_row(
    sample: StrategySample,
    *,
    strategy_name: str,
    selector_version: str,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    default_task_text = resolve_default_task_text(sample, fallback=sample.query_text)
    prompt_payload = resolve_prompt_payload(
        sample,
        default_task_text=default_task_text,
        default_evidence_text=sample.document_text,
    )
    embedding_payload = resolve_embedding_payload(
        sample,
        default_task_text=default_task_text,
        default_evidence_text=sample.document_text,
    )
    candidates = []
    for rank, item in enumerate(chunk_list(retrieval), start=1):
        candidates.append(
            {
                "rank": int(rank),
                "candidate_index": int(item.get("candidate_index", 0) or 0),
                "sentence_indices": [int(index) for index in list(item.get("sentence_indices", []) or [])],
                "text": str(item.get("text", "") or ""),
                "lexical_score": float(item.get("lexical_score", 0.0) or 0.0),
                "vector_score": float(item.get("vector_score", 0.0) or 0.0),
                "rrf_score": float(item.get("rrf_score", 0.0) or 0.0),
            }
        )
    row = {
        "sample_id": str(sample.sample_id),
        "query_id": str(sample.query_id),
        "document_id": str(sample.document_id),
        "query_text": str(sample.query_text),
        "raw_query_text": str(sample.raw_query_text or sample.query_text),
        "claim": str(sample.claim_text or sample.query_text),
        "claim_text": str(sample.claim_text or sample.query_text),
        "prompt_style": str(sample.prompt_style or "query_document"),
        "prompt_task_label": prompt_payload["task_label"],
        "prompt_evidence_label": prompt_payload["evidence_label"],
        "prompt_task_text": prompt_payload["task_text"],
        "prompt_instruction_text": prompt_payload["instruction_text"],
        "prompt_evidence_text": prompt_payload["evidence_text"],
        "embedding_task_text": embedding_payload["task_text"],
        "embedding_instruction_text": embedding_payload["instruction_text"],
        "embedding_evidence_text": embedding_payload["evidence_text"],
        "document_text": str(sample.document_text),
        "retrieval_query_text": str(retrieval.get("retrieval_query_text", "") or ""),
        "strategy": str(strategy_name),
        "selector_version": str(selector_version),
        "label": str(sample.label or ""),
        "query_rewrites": [str(item) for item in list(sample.query_rewrites or ())],
        "hyde_hypotheses": [str(item) for item in list(sample.hyde_hypotheses or ())],
        "selected_query_variant": str(retrieval.get("selected_query_variant", "") or ""),
        "selected_query_source": str(retrieval.get("selected_query_source", "") or ""),
        "sentence_count": int(retrieval.get("sentence_count", 0) or 0),
        "retrieval_candidate_count": int(retrieval.get("retrieval_candidate_count", 0) or 0),
        "retrieval_topk_count": int(retrieval.get("retrieval_topk_count", 0) or 0),
        "window_size": int(retrieval.get("window_size", 0) or 0),
        "embedding_model_call_count": int(retrieval.get("embedding_model_call_count", 0) or 0),
        "embedding_model_request_count": int(retrieval.get("embedding_model_request_count", 0) or 0),
        "embedding_model_input_count": int(retrieval.get("embedding_model_input_count", 0) or 0),
        "embedding_prompt_tokens": int(retrieval.get("embedding_prompt_tokens", 0) or 0),
        "chunks": candidates,
    }
    validate_retrieval_candidate_row(row)
    return row
