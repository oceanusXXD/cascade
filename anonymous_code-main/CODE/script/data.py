from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from script.task_schema import first_nonempty_text
from .ids import build_sample_id
from .labels import normalize_binary_label

def _read_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_documents(path: str | Path) -> list[dict[str, str]]:
    payload = _read_payload(path)
    rows = payload if isinstance(payload, list) else list(payload.get("documents", []) or [])
    documents: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        document_id = str(row.get("document_id", row.get("doc_id", row.get("review_id", row.get("id", "")))) or "").strip()
        document_text = str(
            row.get("document", row.get("original_text", row.get("text", row.get("review", "")))) or ""
        ).strip()
        if document_id and document_text:
            meta = row.get("meta", {})
            claim_text = ""
            if isinstance(meta, dict):
                claim_text = str(meta.get("claim", "") or "").strip()
            if not claim_text:
                claim_text = str(row.get("claim", row.get("claim_text", "")) or "").strip()
            documents.append(
                {
                    "document_id": document_id,
                    "document_text": document_text,
                    "query_id": str(row.get("query_id", "") or "").strip(),
                    "query_text": str(row.get("query", row.get("query_text", "")) or "").strip(),
                    "raw_query_text": str(row.get("raw_query_text", row.get("query", row.get("query_text", ""))) or "").strip(),
                    "claim_text": claim_text,
                    "label": normalize_binary_label(row.get("parsed_answer", row.get("answer", row.get("label", "")))),
                    "stage2_prediction": normalize_binary_label(row.get("stage2_prediction", "")),
                    "stage2_routing_score": row.get("stage2_routing_score", ""),
                    "prompt_style": str(row.get("prompt_style", "") or "").strip(),
                    "prompt_task_label": str(row.get("prompt_task_label", "") or "").strip(),
                    "prompt_evidence_label": str(row.get("prompt_evidence_label", "") or "").strip(),
                    "prompt_task_text": str(row.get("prompt_task_text", "") or "").strip(),
                    "prompt_instruction_text": str(row.get("prompt_instruction_text", "") or "").strip(),
                    "prompt_evidence_text": str(row.get("prompt_evidence_text", "") or "").strip(),
                    "embedding_task_text": str(row.get("embedding_task_text", "") or "").strip(),
                    "embedding_instruction_text": str(row.get("embedding_instruction_text", "") or "").strip(),
                    "embedding_evidence_text": str(row.get("embedding_evidence_text", "") or "").strip(),
                    "meta": meta if isinstance(meta, dict) else {},
                }
            )
    return documents

def load_queries(path: str | Path) -> list[dict[str, str]]:
    payload = _read_payload(path)
    rows = (
        payload
        if isinstance(payload, list)
        else list(
            payload.get("queries", [])
            or payload.get("filter_queries", [])
            or payload.get("rows", [])
            or []
        )
    )
    queries: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id", row.get("id", "")) or "").strip()
        query_text = str(row.get("query", row.get("text", "")) or "").strip()
        if query_id and query_text:
            queries.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "raw_query_text": str(row.get("raw_query_text", query_text) or query_text).strip(),
                    "prompt_style": str(row.get("prompt_style", "") or "").strip(),
                    "prompt_task_label": str(row.get("prompt_task_label", "") or "").strip(),
                    "prompt_evidence_label": str(row.get("prompt_evidence_label", "") or "").strip(),
                    "claim_text": str(row.get("claim_text", "") or "").strip(),
                    "prompt_task_text": str(row.get("prompt_task_text", "") or "").strip(),
                    "prompt_instruction_text": str(row.get("prompt_instruction_text", "") or "").strip(),
                    "prompt_evidence_text": str(row.get("prompt_evidence_text", "") or "").strip(),
                    "embedding_task_text": str(row.get("embedding_task_text", "") or "").strip(),
                    "embedding_instruction_text": str(row.get("embedding_instruction_text", "") or "").strip(),
                    "embedding_evidence_text": str(row.get("embedding_evidence_text", "") or "").strip(),
                }
            )
    return queries

def load_ground_truth_index(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    payload = _read_payload(file_path)
    rows = payload if isinstance(payload, list) else list(payload.get("results", []) or payload.get("rows", []) or [])
    labels: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id", row.get("id", "")) or "").strip()
        document_id = str(row.get("document_id", row.get("review_id", row.get("doc_id", ""))) or "").strip()
        label = normalize_binary_label(row.get("parsed_answer", row.get("answer", row.get("label", ""))))
        if query_id and document_id and label:
            labels[build_sample_id(document_id, query_id)] = label
    return labels

def build_sample_rows(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    ground_truth_path: str | Path | None = None,
) -> list[dict[str, str]]:
    documents = load_documents(documents_path)
    queries = load_queries(queries_path)
    labels = load_ground_truth_index(ground_truth_path)
    pairwise_documents = [
        document
        for document in documents
        if str(document.get("query_id", "") or "").strip()
        and str(document.get("query_text", "") or "").strip()
    ]
    if pairwise_documents:
        if len(pairwise_documents) != len(documents):
            raise ValueError("pairwise document rows cannot be mixed with cartesian document rows")
        rows = []
        for document in pairwise_documents:
            query = {
                "query_id": str(document["query_id"]),
                "query_text": str(document["query_text"]),
                "raw_query_text": str(document.get("raw_query_text", document["query_text"]) or document["query_text"]),
                "prompt_style": "",
                "prompt_task_label": "",
                "prompt_evidence_label": "",
                "claim_text": "",
                "prompt_task_text": "",
                "prompt_instruction_text": "",
                "prompt_evidence_text": "",
                "embedding_task_text": "",
                "embedding_instruction_text": "",
                "embedding_evidence_text": "",
            }
            sample_id = build_sample_id(document["document_id"], query["query_id"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "document_id": document["document_id"],
                    "document_text": document["document_text"],
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "raw_query_text": query["raw_query_text"],
                    "claim_text": first_nonempty_text(
                        document.get("claim_text", ""),
                        document.get("prompt_task_text", ""),
                        query["query_text"],
                    ),
                    "prompt_style": str(
                        document.get("prompt_style", "")
                        or query.get("prompt_style", "")
                        or "query_document"
                    ),
                    "prompt_task_label": str(
                        document.get("prompt_task_label", "")
                        or query.get("prompt_task_label", "")
                        or "Query"
                    ),
                    "prompt_evidence_label": str(
                        document.get("prompt_evidence_label", "")
                        or query.get("prompt_evidence_label", "")
                        or "Document"
                    ),
                    "prompt_task_text": first_nonempty_text(
                        document.get("prompt_task_text", ""),
                        document.get("claim_text", ""),
                        query["query_text"],
                    ),
                    "prompt_instruction_text": first_nonempty_text(
                        document.get("prompt_instruction_text", ""),
                        query.get("prompt_instruction_text", ""),
                    ),
                    "prompt_evidence_text": first_nonempty_text(
                        document.get("prompt_evidence_text", ""),
                        document["document_text"],
                    ),
                    "embedding_task_text": first_nonempty_text(
                        document.get("embedding_task_text", ""),
                        document.get("prompt_task_text", ""),
                        document.get("claim_text", ""),
                        query["query_text"],
                    ),
                    "embedding_instruction_text": first_nonempty_text(
                        document.get("embedding_instruction_text", ""),
                        document.get("prompt_instruction_text", ""),
                    ),
                    "embedding_evidence_text": first_nonempty_text(
                        document.get("embedding_evidence_text", ""),
                        document.get("prompt_evidence_text", ""),
                        document["document_text"],
                    ),
                    "label": labels.get(sample_id, str(document.get("label", "") or "")),
                    "stage2_prediction": str(document.get("stage2_prediction", "") or ""),
                    "stage2_routing_score": document.get("stage2_routing_score", ""),
                }
            )
        rows.sort(key=lambda row: row["sample_id"])
        return rows
    rows = [
        {
            "sample_id": build_sample_id(document["document_id"], query["query_id"]),
            "document_id": document["document_id"],
            "document_text": document["document_text"],
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "raw_query_text": str(query.get("raw_query_text", query["query_text"]) or query["query_text"]),
            "claim_text": first_nonempty_text(
                query.get("claim_text", ""),
                document.get("claim_text", ""),
                document.get("prompt_task_text", ""),
                query.get("prompt_task_text", ""),
            ),
            "prompt_style": str(
                query.get("prompt_style", "")
                or document.get("prompt_style", "")
                or "query_document"
            ),
            "prompt_task_label": str(
                query.get("prompt_task_label", "")
                or document.get("prompt_task_label", "")
                or "Query"
            ),
            "prompt_evidence_label": str(
                document.get("prompt_evidence_label", "")
                or query.get("prompt_evidence_label", "")
                or "Document"
            ),
            "prompt_task_text": first_nonempty_text(
                document.get("prompt_task_text", ""),
                query.get("prompt_task_text", ""),
                document.get("claim_text", ""),
                query["query_text"],
            ),
            "prompt_instruction_text": first_nonempty_text(
                query.get("prompt_instruction_text", ""),
                document.get("prompt_instruction_text", ""),
            ),
            "prompt_evidence_text": first_nonempty_text(
                document.get("prompt_evidence_text", ""),
                document["document_text"],
            ),
            "embedding_task_text": first_nonempty_text(
                document.get("embedding_task_text", ""),
                query.get("embedding_task_text", ""),
                document.get("prompt_task_text", ""),
                query.get("prompt_task_text", ""),
                document.get("claim_text", ""),
                query["query_text"],
            ),
            "embedding_instruction_text": first_nonempty_text(
                query.get("embedding_instruction_text", ""),
                document.get("embedding_instruction_text", ""),
                query.get("prompt_instruction_text", ""),
                document.get("prompt_instruction_text", ""),
            ),
            "embedding_evidence_text": first_nonempty_text(
                document.get("embedding_evidence_text", ""),
                document.get("prompt_evidence_text", ""),
                document["document_text"],
            ),
            "label": labels.get(build_sample_id(document["document_id"], query["query_id"]), ""),
            "stage2_prediction": str(document.get("stage2_prediction", "") or ""),
            "stage2_routing_score": document.get("stage2_routing_score", ""),
        }
        for query in queries
        for document in documents
    ]
    rows.sort(key=lambda row: row["sample_id"])
    return rows
