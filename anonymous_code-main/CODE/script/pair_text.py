from __future__ import annotations

def format_pair_embedding_text(
    text: str,
    query: str,
    *,
    task_label: str = "Query",
    evidence_label: str = "Document",
    instruction_text: str = "",
) -> str:
    query_text = str(query or "").strip()
    document_text = str(text or "").strip()
    instruction_block = ""
    if str(instruction_text or "").strip():
        instruction_block = f"Instruction:\n{str(instruction_text or '').strip()}\n\n"
    return f"{instruction_block}{task_label}:\n{query_text}\n\n{evidence_label}:\n{document_text}"
__all__ = ["format_pair_embedding_text"]
