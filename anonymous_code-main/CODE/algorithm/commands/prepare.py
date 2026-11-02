from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from algorithm.data.prepare import prepare_chunk_artifacts, prepare_document_artifacts
from algorithm.settings import CascadeSettings, DEFAULT_STEP1_BATCH_SIZE
from script.stream import StreamingRunWriter
from script.vllm import vllm_embed_texts


def _settings_from_args(args: Namespace) -> CascadeSettings:
    return CascadeSettings(
        command="prepare",
        mode=str(args.mode),
        queries=str(args.queries),
        documents=str(args.documents),
        ground_truth=str(getattr(args, "ground_truth", "") or ""),
        artifacts_root=str(args.artifacts_root),
        embedding_model=str(args.embedding_model),
        embedding_base_url=str(args.embedding_base_url),
        step1_batch_size=int(getattr(args, "step1_batch_size", DEFAULT_STEP1_BATCH_SIZE)),
    )


def run_from_args(args: Namespace) -> dict[str, Any]:
    settings = _settings_from_args(args)
    writer = StreamingRunWriter(Path(settings.artifacts_root))
    if settings.mode == "document":
        return prepare_document_artifacts(settings, writer=writer, embed_texts_func=vllm_embed_texts)
    if settings.mode == "chunk":
        return prepare_chunk_artifacts(settings, writer=writer, embed_texts_func=vllm_embed_texts)
    raise ValueError(f"unsupported prepare mode: {settings.mode!r}")


__all__ = ["run_from_args"]
