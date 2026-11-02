from __future__ import annotations
import argparse
from .commands.bm25 import build_bm25_candidates
from .commands.embedding import build_embedding_candidates
from .commands.rerank import rerank_retrieval_candidates
from .commands.rrf import build_rrf_candidates
from .commands.selectchunk import select_chunks_from_rerank_cache
from .settings import ChunkSelectionSettings, DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_INSTRUCTION

def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--use-hyde-cache", action="store_true")
    parser.add_argument("--use-query-rewrite-cache", action="store_true")

def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-timeout", type=int, default=180)

def _add_rerank_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--reranker-base-url", required=True)
    parser.add_argument("--reranker-api-key", default="EMPTY")
    parser.add_argument("--reranker-timeout", type=int, default=180)
    parser.add_argument("--reranker-instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--reranker-max-length", type=int, default=8192)
    parser.add_argument("--rerank-request-batch-size", type=int, default=0)

def _add_select_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rerank-results-root", required=True)
    parser.add_argument("--artifacts-root", required=True)

def build_retrieval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bm25_parser = subparsers.add_parser("bm25")
    _add_input_arguments(bm25_parser)
    embedding_parser = subparsers.add_parser("embedding")
    embedding_parser.add_argument("--bm25-candidates-root", required=True)
    embedding_parser.add_argument("--artifacts-root", required=True)
    _add_embedding_arguments(embedding_parser)
    rrf_parser = subparsers.add_parser("rrf")
    rrf_parser.add_argument("--embedding-candidates-root", required=True)
    rrf_parser.add_argument("--artifacts-root", required=True)
    rrf_parser.add_argument("--rrf-k", type=int, default=60)
    rerank_parser = subparsers.add_parser("rerank")
    rerank_parser.add_argument("--retrieval-candidates-root", required=True)
    rerank_parser.add_argument("--artifacts-root", required=True)
    _add_rerank_arguments(rerank_parser)
    select_parser = subparsers.add_parser("select")
    _add_select_arguments(select_parser)
    selectchunk_parser = subparsers.add_parser("selectchunk")
    _add_select_arguments(selectchunk_parser)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_retrieval_parser()
    args = parser.parse_args(argv)
    if args.command == "bm25":
        settings = ChunkSelectionSettings(
            queries=str(args.queries),
            documents=str(args.documents),
            ground_truth=str(getattr(args, "ground_truth", "") or ""),
            artifacts_root=str(args.artifacts_root),
            window_size=int(getattr(args, "window_size", 3)),
            rrf_k=int(getattr(args, "rrf_k", 60)),
            use_hyde_cache=bool(getattr(args, "use_hyde_cache", False)),
            use_query_rewrite_cache=bool(getattr(args, "use_query_rewrite_cache", False)),
        )
        build_bm25_candidates(settings)
        return 0
    if args.command == "embedding":
        settings = ChunkSelectionSettings(
            queries="",
            documents="",
            artifacts_root=str(args.artifacts_root),
            embedding_model=str(getattr(args, "embedding_model", "") or ""),
            embedding_base_url=str(getattr(args, "embedding_base_url", "") or ""),
            embedding_api_key=str(getattr(args, "embedding_api_key", "EMPTY") or "EMPTY"),
            embedding_timeout=int(getattr(args, "embedding_timeout", 180)),
        )
        build_embedding_candidates(settings, bm25_root=str(args.bm25_candidates_root))
        return 0
    if args.command == "rrf":
        settings = ChunkSelectionSettings(
            queries="",
            documents="",
            artifacts_root=str(args.artifacts_root),
            rrf_k=int(getattr(args, "rrf_k", 60)),
        )
        build_rrf_candidates(settings, embedding_root=str(args.embedding_candidates_root))
        return 0
    if args.command == "rerank":
        settings = ChunkSelectionSettings(
            queries="",
            documents="",
            artifacts_root=str(args.artifacts_root),
            reranker_model=str(getattr(args, "reranker_model", DEFAULT_RERANKER_MODEL) or ""),
            reranker_base_url=str(getattr(args, "reranker_base_url", "") or ""),
            reranker_api_key=str(getattr(args, "reranker_api_key", "EMPTY") or "EMPTY"),
            reranker_timeout=int(getattr(args, "reranker_timeout", 180)),
            reranker_instruction=str(getattr(args, "reranker_instruction", DEFAULT_RERANK_INSTRUCTION) or DEFAULT_RERANK_INSTRUCTION),
            reranker_max_length=int(getattr(args, "reranker_max_length", 8192)),
            rerank_request_batch_size=int(getattr(args, "rerank_request_batch_size", 0)),
        )
        rerank_retrieval_candidates(settings, retrieval_root=str(args.retrieval_candidates_root))
        return 0
    if args.command in {"select", "selectchunk"}:
        select_chunks_from_rerank_cache(
            rerank_root=str(args.rerank_results_root),
            artifacts_root=str(args.artifacts_root),
        )
        return 0
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
