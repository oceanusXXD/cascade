from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from .vllm_common import get_local_vllm_gpu_profile
_LOCAL_GENERATION_RERANK_BASE_URL_PREFIX = "local-vllm://"
_LOCAL_GENERATION_RERANK_RUNNER_CACHE: dict[tuple[str, float, int, int, int, bool], "_LocalGenerationRerankRunner"] = {}
_LOCAL_GENERATION_RERANK_RUNNER_LOCK = threading.Lock()

@dataclass(frozen=True)

class _LocalGenerationRerankRunner:
    llm: Any
    tokenizer: Any
    true_token_id: int
    false_token_id: int
    sampling_params: Any

def _score_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("reranker_base_url is required for score backend")
    return f"{normalized}/score"

def _env_has_value(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip() != ""

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}

def _local_generation_rerank_settings() -> tuple[float, int, int, int, bool]:
    gpu_profile = get_local_vllm_gpu_profile() or {}
    default_gpu_memory = float(gpu_profile.get("gpu_memory_utilization", 0.90) or 0.90)
    default_max_num_seqs = max(1, int(gpu_profile.get("rerank_max_num_seqs", 128) or 128))
    default_max_num_batched_tokens = max(0, int(gpu_profile.get("rerank_max_num_batched_tokens", 32768) or 32768))
    default_max_model_len = max(0, int(gpu_profile.get("rerank_max_model_len", 4096) or 4096))
    default_enforce_eager = bool(gpu_profile.get("rerank_enforce_eager", True))
    if _env_has_value("CASCADE_LOCAL_VLLM_RERANK_GPU_MEMORY_UTILIZATION"):
        gpu_memory_utilization = _env_float("CASCADE_LOCAL_VLLM_RERANK_GPU_MEMORY_UTILIZATION", default_gpu_memory)
    else:
        gpu_memory_utilization = default_gpu_memory
    if _env_has_value("CASCADE_LOCAL_VLLM_RERANK_MAX_NUM_SEQS"):
        max_num_seqs = _env_int("CASCADE_LOCAL_VLLM_RERANK_MAX_NUM_SEQS", default_max_num_seqs)
    else:
        max_num_seqs = default_max_num_seqs
    if _env_has_value("CASCADE_LOCAL_VLLM_RERANK_MAX_NUM_BATCHED_TOKENS"):
        max_num_batched_tokens = _env_int(
            "CASCADE_LOCAL_VLLM_RERANK_MAX_NUM_BATCHED_TOKENS",
            default_max_num_batched_tokens,
        )
    else:
        max_num_batched_tokens = default_max_num_batched_tokens
    if _env_has_value("CASCADE_LOCAL_VLLM_RERANK_MAX_MODEL_LEN"):
        max_model_len = _env_int("CASCADE_LOCAL_VLLM_RERANK_MAX_MODEL_LEN", default_max_model_len)
    else:
        max_model_len = default_max_model_len
    if _env_has_value("CASCADE_LOCAL_VLLM_RERANK_ENFORCE_EAGER"):
        enforce_eager = _env_bool("CASCADE_LOCAL_VLLM_RERANK_ENFORCE_EAGER", default_enforce_eager)
    else:
        enforce_eager = default_enforce_eager
    return (
        max(0.05, min(float(gpu_memory_utilization), 0.98)),
        max(1, int(max_num_seqs)),
        max(0, int(max_num_batched_tokens)),
        max(0, int(max_model_len)),
        bool(enforce_eager),
    )

def _should_use_local_generation_reranker(*, model: str, base_url: str) -> bool:
    normalized_base_url = str(base_url or "").strip().lower()
    if normalized_base_url.startswith(_LOCAL_GENERATION_RERANK_BASE_URL_PREFIX):
        return True
    if normalized_base_url.startswith(("http://", "https://")):
        return False
    return Path(str(model or "")).exists()

def _normalize_model_for_vllm(model: str) -> str:
    raw = str(model or "").strip()
    model_path = Path(raw)
    if raw and model_path.exists():
        return str(model_path.resolve())
    return raw

def _get_local_generation_rerank_runner(model: str) -> _LocalGenerationRerankRunner:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    normalized_model = _normalize_model_for_vllm(model)
    (
        gpu_memory_utilization,
        max_num_seqs,
        max_num_batched_tokens,
        max_model_len,
        enforce_eager,
    ) = _local_generation_rerank_settings()
    cache_key = (
        normalized_model,
        float(gpu_memory_utilization),
        int(max_num_seqs),
        int(max_num_batched_tokens),
        int(max_model_len),
        bool(enforce_eager),
    )
    with _LOCAL_GENERATION_RERANK_RUNNER_LOCK:
        cached = _LOCAL_GENERATION_RERANK_RUNNER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        tokenizer = AutoTokenizer.from_pretrained(normalized_model, trust_remote_code=True)
        true_token_ids = tokenizer.encode("yes", add_special_tokens=False)
        false_token_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(true_token_ids) != 1 or len(false_token_ids) != 1:
            raise RuntimeError(
                "local generation rerank requires single-token yes/no labels: "
                f"yes={true_token_ids} no={false_token_ids}"
            )
        true_token_id = int(true_token_ids[0])
        false_token_id = int(false_token_ids[0])
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            logprob_token_ids=[true_token_id, false_token_id],
            allowed_token_ids=[true_token_id, false_token_id],
        )
        llm_kwargs: dict[str, Any] = {
            "model": normalized_model,
            "trust_remote_code": True,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "max_num_seqs": int(max_num_seqs),
            "enforce_eager": bool(enforce_eager),
        }
        if int(max_num_batched_tokens) > 0:
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if int(max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
        runner = _LocalGenerationRerankRunner(
            llm=LLM(**llm_kwargs),
            tokenizer=tokenizer,
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            sampling_params=sampling_params,
        )
        _LOCAL_GENERATION_RERANK_RUNNER_CACHE[cache_key] = runner
        return runner

def _post_json(*, url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    resolved_api_key = str(api_key or "").strip()
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    request = Request(
        str(url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=max(1, int(timeout))) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"score request failed: status={exc.code} body={body}") from exc
    data = json.loads(body)
    if not isinstance(data, dict):
        raise RuntimeError("score response must be a JSON object")
    if "error" in data:
        raise RuntimeError(f"score request failed: {data['error']}")
    return data

def _scores_from_score_response(response: dict[str, Any], expected_count: int) -> list[float]:
    raw_items = response.get("data")
    if not isinstance(raw_items, list):
        raise RuntimeError("score response is missing data list")
    scores: list[float | None] = [None for _ in range(int(expected_count))]
    for item in raw_items:
        if not isinstance(item, dict):
            raise RuntimeError("score response data item must be an object")
        index = int(item.get("index", -1))
        if index < 0 or index >= expected_count:
            raise RuntimeError(f"score response index out of range: {index}")
        score = float(item.get("score"))
        if not math.isfinite(score):
            raise RuntimeError(f"score response contains non-finite score at index {index}")
        scores[index] = score
    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise RuntimeError(f"score response missing scores for indices: {missing}")
    return [float(score) for score in scores]

def _generation_rerank_prompt_token_ids(
    *,
    tokenizer: Any,
    instruction: str,
    query: str,
    document: str,
) -> list[int]:
    messages = [
        {"role": "system", "content": str(instruction or "").strip()},
        {"role": "query", "content": str(query or "")},
        {"role": "document", "content": str(document or "")},
    ]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", [])
    elif hasattr(token_ids, "input_ids"):
        token_ids = getattr(token_ids, "input_ids")
    return [int(token_id) for token_id in list(token_ids or [])]

def _coerce_logprob_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        raw = value.get("logprob")
    else:
        raw = getattr(value, "logprob", None)
    if raw is None:
        return None
    return float(raw)

def _extract_token_logprob(top_logprobs: Any, token_id: int, decoded_token: str) -> float | None:
    if isinstance(top_logprobs, dict):
        for key in (token_id, str(token_id)):
            if key in top_logprobs:
                value = _coerce_logprob_value(top_logprobs[key])
                if value is not None:
                    return value
        values = top_logprobs.values()
    else:
        values = top_logprobs or []
    for item in values:
        if isinstance(item, dict):
            item_token_id = item.get("token_id")
            item_decoded_token = item.get("decoded_token")
        else:
            item_token_id = getattr(item, "token_id", None)
            item_decoded_token = getattr(item, "decoded_token", None)
        token_id_matches = item_token_id is not None and int(item_token_id) == int(token_id)
        decoded_token_matches = str(item_decoded_token or "").strip() == str(decoded_token)
        if token_id_matches or decoded_token_matches:
            return _coerce_logprob_value(item)
    return None

def _yes_probability_from_logprobs(*, true_logprob: float | None, false_logprob: float | None) -> float:
    if true_logprob is None and false_logprob is None:
        raise RuntimeError("local generation rerank output missing both yes/no logprobs")
    yes_lp = float(true_logprob) if true_logprob is not None else -1.0e30
    no_lp = float(false_logprob) if false_logprob is not None else -1.0e30
    max_logprob = max(yes_lp, no_lp)
    if not math.isfinite(max_logprob):
        raise RuntimeError("local generation rerank output contains non-finite yes/no logprobs")
    yes_exp = math.exp(yes_lp - max_logprob)
    no_exp = math.exp(no_lp - max_logprob)
    denominator = yes_exp + no_exp
    if denominator <= 0.0:
        raise RuntimeError("local generation rerank output produced an invalid yes/no denominator")
    return float(yes_exp / denominator)

def _score_query_document_pairs_with_local_generation_reranker(
    *,
    query_document_pairs: Sequence[tuple[str, str]],
    instruction: str,
    model: str,
    request_batch_size: int,
) -> list[float]:
    from vllm import TokensPrompt
    runner = _get_local_generation_rerank_runner(str(model))
    pairs = [(str(query or ""), str(document or "")) for query, document in list(query_document_pairs or [])]
    effective_batch_size = max(1, int(request_batch_size))
    all_scores: list[float] = []
    for start in range(0, len(pairs), effective_batch_size):
        batch = pairs[start : start + effective_batch_size]
        prompts = [
            TokensPrompt(
                prompt_token_ids=_generation_rerank_prompt_token_ids(
                    tokenizer=runner.tokenizer,
                    instruction=str(instruction or ""),
                    query=query,
                    document=document,
                )
            )
            for query, document in batch
        ]
        outputs = list(runner.llm.generate(prompts, runner.sampling_params, use_tqdm=False))
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"local generation rerank response size mismatch: expected={len(batch)} actual={len(outputs)}"
            )
        for output in outputs:
            generated = list(getattr(output, "outputs", []) or [])
            if not generated:
                raise RuntimeError("local generation rerank output is missing completion candidates")
            logprobs = list(getattr(generated[0], "logprobs", []) or [])
            if not logprobs:
                raise RuntimeError("local generation rerank output is missing token logprobs")
            first_token_logprobs = logprobs[0]
            true_logprob = _extract_token_logprob(first_token_logprobs, runner.true_token_id, "yes")
            false_logprob = _extract_token_logprob(first_token_logprobs, runner.false_token_id, "no")
            all_scores.append(
                _yes_probability_from_logprobs(
                    true_logprob=true_logprob,
                    false_logprob=false_logprob,
                )
            )
    return all_scores

def vllm_score_query_document_pairs(
    *,
    query_document_pairs: Sequence[tuple[str, str]],
    instruction: str,
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    timeout: int = 180,
    request_batch_size: int = 64,
    max_length: int = 8192,
) -> list[float]:
    del max_length
    pairs = [(str(query or ""), str(document or "")) for query, document in list(query_document_pairs or [])]
    if not pairs:
        return []
    if _should_use_local_generation_reranker(model=str(model), base_url=str(base_url)):
        return _score_query_document_pairs_with_local_generation_reranker(
            query_document_pairs=pairs,
            instruction=str(instruction or ""),
            model=str(model),
            request_batch_size=int(request_batch_size),
        )
    del instruction
    resolved_url = _score_url(base_url)
    effective_batch_size = max(1, int(request_batch_size))
    all_scores: list[float] = []
    for start in range(0, len(pairs), effective_batch_size):
        batch = pairs[start : start + effective_batch_size]
        payload = {
            "model": str(model),
            "text_1": [query for query, _ in batch],
            "text_2": [document for _, document in batch],
            "use_activation": True,
        }
        response = _post_json(
            url=resolved_url,
            payload=payload,
            api_key=str(api_key or "EMPTY"),
            timeout=int(timeout),
        )
        all_scores.extend(_scores_from_score_response(response, len(batch)))
    return all_scores
__all__ = ["vllm_score_query_document_pairs"]
