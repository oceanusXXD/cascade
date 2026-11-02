

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from algorithm.core.wsr import betting_log_wealth, betting_population_upper_bound
from algorithm.common.runtime import (
    STEP1_ROUTABLE_SIGNAL_QUALITIES,
    Sample,
)
from script.ids import split_sample_id
from algorithm.common.metrics import (
    neighbor_support_summary_from_samples,
    numeric_summary,
    signal_quality_distribution_from_samples,
)
from script.labels import normalize_binary_label
from script.model_usage import estimate_text_tokens


def stable_key(value: Any) -> str:

    item = value
    if hasattr(item, "item"):
        try:
            item = item.item()
        except Exception:
            pass
    return str(item or "").strip()


class NeighborSupportBank:


    def __init__(
        self,
        *,
        embedding_model: str = "",
        enable_neighbor_support: bool = True,
        stage: str = "stage2",
        embedding_attr: str = "_stage2_pair_embedding",
        prediction_attr: str = "stage1_prediction",
        support_attr: str = "",
        k_eff_attr: str = "",
        bar_w_attr: str = "",
        fallback_embedding_attr: str = "",
        top_k: int | None = None,
    ):

        self.enabled = bool(enable_neighbor_support)
        self.embedding_model = str(embedding_model or "").strip()
        self.stage = str(stage or "").strip() or "stage2"
        self.embedding_attr = str(embedding_attr or "").strip() or "_stage2_pair_embedding"
        self.prediction_attr = str(prediction_attr or "stage1_prediction").strip()
        self.support_attr = str(support_attr or f"{self.stage}_neighbor_support").strip()
        self.k_eff_attr = str(k_eff_attr or f"{self.stage}_neighbor_k_eff").strip()
        self.bar_w_attr = str(bar_w_attr or f"{self.stage}_neighbor_bar_w").strip()
        self.fallback_embedding_attr = str(fallback_embedding_attr or "").strip()
        self.top_k = self._resolve_top_k(top_k)

        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._preds = np.asarray([], dtype=object)
        self._sample_index_by_id: Dict[str, int] = {}
        self._last_pair_cache_read: Dict[str, Any] = {}
        self._fitted = False

    @staticmethod
    def _resolve_top_k(value: int | None = None) -> int:
        raw_value = value if value is not None else os.environ.get("CASCADE_NS_TOP_K", "32")
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            parsed = 32
        return max(1, parsed)

    def reset(self) -> None:


        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._preds = np.asarray([], dtype=object)
        self._sample_index_by_id = {}
        self._last_pair_cache_read = {}
        self._fitted = False

    def _sample_stage_prediction(self, sample: Sample) -> str:

        pred = str(getattr(sample, self.prediction_attr, "") or "").strip().lower()
        return pred if pred in {"yes", "no"} else ""

    @staticmethod
    def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:


        if arr.size == 0:
            return arr.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        return (arr / norms).astype(np.float32)

    def _sample_cached_pair_vector(self, sample: Sample) -> Optional[np.ndarray]:

        value = getattr(sample, self.embedding_attr, None)
        source_attr = self.embedding_attr
        if not isinstance(value, np.ndarray) and self.fallback_embedding_attr:
            value = getattr(sample, self.fallback_embedding_attr, None)
            source_attr = self.fallback_embedding_attr
        if not isinstance(value, np.ndarray):
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        setattr(sample, f"_{self.stage}_neighbor_embedding_source", source_attr)
        return arr

    def _embeddings_from_prepared_samples(self, samples: List[Sample]) -> np.ndarray:

        if not samples:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.enabled:
            return np.zeros((0, 0), dtype=np.float32)

        cached_pair_vectors = [
            self._sample_cached_pair_vector(sample)
            for sample in samples
        ]
        if cached_pair_vectors and all(vector is not None for vector in cached_pair_vectors):
            self._last_pair_cache_read = {
                "source": f"prepared_artifact_{self.stage}_pair_embeddings",
                "stage": str(self.stage),
                "embedding_attr": str(self.embedding_attr),
                "sample_count": int(len(samples)),
            }
            return self._normalize_embeddings(
                np.vstack(
                    [
                        np.asarray(vector, dtype=np.float32)
                        for vector in cached_pair_vectors
                        if vector is not None
                    ]
                )
            )

        raise RuntimeError(
            f"{self.stage} neighbor support requires prepared {self.stage} pair embeddings; "
            "missing or invalid prepared embedding rows were found"
        )

    def fit(self, samples: List[Sample]) -> None:

        self.reset()
        if not self.enabled:
            return
        bank_samples = [
            sample
            for sample in samples
            if self._sample_stage_prediction(sample) in {"yes", "no"}
        ]
        if not bank_samples:
            return

        embeddings = self._embeddings_from_prepared_samples(bank_samples)
        if embeddings.size == 0:
            return

        self._embeddings = embeddings
        self._preds = np.asarray(
            [self._sample_stage_prediction(sample) for sample in bank_samples],
            dtype=object,
        )
        self._sample_index_by_id = {
            str(sample.sample_id): idx
            for idx, sample in enumerate(bank_samples)
            if str(sample.sample_id or "").strip()
        }
        self._fitted = bool(self._embeddings.shape[0] > 0)

    def _support_from_similarities(self, similarities: np.ndarray, pred: str) -> tuple[float, float, float]:

        if pred not in {"yes", "no"} or similarities.size == 0:
            return 0.5, 0.0, 0.0
        finite_mask = np.isfinite(similarities)
        if not np.any(finite_mask):
            return 0.5, 0.0, 0.0

        finite_indices = np.flatnonzero(finite_mask)
        order = np.lexsort((finite_indices, -similarities[finite_indices]))
        top_indices = finite_indices[order[: min(self.top_k, finite_indices.size)]]
        if top_indices.size == 0:
            return 0.5, 0.0, 0.0
        weights = np.clip(similarities[top_indices], a_min=0.0, a_max=None).astype(np.float64)
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            return 0.5, 0.0, 0.0
        support_weight = float(np.sum(weights[self._preds[top_indices] == pred]))
        neighbor_count = float(top_indices.size)
        bar_w = float(total_weight / neighbor_count)
        support = float(max(0.0, min(1.0, support_weight / total_weight)))
        return support, neighbor_count, bar_w

    @staticmethod
    def _attach_batch_size() -> int:
        raw_value = os.environ.get("CASCADE_NS_BATCH_SIZE", "8192")
        try:
            value = int(raw_value)
        except ValueError:
            value = 8192
        return max(256, value)

    def attach(self, samples: List[Sample], *, exclude_self: bool = False) -> None:


        for sample in samples:
            setattr(sample, self.support_attr, 0.5)
            setattr(sample, self.k_eff_attr, 0.0)
            setattr(sample, self.bar_w_attr, 0.0)

        if not self.enabled or not samples or not self._fitted:
            return

        target_samples = [
            sample
            for sample in samples
            if self._sample_stage_prediction(sample) in {"yes", "no"}
        ]
        if not target_samples:
            return

        query_embeddings = self._embeddings_from_prepared_samples(target_samples)
        if query_embeddings.size == 0:
            return

        target_preds = np.asarray(
            [self._sample_stage_prediction(sample) for sample in target_samples],
            dtype=object,
        )
        supports = np.full(len(target_samples), 0.5, dtype=np.float64)
        k_effs = np.zeros(len(target_samples), dtype=np.float64)
        bar_ws = np.zeros(len(target_samples), dtype=np.float64)
        batch_size = self._attach_batch_size()

        for start in range(0, len(target_samples), batch_size):
            stop = min(len(target_samples), start + batch_size)

            sim_matrix = np.matmul(query_embeddings[start:stop], self._embeddings.T)
            if exclude_self:

                for local_idx, sample in enumerate(target_samples[start:stop]):
                    sample_id = str(sample.sample_id or "").strip()
                    bank_idx = self._sample_index_by_id.get(sample_id)
                    if bank_idx is not None and bank_idx < sim_matrix.shape[1]:
                        sim_matrix[local_idx, bank_idx] = -np.inf

            batch_preds = target_preds[start:stop]
            for local_idx, pred in enumerate(batch_preds.tolist()):
                support, k_eff, bar_w = self._support_from_similarities(
                    sim_matrix[local_idx],
                    str(pred or ""),
                )
                row_idx = start + local_idx
                supports[row_idx] = support
                k_effs[row_idx] = k_eff
                bar_ws[row_idx] = bar_w

        for sample, support, k_eff, bar_w in zip(
            target_samples,
            supports.tolist(),
            k_effs.tolist(),
            bar_ws.tolist(),
            strict=True,
        ):
            setattr(sample, self.support_attr, float(max(0.0, min(1.0, support))))
            setattr(sample, self.k_eff_attr, float(max(0.0, k_eff)))
            setattr(sample, self.bar_w_attr, float(max(0.0, bar_w)))

    def snapshot(self) -> Dict[str, Any]:


        return {
            "enabled": bool(self.enabled),
            "is_fitted": bool(self._fitted),
            "embedding_model": str(self.embedding_model),
            "embedding_source": str(
                self._last_pair_cache_read.get("source", f"prepared_artifact_{self.stage}_pair_embeddings")
                or f"prepared_artifact_{self.stage}_pair_embeddings"
            ),
            "last_pair_cache_read": dict(self._last_pair_cache_read),
            "bank_size": int(self._embeddings.shape[0] if self._embeddings.ndim == 2 else 0),
            "support_prediction_source": f"{self.stage}:{self.prediction_attr}",
            "support_stage": str(self.stage),
            "embedding_attr": str(self.embedding_attr),
            "prediction_attr": str(self.prediction_attr),
            "support_attr": str(self.support_attr),
            "k_eff_attr": str(self.k_eff_attr),
            "bar_w_attr": str(self.bar_w_attr),
            "top_k": int(self.top_k),
        }


class Step2WSR:


    _DECISION_TOL = 1e-12
    _BATCH_ID_RE = re.compile(r"batch_id=([^,\)]+)")
    def __init__(
        self,
        *,
        tau: float,
        embedding_model: str = "",
        enable_neighbor_support: bool = True,
        delta_w: float = 0.05,
        calibration_level: str = "sample",
        n_tracks: int = 1,
        combiner_source: str = "source1",
        combiner_theta_grid: List[Dict[str, Any]] | None = None,
    ):

        self.mode = "wsr_step2_with_label_free_tilde_b"
        self.lambda_search_method = "bonferroni_fixed_sequence_joint_search"
        self.threshold_function = "stage1:tilde_b>=lambda1; stage2:tilde_b>=lambda2"
        self.calibration_method = "finite_population_wsr_fixed_sequence_error_rate"
        self.tau = float(tau)
        if not (0.0 <= self.tau <= 1.0):
            raise RuntimeError("tau must be within [0, 1]")
        self.epsilon = float(1.0 - self.tau)
        self.delta_w = float(delta_w)
        if not (0.0 < self.delta_w <= 1.0):
            raise RuntimeError("delta_w must be within (0, 1]")
        self.calibration_level = (
            "batch"
            if str(calibration_level or "").strip().lower() == "batch"
            else "sample"
        )


        self.n_tracks = max(1, int(n_tracks))
        self.eta_f = float(self.delta_w) / self.n_tracks
        self.a_1 = 1.0
        self.a_2 = 1.0
        self.c_1 = 1.0
        self.c_2 = 1.0
        self.beta_1 = 1.0
        self.beta_2 = 1.0
        self.ell_epsilon = 1e-3
        self.nu = 1e-2
        self.sigma0_sq = 1.0
        self.stage1_sigma0_sq = 0.0
        self.stage2_sigma0_sq = 0.0
        self.combiner_source = self._normalize_combiner_source(combiner_source)
        self._combiner_theta_grid = self._resolve_combiner_theta_grid(combiner_theta_grid)

        self._calibrated = False
        self._threshold_state: Dict[str, Any] | None = None
        self._stage1_support_bank = NeighborSupportBank(
            embedding_model=str(embedding_model or ""),
            enable_neighbor_support=bool(enable_neighbor_support),
            stage="stage1",
            embedding_attr="_stage1_pair_embedding",
            prediction_attr="stage1_prediction",
            support_attr="step1_neighbor_support",
            k_eff_attr="step1_neighbor_k_eff",
            bar_w_attr="step1_neighbor_bar_w",
            fallback_embedding_attr="",
        )
        self._stage2_support_bank = NeighborSupportBank(
            embedding_model=str(embedding_model or ""),
            enable_neighbor_support=bool(enable_neighbor_support),
            stage="stage2",
            embedding_attr="_stage2_pair_embedding",
            prediction_attr="stage2_prediction",
            support_attr="step2_neighbor_support",
            k_eff_attr="step2_neighbor_k_eff",
            bar_w_attr="step2_neighbor_bar_w",
            fallback_embedding_attr="",
        )
        self._support_bank = self._stage2_support_bank
        self._signal_quality_distribution: Dict[str, int] = {}
        self._stage1_neighbor_support_summary: Dict[str, float] = numeric_summary([])
        self._neighbor_support_summary: Dict[str, float] = numeric_summary([])
        self._stage1_tilde_b_summary: Dict[str, float] = numeric_summary([])
        self._stage2_tilde_b_summary: Dict[str, float] = numeric_summary([])

    @staticmethod
    def _clip_unit(value: Any) -> float:

        value = float(value or 0.0)
        if np.isnan(value) or value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    @staticmethod
    def _normalize_combiner_source(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"source2", "ii", "2", "theta", "theta_grid", "grid"}:
            return "source2"
        return "source1"

    @staticmethod
    def _default_combiner_params() -> Dict[str, float]:
        return {
            "a_1": 1.0,
            "a_2": 1.0,
            "c_1": 1.0,
            "c_2": 1.0,
            "beta_1": 1.0,
            "beta_2": 1.0,
            "ell_epsilon": 1e-3,
            "nu": 1e-2,
            "sigma0_sq": 1.0,
        }

    @classmethod
    def _default_source2_theta_grid(cls) -> List[Dict[str, Any]]:
        base = cls._default_combiner_params()
        return [
            {**base, "theta_id": "default"},
            {**base, "theta_id": "base_only", "c_1": 0.0, "c_2": 0.0},
            {**base, "theta_id": "strong_ns", "c_1": 2.0, "c_2": 2.0},
        ]

    @classmethod
    def _sanitize_combiner_theta(
        cls,
        raw_theta: Dict[str, Any],
        *,
        fallback: Dict[str, float],
        theta_index: int,
    ) -> Dict[str, Any]:
        theta: Dict[str, Any] = {
            "theta_index": int(theta_index),
            "theta_id": str(
                raw_theta.get("theta_id", f"theta_{theta_index}") or f"theta_{theta_index}"
            ),
        }
        for key, fallback_value in fallback.items():
            try:
                value = float(raw_theta.get(key, fallback_value))
            except (TypeError, ValueError):
                value = float(fallback_value)
            if not np.isfinite(value):
                value = float(fallback_value)
            if key in {"beta_1", "beta_2", "ell_epsilon", "sigma0_sq"}:
                value = max(value, cls._DECISION_TOL)
            elif key == "nu":
                value = max(value, 0.0)
            theta[key] = float(value)
        return theta

    def _resolve_combiner_theta_grid(
        self,
        raw_grid: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]]:
        fallback = self._default_combiner_params()
        if self.combiner_source == "source1":
            return [
                self._sanitize_combiner_theta(
                    {"theta_id": "source1_default", **fallback},
                    fallback=fallback,
                    theta_index=0,
                )
            ]
        source_grid = list(raw_grid or self._default_source2_theta_grid())
        if not source_grid:
            source_grid = self._default_source2_theta_grid()
        return [
            self._sanitize_combiner_theta(
                dict(theta),
                fallback=fallback,
                theta_index=index,
            )
            for index, theta in enumerate(source_grid)
        ]

    def _apply_combiner_params(self, theta: Dict[str, Any]) -> None:
        self.a_1 = float(theta.get("a_1", self.a_1))
        self.a_2 = float(theta.get("a_2", self.a_2))
        self.c_1 = float(theta.get("c_1", self.c_1))
        self.c_2 = float(theta.get("c_2", self.c_2))
        self.beta_1 = float(theta.get("beta_1", self.beta_1))
        self.beta_2 = float(theta.get("beta_2", self.beta_2))
        self.ell_epsilon = float(theta.get("ell_epsilon", self.ell_epsilon))
        self.nu = float(theta.get("nu", self.nu))
        self.sigma0_sq = float(theta.get("sigma0_sq", self.sigma0_sq))

    def _combiner_theta_candidates(self) -> List[Dict[str, Any]]:
        return [dict(theta) for theta in self._combiner_theta_grid]

    @staticmethod
    def _sample_query_id(sample: Sample) -> str:

        try:
            _, query_id = split_sample_id(sample.sample_id)
        except Exception:
            return ""
        return stable_key(query_id)

    @staticmethod
    def _stage1_prediction(sample: Sample) -> str:
        return normalize_binary_label(getattr(sample, "stage1_prediction", ""))

    @staticmethod
    def _stage2_prediction(sample: Sample) -> str:
        return normalize_binary_label(getattr(sample, "stage2_prediction", ""))

    @classmethod
    def _stage2_prediction_available(cls, sample: Sample) -> bool:
        return cls._stage2_prediction(sample) in {"yes", "no"}

    @classmethod
    def _stage2_base_score(cls, sample: Sample) -> float:
        try:
            score = float(getattr(sample, "stage2_routing_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if cls._stage2_prediction_available(sample) and score <= cls._DECISION_TOL:
            return 0.5
        return cls._clip_unit(score)

    @classmethod
    def _sample_calibration_group(cls, sample: Sample) -> str:

        explicit = stable_key(getattr(sample, "step1_batch_id", ""))
        if explicit:
            return explicit
        for path_item in list(getattr(sample, "path", []) or []):
            match = cls._BATCH_ID_RE.search(str(path_item or ""))
            if match:
                return stable_key(match.group(1))
        return stable_key(getattr(sample, "sample_id", ""))

    def _score_params(
        self,
        stage: str,
        theta: Dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        params = theta or {}
        if str(stage) == "stage1":
            return (
                float(params.get("a_1", self.a_1)),
                float(params.get("c_1", self.c_1)),
                float(params.get("beta_1", self.beta_1)),
            )
        return (
            float(params.get("a_2", self.a_2)),
            float(params.get("c_2", self.c_2)),
            float(params.get("beta_2", self.beta_2)),
        )

    def _combiner_params_snapshot(self) -> Dict[str, Any]:
        return {
            "combiner_source": str(self.combiner_source),
            "a_1": float(self.a_1),
            "a_2": float(self.a_2),
            "c_1": float(self.c_1),
            "c_2": float(self.c_2),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
            "ell_epsilon": float(self.ell_epsilon),
            "nu": float(self.nu),
            "sigma0_sq": float(self.sigma0_sq),
            "stage1_sigma0_sq": float(self.stage1_sigma0_sq),
            "stage2_sigma0_sq": float(self.stage2_sigma0_sq),
        }

    @staticmethod
    def _env_nonnegative_float(name: str, default: float) -> float:
        try:
            value = float(str(os.environ.get(name, default)).strip())
        except (TypeError, ValueError):
            value = float(default)
        if not np.isfinite(value) or value < 0.0:
            return float(default)
        return float(value)

    def _cost_model(self, population_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        chunk_token_values = np.asarray(
            [
                float(row.get("chunk_input_tokens", 0.0) or 0.0)
                for row in population_rows
            ],
            dtype=np.float64,
        )
        chunk_token_values = chunk_token_values[
            np.isfinite(chunk_token_values) & (chunk_token_values > 0.0)
        ]
        doc_token_values = np.asarray(
            [
                float(row.get("document_input_tokens", 0.0) or 0.0)
                for row in population_rows
            ],
            dtype=np.float64,
        )
        doc_token_values = doc_token_values[
            np.isfinite(doc_token_values) & (doc_token_values > 0.0)
        ]
        tau_chunk = (
            float(np.mean(chunk_token_values, dtype=np.float64))
            if chunk_token_values.size
            else 1.0
        )
        tau_doc = (
            float(np.mean(doc_token_values, dtype=np.float64))
            if doc_token_values.size
            else 1.0
        )
        slm_input_rate = self._env_nonnegative_float(
            "CASCADE_COST_SLM_INPUT_PER_MILLION", 0.2
        )
        slm_output_rate = self._env_nonnegative_float(
            "CASCADE_COST_SLM_OUTPUT_PER_MILLION", 0.2
        )
        oracle_input_rate = self._env_nonnegative_float(
            "CASCADE_COST_ORACLE_INPUT_PER_MILLION", 5.0
        )
        oracle_output_rate = self._env_nonnegative_float(
            "CASCADE_COST_ORACLE_OUTPUT_PER_MILLION", 30.0
        )
        tau_slm_output = self._env_nonnegative_float(
            "CASCADE_COST_SLM_OUTPUT_TOKENS", 1.0
        )
        tau_oracle_output = self._env_nonnegative_float(
            "CASCADE_COST_ORACLE_OUTPUT_TOKENS", 1.0
        )
        stage1_constant_per_record = (
            slm_input_rate * tau_chunk + slm_output_rate * tau_slm_output
        ) / 1_000_000.0
        stage2_per_reached_record = (
            slm_input_rate * tau_doc + slm_output_rate * tau_slm_output
        ) / 1_000_000.0
        stage3_per_oracle_record = (
            oracle_input_rate * tau_doc + oracle_output_rate * tau_oracle_output
        ) / 1_000_000.0
        return {
            "tau_chunk_input_tokens": float(tau_chunk),
            "tau_doc_input_tokens": float(tau_doc),
            "tau_slm_output_tokens": float(tau_slm_output),
            "tau_oracle_output_tokens": float(tau_oracle_output),
            "slm_input_per_million": float(slm_input_rate),
            "slm_output_per_million": float(slm_output_rate),
            "oracle_input_per_million": float(oracle_input_rate),
            "oracle_output_per_million": float(oracle_output_rate),
            "stage1_constant_per_record_usd": float(stage1_constant_per_record),
            "stage2_per_reached_record_usd": float(stage2_per_reached_record),
            "stage3_per_oracle_record_usd": float(stage3_per_oracle_record),
        }

    @staticmethod
    def _candidate_cost(
        *,
        population_count: int,
        r_reach2: float,
        r3: float,
        cost_model: Dict[str, float],
    ) -> tuple[float, Dict[str, float]]:
        n = max(0, int(population_count))
        stage1_cost = float(n) * float(
            cost_model.get("stage1_constant_per_record_usd", 0.0) or 0.0
        )
        stage2_cost = (
            float(n)
            * float(r_reach2)
            * float(cost_model.get("stage2_per_reached_record_usd", 0.0) or 0.0)
        )
        stage3_cost = (
            float(n)
            * float(r3)
            * float(cost_model.get("stage3_per_oracle_record_usd", 0.0) or 0.0)
        )
        total = stage1_cost + stage2_cost + stage3_cost
        return float(total), {
            "stage1_constant_usd": float(stage1_cost),
            "stage2_reach_usd": float(stage2_cost),
            "stage3_oracle_usd": float(stage3_cost),
        }

    @staticmethod
    def _grid_size(name: str, default: int) -> int:
        try:
            return max(2, int(str(os.environ.get(name, default)).strip()))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _score_grid(values: np.ndarray, *, max_count: int) -> np.ndarray:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.asarray([0.0], dtype=np.float64)
        candidates = np.unique(finite)
        if candidates.size <= max_count:
            return candidates.astype(np.float64)
        quantile_points = np.linspace(0.0, 1.0, max(2, int(max_count)))
        return np.unique(np.quantile(finite, quantile_points)).astype(np.float64)

    def _lambda1_candidates_from_samples(self, samples: List[Sample]) -> np.ndarray:
        return np.sort(
            self._score_grid(
                np.asarray([float(sample.step1_tilde_b) for sample in samples], dtype=np.float64),
                max_count=self._grid_size("CASCADE_WSR_MAX_LAMBDA1_CANDIDATES", 32),
            )
        )[::-1]

    def _stage1_tilde_b_for_sample(self, sample: Sample, theta: Dict[str, Any]) -> float:
        value, _ = self._tilde_b(
            float(sample.step1_routing_score or 0.0),
            float(sample.step1_neighbor_support or 0.5),
            float(sample.step1_neighbor_k_eff or 0.0),
            float(sample.step1_neighbor_bar_w or 0.0),
            stage="stage1",
            theta=theta,
        )
        return float(value)

    def _lambda1_max_by_theta(
        self,
        samples: List[Sample],
        theta_grid: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for theta in theta_grid:
            values = np.asarray(
                [self._stage1_tilde_b_for_sample(sample, theta) for sample in samples],
                dtype=np.float64,
            )
            candidates = self._score_grid(
                values,
                max_count=self._grid_size("CASCADE_WSR_MAX_LAMBDA1_CANDIDATES", 32),
            )
            theta_index = int(theta.get("theta_index", 0) or 0)
            result[theta_index] = float(np.max(candidates)) if candidates.size else 0.0
        return result

    def _in_any_i2_max(
        self,
        sample: Sample,
        theta_grid: List[Dict[str, Any]],
        lambda1_max_by_theta: Dict[int, float],
    ) -> bool:
        for theta in theta_grid:
            theta_index = int(theta.get("theta_index", 0) or 0)
            lambda1_max = float(lambda1_max_by_theta.get(theta_index, 0.0))
            if self._stage1_tilde_b_for_sample(sample, theta) < lambda1_max - self._DECISION_TOL:
                return True
        return False

    def _tilde_arrays_from_rows(
        self,
        rows: List[Dict[str, Any]],
        theta: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        stage1_values: list[float] = []
        stage2_values: list[float] = []
        for row in rows:
            stage1_value, _ = self._tilde_b(
                float(row.get("routing_score", 0.0) or 0.0),
                float(row.get("stage1_neighbor_support", 0.5) or 0.5),
                float(row.get("stage1_neighbor_k_eff", 0.0) or 0.0),
                float(row.get("stage1_neighbor_bar_w", 0.0) or 0.0),
                stage="stage1",
                theta=theta,
            )
            stage2_value, _ = self._tilde_b(
                float(row.get("stage2_routing_score", 0.0) or 0.0),
                float(row.get("stage2_neighbor_support", 0.5) or 0.5),
                float(row.get("stage2_neighbor_k_eff", 0.0) or 0.0),
                float(row.get("stage2_neighbor_bar_w", 0.0) or 0.0),
                stage="stage2",
                theta=theta,
            )
            stage1_values.append(float(stage1_value))
            stage2_values.append(float(stage2_value))
        return (
            np.asarray(stage1_values, dtype=np.float64),
            np.asarray(stage2_values, dtype=np.float64),
        )

    def _ell_epsilon(self, b: float, theta: Dict[str, Any] | None = None) -> float:
        value = self._clip_unit(b)
        eps = max(float((theta or {}).get("ell_epsilon", self.ell_epsilon)), self._DECISION_TOL)
        return float(np.log((value + eps) / (1.0 - value + eps)))

    @classmethod
    def _m(cls, NS: float, beta: float) -> float:
        centered = (2.0 * cls._clip_unit(NS)) - 1.0
        if abs(centered) <= cls._DECISION_TOL:
            return 0.0
        beta_value = max(float(beta), cls._DECISION_TOL)
        return float(np.sign(centered) * (abs(centered) ** beta_value))

    def _rho(
        self,
        NS: float,
        K_eff: float,
        bar_w: float,
        *,
        stage: str,
        theta: Dict[str, Any] | None = None,
    ) -> float:
        weight_mass = float(K_eff) * float(bar_w)
        if not np.isfinite(weight_mass) or weight_mass <= self._DECISION_TOL:
            return 0.0
        sigma0_sq = (
            float(self.stage1_sigma0_sq)
            if str(stage) == "stage1"
            else float(self.stage2_sigma0_sq)
        )
        sigma0_sq = max(0.0, sigma0_sq)
        return float(max(0.0, min(1.0, weight_mass / (weight_mass + sigma0_sq))))

    def _set_stage_sigma0_sq(self, samples: List[Sample], *, stage: str) -> None:
        if str(stage) == "stage1":
            values = [
                float(sample.step1_neighbor_k_eff or 0.0)
                * float(sample.step1_neighbor_bar_w or 0.0)
                for sample in samples
            ]
            self.stage1_sigma0_sq = float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0
            return
        values = [
            float(sample.step2_neighbor_k_eff or 0.0)
            * float(sample.step2_neighbor_bar_w or 0.0)
            for sample in samples
        ]
        self.stage2_sigma0_sq = float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0

    def _tilde_b(
        self,
        b: float,
        NS: float,
        K_eff: float,
        bar_w: float,
        *,
        stage: str,
        theta: Dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        _, c_s, _ = self._score_params(stage, theta=theta)
        rho = self._rho(NS, K_eff, bar_w, stage=stage, theta=theta)
        support = self._clip_unit(NS)
        theta_hat = (rho * support) + ((1.0 - rho) * 0.5)
        value = self._ell_epsilon(b, theta=theta) + (c_s * ((2.0 * theta_hat) - 1.0))
        return float(value), float(rho)

    def _annotate_tilde_b(
        self,
        samples: List[Sample],
        *,
        theta: Dict[str, Any] | None = None,
    ) -> None:
        for sample in samples:
            step1_tilde_b, step1_rho = self._tilde_b(
                float(sample.step1_routing_score or 0.0),
                float(sample.step1_neighbor_support or 0.5),
                float(sample.step1_neighbor_k_eff or 0.0),
                float(sample.step1_neighbor_bar_w or 0.0),
                stage="stage1",
                theta=theta,
            )
            step2_tilde_b, step2_rho = self._tilde_b(
                self._stage2_base_score(sample),
                float(sample.step2_neighbor_support or 0.5),
                float(sample.step2_neighbor_k_eff or 0.0),
                float(sample.step2_neighbor_bar_w or 0.0),
                stage="stage2",
                theta=theta,
            )
            sample.step1_neighbor_rho = step1_rho
            sample.step2_neighbor_rho = step2_rho
            sample.step1_tilde_b = step1_tilde_b
            sample.step2_tilde_b = step2_tilde_b

    @staticmethod
    def _state_lambda(threshold_state: Dict[str, Any], key: str) -> Optional[float]:
        value = threshold_state.get(key)
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    def _cascade_decision_details(
        self,
        sample: Sample,
        threshold_state: Dict[str, Any],
    ) -> Dict[str, Any]:

        lambda1 = self._state_lambda(threshold_state, "lambda1_hat")
        lambda2 = self._state_lambda(threshold_state, "lambda2_hat")
        lambda1_max = self._state_lambda(threshold_state, "lambda1_max")
        stage1_tilde_b = float(sample.step1_tilde_b)
        stage2_tilde_b = float(sample.step2_tilde_b)
        stage2_prediction_available = self._stage2_prediction_available(sample)
        in_I2_max = bool(
            True
            if lambda1_max is None
            else stage1_tilde_b < float(lambda1_max) - self._DECISION_TOL
        )
        stage1_accept = bool(
            lambda1 is not None
            and stage1_tilde_b >= float(lambda1) - self._DECISION_TOL
        )
        stage2_accept = bool(
            not stage1_accept
            and in_I2_max
            and stage2_prediction_available
            and lambda2 is not None
            and stage2_tilde_b >= float(lambda2) - self._DECISION_TOL
        )
        decision_threshold = lambda1 if stage1_accept else lambda2
        decision_stage = "stage1" if stage1_accept else ("stage2" if stage2_accept else "stage3")
        return {
            "accepted": bool(stage1_accept or stage2_accept),
            "decision_stage": decision_stage,
            "stage1_accept": bool(stage1_accept),
            "stage2_accept": bool(stage2_accept),
            "in_I2_max": bool(in_I2_max),
            "stage2_prediction_available": bool(stage2_prediction_available),
            "stage1_decision_threshold": lambda1,
            "stage2_decision_threshold": lambda2,
            "decision_threshold": decision_threshold,
        }

    def annotate_neighbor_support(self, samples: List[Sample], *, exclude_self: bool = False) -> None:


        self._stage1_support_bank.attach(samples, exclude_self=exclude_self)
        self._stage2_support_bank.attach(samples, exclude_self=exclude_self)
        self._annotate_tilde_b(samples)

    @staticmethod
    def _is_step2_eligible_sample(sample: Sample) -> bool:

        label = str(sample.label or "").strip().lower()
        pred = Step2WSR._stage1_prediction(sample)
        signal_quality = str(sample.step1_signal_quality or "").strip()
        return (
            label in {"yes", "no"}
            and pred in {"yes", "no"}
            and signal_quality in STEP1_ROUTABLE_SIGNAL_QUALITIES
        )

    def _collect_step2_eligible_samples(self, samples: List[Sample]) -> List[Sample]:


        return [sample for sample in samples if self._is_step2_eligible_sample(sample)]

    @staticmethod
    def _is_stage1_support_sample(sample: Sample) -> bool:

        pred = Step2WSR._stage1_prediction(sample)
        signal_quality = str(sample.step1_signal_quality or "").strip()
        return pred in {"yes", "no"} and signal_quality in STEP1_ROUTABLE_SIGNAL_QUALITIES

    @staticmethod
    def _is_stage2_support_sample(sample: Sample) -> bool:

        return (
            Step2WSR._is_stage1_support_sample(sample)
            and Step2WSR._stage2_prediction(sample) in {"yes", "no"}
        )

    def _collect_stage1_support_samples(self, samples: List[Sample]) -> List[Sample]:

        return [sample for sample in samples if self._is_stage1_support_sample(sample)]

    def _collect_stage2_support_samples(self, samples: List[Sample]) -> List[Sample]:

        return [sample for sample in samples if self._is_stage2_support_sample(sample)]

    def _build_step2_rows(
        self,
        samples: List[Sample],
        *,
        include_wrong: bool = True,
    ) -> List[Dict[str, Any]]:

        rows: List[Dict[str, Any]] = []
        for sample in samples:
            stage1_pred = self._stage1_prediction(sample)
            stage2_pred = self._stage2_prediction(sample)
            signal_quality = str(sample.step1_signal_quality or "").strip()
            query_id = self._sample_query_id(sample)
            stage1_wrong = False
            stage2_wrong = False
            if include_wrong:
                label = normalize_binary_label(sample.label)
                stage1_wrong = bool(stage1_pred in {"yes", "no"} and stage1_pred != label)
                stage2_wrong = bool(stage2_pred in {"yes", "no"} and stage2_pred != label)
            rows.append(
                {
                    "query_id": query_id,
                    "calibration_group": self._sample_calibration_group(sample),
                    "pred": stage1_pred,
                    "stage1_pred": stage1_pred,
                    "stage2_pred": stage2_pred,
                    "signal_quality": signal_quality,
                    "routing_score": self._clip_unit(float(sample.step1_routing_score or 0.0)),
                    "stage2_routing_score": self._stage2_base_score(sample),
                    "neighbor_support": self._clip_unit(float(sample.step2_neighbor_support or 0.5)),
                    "stage1_neighbor_support": self._clip_unit(
                        float(sample.step1_neighbor_support or 0.5)
                    ),
                    "stage1_neighbor_k_eff": float(sample.step1_neighbor_k_eff or 0.0),
                    "stage1_neighbor_bar_w": float(sample.step1_neighbor_bar_w or 0.0),
                    "stage1_neighbor_rho": float(sample.step1_neighbor_rho or 0.0),
                    "tilde_b_stage1": float(sample.step1_tilde_b),
                    "stage2_neighbor_support": self._clip_unit(
                        float(sample.step2_neighbor_support or 0.5)
                    ),
                    "stage2_neighbor_k_eff": float(sample.step2_neighbor_k_eff or 0.0),
                    "stage2_neighbor_bar_w": float(sample.step2_neighbor_bar_w or 0.0),
                    "stage2_neighbor_rho": float(sample.step2_neighbor_rho or 0.0),
                    "tilde_b_stage2": float(sample.step2_tilde_b),
                    "in_I2_max": bool(getattr(sample, "_in_I2_max", True)),
                    "stage2_prediction_available": bool(stage2_pred in {"yes", "no"}),
                    "chunk_input_tokens": int(
                        estimate_text_tokens(sample.prompt_evidence_text or sample.text)
                    ),
                    "document_input_tokens": int(estimate_text_tokens(sample.text)),
                    "stage1_wrong": stage1_wrong,
                    "stage2_wrong": stage2_wrong,
                }
            )
        return rows

    def _disabled_threshold_state(
        self,
        *,
        reason: str,
        fit_support: int = 0,
        population_size: int = 0,
        population_group_count: int = 0,
    ) -> Dict[str, Any]:

        return {
            "fit_support": int(fit_support),
            "population_size": int(population_size or fit_support),
            "population_group_count": int(population_group_count or population_size or fit_support),
            "accepted_count": 0,
            "accepted_rate": 0.0,
            "wrong_accept_count": 0,
            "empirical_error_at_lambda": None,
            "u_at_lambda": None,
            "observed_accuracy_at_lambda": None,
            "lambda_hat": None,
            "lambda1_hat": None,
            "lambda2_hat": None,
            "selected_r_reach2": None,
            "selected_r3": None,
            "selected_cost": None,
            "certified_candidate_count": 0,
            "candidate_count": 0,
            "tau": float(self.tau),
            "epsilon": float(self.epsilon),
            "delta_w": float(self.delta_w),
            "n_tracks": int(self.n_tracks),
            "eta_f": float(self.eta_f),
            "enabled": False,
            "disable_reason": str(reason),
            "method": str(self.calibration_method),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "calibration_level": str(self.calibration_level),
            "requested_calibration_level": str(self.calibration_level),
            "wsr": {
                "selection_rule": "bonferroni_fixed_sequence_joint_search",
                "tracks": [],
                "certified_candidates": [],
                "selected_candidate": None,
            },
            "combiner_params": self._combiner_params_snapshot(),
            "I2_max_size": 0,
            "I2_max_stage2_prediction_count": 0,
        }

    def _learn_threshold_state(
        self,
        *,
        rows: List[Dict[str, Any]],
        population_rows: List[Dict[str, Any]] | None = None,
        population_size: int | None = None,
        population_group_count: int | None = None,
        calibration_level: str | None = None,
    ) -> Dict[str, Any]:




        fit_support = int(len(rows))
        population_n = max(int(population_size or fit_support), fit_support)
        requested_level = (
            "batch"
            if str(calibration_level or self.calibration_level).strip().lower() == "batch"
            else "sample"
        )
        group_ids = [stable_key(row.get("calibration_group")) for row in rows]
        usable_batch_groups = requested_level == "batch" and any(group_ids)
        group_population_n = max(
            int(population_group_count or 0),
            len({group for group in group_ids if group}),
        )
        if not rows:
            return self._disabled_threshold_state(
                reason="wsr_no_labeled_rows",
                fit_support=0,
                population_size=population_n,
                population_group_count=group_population_n,
            )

        stage1_wrong_indicators = np.asarray(
            [1 if bool(row.get("stage1_wrong", False)) else 0 for row in rows],
            dtype=bool,
        )
        stage2_wrong_indicators = np.asarray(
            [1 if bool(row.get("stage2_wrong", False)) else 0 for row in rows],
            dtype=bool,
        )
        stage2_available_mask = np.asarray(
            [bool(row.get("stage2_prediction_available", False)) for row in rows],
            dtype=bool,
        )
        grid_rows = list(population_rows or rows)
        population_stage2_available_mask = np.asarray(
            [bool(row.get("stage2_prediction_available", False)) for row in grid_rows],
            dtype=bool,
        )
        population_count = max(int(len(grid_rows)), int(population_n))
        cost_model = self._cost_model(grid_rows)
        theta_grid = self._combiner_theta_candidates()
        theta_states: list[dict[str, Any]] = []
        track_count_base = 0
        candidate_count = 0
        for theta in theta_grid:
            tilde_b_stage1, tilde_b_stage2 = self._tilde_arrays_from_rows(rows, theta)
            population_tilde_b_stage1, population_tilde_b_stage2 = self._tilde_arrays_from_rows(
                grid_rows,
                theta,
            )
            lambda1_candidates = np.sort(
                self._score_grid(
                    population_tilde_b_stage1,
                    max_count=self._grid_size("CASCADE_WSR_MAX_LAMBDA1_CANDIDATES", 32),
                )
            )[::-1]
            lambda1_max = float(np.max(lambda1_candidates)) if lambda1_candidates.size else 0.0
            i2_max_mask = tilde_b_stage1 < lambda1_max - self._DECISION_TOL
            population_i2_max_mask = (
                population_tilde_b_stage1 < lambda1_max - self._DECISION_TOL
            )
            lambda2_population = population_tilde_b_stage2[population_i2_max_mask]
            if lambda2_population.size == 0:
                lambda2_population = population_tilde_b_stage2
            lambda2_candidates = np.sort(
                self._score_grid(
                    lambda2_population,
                    max_count=self._grid_size("CASCADE_WSR_MAX_LAMBDA2_CANDIDATES", 64),
                )
            )[::-1]
            track_count_base += int(lambda1_candidates.size)
            candidate_count += int(lambda1_candidates.size * lambda2_candidates.size)
            theta_states.append(
                {
                    "theta": dict(theta),
                    "tilde_b_stage1": tilde_b_stage1,
                    "tilde_b_stage2": tilde_b_stage2,
                    "population_tilde_b_stage1": population_tilde_b_stage1,
                    "population_tilde_b_stage2": population_tilde_b_stage2,
                    "i2_max_mask": i2_max_mask,
                    "population_i2_max_mask": population_i2_max_mask,
                    "lambda1_candidates": lambda1_candidates,
                    "lambda2_candidates": lambda2_candidates,
                    "lambda1_max": float(lambda1_max),
                }
            )
        track_count = max(1, int(self.n_tracks) * int(track_count_base))
        eta_f = float(self.delta_w) / float(track_count)

        certified_candidates: list[dict[str, Any]] = []
        track_diagnostics: list[dict[str, Any]] = []
        track_index = 0
        lambda1_candidate_count = 0
        lambda2_candidate_count = 0
        for theta_state in theta_states:
            theta = dict(theta_state["theta"])
            theta_index = int(theta.get("theta_index", 0) or 0)
            theta_id = str(theta.get("theta_id", f"theta_{theta_index}") or f"theta_{theta_index}")
            tilde_b_stage1 = theta_state["tilde_b_stage1"]
            tilde_b_stage2 = theta_state["tilde_b_stage2"]
            population_tilde_b_stage1 = theta_state["population_tilde_b_stage1"]
            population_tilde_b_stage2 = theta_state["population_tilde_b_stage2"]
            i2_max_mask = theta_state["i2_max_mask"]
            population_i2_max_mask = theta_state["population_i2_max_mask"]
            lambda1_candidates = theta_state["lambda1_candidates"]
            lambda2_candidates = theta_state["lambda2_candidates"]
            lambda1_candidate_count += int(lambda1_candidates.size)
            lambda2_candidate_count = max(lambda2_candidate_count, int(lambda2_candidates.size))
            for lambda1 in lambda1_candidates.tolist():
                track_index += 1
                lambda1_value = float(lambda1)
                stage1_accept = tilde_b_stage1 >= float(lambda1_value) - self._DECISION_TOL
                population_stage1_accept = (
                    population_tilde_b_stage1 >= float(lambda1_value) - self._DECISION_TOL
                )
                stage2_accept_matrix = (
                    (~stage1_accept)[None, :]
                    & i2_max_mask[None, :]
                    & stage2_available_mask[None, :]
                    & (
                        tilde_b_stage2[None, :]
                        >= lambda2_candidates[:, None] - self._DECISION_TOL
                    )
                )
                population_stage2_accept_matrix = (
                    (~population_stage1_accept)[None, :]
                    & population_i2_max_mask[None, :]
                    & population_stage2_available_mask[None, :]
                    & (
                        population_tilde_b_stage2[None, :]
                        >= lambda2_candidates[:, None] - self._DECISION_TOL
                    )
                )
                accept_matrix = stage1_accept[None, :] | stage2_accept_matrix
                population_accept_matrix = population_stage1_accept[None, :] | population_stage2_accept_matrix
                stage1_error = stage1_accept & stage1_wrong_indicators
                stage2_error_matrix = stage2_accept_matrix & stage2_wrong_indicators[None, :]
                error_matrix = stage1_error[None, :] | stage2_error_matrix
                track_candidates: list[dict[str, Any]] = []
                stop_index: int | None = None
                stop_reason = ""

                for lambda2_index, lambda2 in enumerate(lambda2_candidates.tolist(), start=1):
                    errors = error_matrix[lambda2_index - 1].astype(np.float64, copy=False)
                    log_wealth = betting_log_wealth(
                        errors,
                        M=int(population_n),
                        u=float(self.epsilon),
                    )
                    log_boundary = float(np.log(1.0 / float(eta_f)))
                    certified = bool(log_wealth >= log_boundary - self._DECISION_TOL)
                    upper_count, upper_bound = betting_population_upper_bound(
                        errors,
                        M=int(population_n),
                        alpha=float(eta_f),
                    )
                    accepted_count = int(np.sum(accept_matrix[lambda2_index - 1], dtype=np.int64))
                    population_accepted_count = int(
                        np.sum(population_accept_matrix[lambda2_index - 1], dtype=np.int64)
                    )
                    wrong_accept_count = int(np.sum(error_matrix[lambda2_index - 1], dtype=np.int64))
                    empirical_error = float(wrong_accept_count / fit_support) if fit_support else 0.0
                    population_reach2_count = int(
                        np.sum(
                            (~population_stage1_accept) & population_i2_max_mask,
                            dtype=np.int64,
                        )
                    )
                    r_reach2 = (
                        float(population_reach2_count / population_count)
                        if population_count
                        else 1.0
                    )
                    r3 = (
                        float(1.0 - (population_accepted_count / population_count))
                        if population_count
                        else 1.0
                    )
                    candidate_cost, cost_components = self._candidate_cost(
                        population_count=population_count,
                        r_reach2=r_reach2,
                        r3=r3,
                        cost_model=cost_model,
                    )
                    candidate = {
                        "track_index": int(track_index),
                        "theta_index": int(theta_index),
                        "theta_id": theta_id,
                        "theta": dict(theta),
                        "lambda2_index": int(lambda2_index),
                        "lambda1": float(lambda1_value),
                        "lambda2": float(lambda2),
                        "lambda1_max": float(theta_state["lambda1_max"]),
                        "accepted_count": int(accepted_count),
                        "accepted_rate": float(accepted_count / fit_support) if fit_support else 0.0,
                        "population_accepted_count": int(population_accepted_count),
                        "population_accepted_rate": (
                            float(population_accepted_count / population_count)
                            if population_count
                            else 0.0
                        ),
                        "wrong_accept_count": int(wrong_accept_count),
                        "empirical_error": float(empirical_error),
                        "upper_count": int(upper_count),
                        "upper_bound": float(upper_bound),
                        "log_wealth_at_epsilon": float(log_wealth),
                        "log_certification_boundary": float(log_boundary),
                        "r_reach2": float(r_reach2),
                        "r3": float(r3),
                        "cost": float(candidate_cost),
                        "cost_components": dict(cost_components),
                        "I2_max_size": int(np.sum(population_i2_max_mask, dtype=np.int64)),
                        "I2_max_stage2_prediction_count": int(
                            np.sum(
                                population_i2_max_mask & population_stage2_available_mask,
                                dtype=np.int64,
                            )
                        ),
                        "certified": bool(certified),
                    }
                    track_candidates.append(candidate)
                    if candidate["certified"]:
                        certified_candidates.append(candidate)
                        continue
                    stop_index = int(lambda2_index)
                    stop_reason = "single_point_wealth_below_boundary"
                    break

                track_diagnostics.append(
                    {
                        "track_index": int(track_index),
                        "theta_index": int(theta_index),
                        "theta_id": theta_id,
                        "lambda1": float(lambda1_value),
                        "lambda1_max": float(theta_state["lambda1_max"]),
                        "eta_f": float(eta_f),
                        "stop_index": (
                            int(stop_index)
                            if stop_index is not None
                            else int(lambda2_candidates.size + 1)
                        ),
                        "stop_reason": stop_reason or "completed",
                        "tested_count": int(len(track_candidates)),
                        "certified_count": int(
                            sum(1 for item in track_candidates if bool(item.get("certified", False)))
                        ),
                        "candidates": track_candidates,
                    }
                )

        if not certified_candidates:
            return self._disabled_threshold_state(
                reason="wsr_fixed_sequence_empty_certified_set",
                fit_support=fit_support,
                population_size=population_n,
                population_group_count=group_population_n,
            )

        selected_candidate = min(
            certified_candidates,
            key=lambda item: (
                float(item["cost"]),
                float(item["r3"]),
                float(item["r_reach2"]),
                -float(item["lambda1"]),
                -float(item["lambda2"]),
                int(item.get("theta_index", 0) or 0),
            ),
        )
        accepted_count = int(selected_candidate["accepted_count"])
        wrong_accept_count = int(selected_candidate["wrong_accept_count"])
        empirical_error = float(selected_candidate["empirical_error"])
        lambda1_hat = float(selected_candidate["lambda1"])
        lambda2_hat = float(selected_candidate["lambda2"])
        selected_theta = dict(selected_candidate.get("theta", {}) or {})
        wsr_summary = {
            "selection_rule": "bonferroni_fixed_sequence_joint_search",
            "tau": float(self.tau),
            "epsilon": float(self.epsilon),
            "delta_w": float(self.delta_w),
            "eta_f": float(eta_f),
            "track_count": int(track_count),
            "candidate_count": int(candidate_count),
            "certified_candidate_count": int(len(certified_candidates)),
            "theta_count": int(len(theta_grid)),
            "theta_grid": [dict(theta) for theta in theta_grid],
            "selected_theta": dict(selected_theta),
            "lambda1_candidate_count": int(lambda1_candidate_count),
            "lambda2_candidate_count_max": int(lambda2_candidate_count),
            "selected_candidate": dict(selected_candidate),
            "certified_candidates": [dict(item) for item in certified_candidates],
            "tracks": track_diagnostics,
            "fixed_sequence_order": "lambda2_descending_strict_to_loose",
            "bonferroni_tracks": "lambda1_x_theta",
            "cost_model": dict(cost_model),
            "joint_search_sort_key": "min_cost_then_min_r3_then_min_r_reach2_then_descending_lambda1_then_descending_lambda2_then_theta_index",
        }

        return {
            "fit_support": int(fit_support),
            "population_size": int(population_n),
            "population_group_count": int(group_population_n or population_n),
            "accepted_count": int(accepted_count),
            "accepted_rate": float(accepted_count / fit_support) if fit_support else 0.0,
            "wrong_accept_count": int(wrong_accept_count),
            "empirical_error_at_lambda": float(empirical_error),
            "u_at_lambda": float(selected_candidate["upper_bound"]),
            "observed_accuracy_at_lambda": float(1.0 - empirical_error),
            "lambda_hat": float(lambda2_hat),
            "lambda1_hat": float(lambda1_hat),
            "lambda2_hat": float(lambda2_hat),
            "selected_r_reach2": float(selected_candidate["r_reach2"]),
            "selected_r3": float(selected_candidate["r3"]),
            "selected_cost": float(selected_candidate["cost"]),
            "certified_candidate_count": int(len(certified_candidates)),
            "candidate_count": int(candidate_count),
            "tau": float(self.tau),
            "epsilon": float(self.epsilon),
            "delta_w": float(self.delta_w),
            "n_tracks": int(track_count),
            "eta_f": float(eta_f),
            "enabled": True,
            "disable_reason": "",
            "method": str(self.calibration_method),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "calibration_level": "sample",
            "requested_calibration_level": str(requested_level),
            "wsr": wsr_summary,
            "combiner_params": dict(selected_theta),
            "combiner_source": str(self.combiner_source),
            "lambda1_max": float(selected_candidate["lambda1_max"]),
            "I2_max_size": int(selected_candidate["I2_max_size"]),
            "I2_max_stage2_prediction_count": int(
                selected_candidate["I2_max_stage2_prediction_count"]
            ),
        }

    def calibrate(
        self,
        samples: List[Sample],
        *,
        population_size: int | None = None,
        population_group_count: int | None = None,
        calibration_level: str | None = None,
        neighbor_support_samples: List[Sample] | None = None,
    ) -> None:


        self._threshold_state = None
        self._calibrated = False
        self._signal_quality_distribution = {}
        self._stage1_neighbor_support_summary = numeric_summary([])
        self._neighbor_support_summary = numeric_summary([])
        self._stage1_tilde_b_summary = numeric_summary([])
        self._stage2_tilde_b_summary = numeric_summary([])

        eligible_samples = self._collect_step2_eligible_samples(samples)
        if not eligible_samples:
            self._stage1_support_bank.reset()
            self._stage2_support_bank.reset()
            self._threshold_state = None
            return
        support_source = samples if neighbor_support_samples is None else neighbor_support_samples
        stage1_support_samples = self._collect_stage1_support_samples(support_source)
        if not stage1_support_samples:
            self._stage1_support_bank.reset()
            self._stage2_support_bank.reset()
            self._threshold_state = None
            return
        stage2_support_samples = self._collect_stage2_support_samples(support_source)



        self._stage1_support_bank.fit(stage1_support_samples)
        self._stage2_support_bank.reset()
        if neighbor_support_samples is None:
            self._stage1_support_bank.attach(eligible_samples, exclude_self=True)
            self._set_stage_sigma0_sq(eligible_samples, stage="stage1")
            self._annotate_tilde_b(eligible_samples)
            population_samples = eligible_samples
        else:
            self._stage1_support_bank.attach(stage1_support_samples, exclude_self=True)
            self._set_stage_sigma0_sq(stage1_support_samples, stage="stage1")
            self._annotate_tilde_b(stage1_support_samples)
            population_samples = stage1_support_samples
            self._stage1_support_bank.attach(eligible_samples, exclude_self=True)
            self._annotate_tilde_b(eligible_samples)

        theta_grid = self._combiner_theta_candidates()
        lambda1_max_by_theta = self._lambda1_max_by_theta(population_samples, theta_grid)
        for sample in stage1_support_samples:
            setattr(
                sample,
                "_in_I2_max",
                bool(self._in_any_i2_max(sample, theta_grid, lambda1_max_by_theta)),
            )
        for sample in stage2_support_samples:
            setattr(
                sample,
                "_in_I2_max",
                bool(self._in_any_i2_max(sample, theta_grid, lambda1_max_by_theta)),
            )
        for sample in eligible_samples:
            setattr(
                sample,
                "_in_I2_max",
                bool(self._in_any_i2_max(sample, theta_grid, lambda1_max_by_theta)),
            )
        i2_max_population_samples = [
            sample
            for sample in stage1_support_samples
            if bool(getattr(sample, "_in_I2_max", False))
        ]
        i2_max_support_samples = [
            sample
            for sample in stage2_support_samples
            if bool(getattr(sample, "_in_I2_max", False))
        ]
        self._stage2_support_bank.fit(i2_max_support_samples)
        if neighbor_support_samples is None:
            stage2_sigma_samples = [
                sample
                for sample in eligible_samples
                if bool(getattr(sample, "_in_I2_max", False))
                and self._stage2_prediction_available(sample)
            ]
            self._stage2_support_bank.attach(stage2_sigma_samples, exclude_self=True)
            self._set_stage_sigma0_sq(stage2_sigma_samples, stage="stage2")
            self._stage2_support_bank.attach(eligible_samples, exclude_self=True)
            self._annotate_tilde_b(eligible_samples)
            population_samples = eligible_samples
        else:
            self._stage2_support_bank.attach(i2_max_support_samples, exclude_self=True)
            self._set_stage_sigma0_sq(i2_max_support_samples, stage="stage2")
            self._stage2_support_bank.attach(stage1_support_samples, exclude_self=True)
            self._annotate_tilde_b(stage1_support_samples)
            population_samples = stage1_support_samples
            self._stage2_support_bank.attach(eligible_samples, exclude_self=True)
            self._annotate_tilde_b(eligible_samples)

        population_rows = self._build_step2_rows(
            population_samples,
            include_wrong=False,
        )
        step2_rows = self._build_step2_rows(
            eligible_samples,
        )
        self._signal_quality_distribution = signal_quality_distribution_from_samples(
            eligible_samples,
            labeled_only=True,
        )
        self._neighbor_support_summary = neighbor_support_summary_from_samples(
            eligible_samples,
            labeled_only=True,
            support_attr="step2_neighbor_support",
            prediction_attr="stage2_prediction",
        )
        self._stage1_neighbor_support_summary = neighbor_support_summary_from_samples(
            eligible_samples,
            labeled_only=True,
            support_attr="step1_neighbor_support",
            prediction_attr="stage1_prediction",
        )
        self._threshold_state = self._learn_threshold_state(
            rows=step2_rows,
            population_rows=population_rows,
            population_size=population_size,
            population_group_count=population_group_count,
            calibration_level=calibration_level,
        )
        if bool(self._threshold_state.get("enabled", False)):
            selected_theta = dict(
                self._threshold_state.get("combiner_params", self._combiner_params_snapshot())
                or {}
            )
            self._apply_combiner_params(selected_theta)
            if neighbor_support_samples is None:
                self._annotate_tilde_b(eligible_samples)
            else:
                self._annotate_tilde_b(stage1_support_samples)
                self._annotate_tilde_b(eligible_samples)
            step2_rows = self._build_step2_rows(eligible_samples)
        self._stage1_tilde_b_summary = numeric_summary(
            [float(row.get("tilde_b_stage1", 0.0) or 0.0) for row in step2_rows]
        )
        self._stage2_tilde_b_summary = numeric_summary(
            [float(row.get("tilde_b_stage2", 0.0) or 0.0) for row in step2_rows]
        )
        self._calibrated = bool(self._threshold_state.get("enabled", False))

    def _normalized_threshold_state(self) -> Dict[str, Any] | None:

        state = self._threshold_state
        if not isinstance(state, dict):
            return None
        return {
            "fit_support": int(state.get("fit_support", 0) or 0),
            "accepted_count": int(state.get("accepted_count", 0) or 0),
            "accepted_rate": float(state.get("accepted_rate", 0.0) or 0.0),
            "wrong_accept_count": int(state.get("wrong_accept_count", 0) or 0),
            "empirical_error_at_lambda": state.get("empirical_error_at_lambda"),
            "u_at_lambda": state.get("u_at_lambda"),
            "observed_accuracy_at_lambda": state.get("observed_accuracy_at_lambda"),
            "lambda_hat": (
                None
                if state.get("lambda_hat") is None
                else float(state.get("lambda_hat", 0.0) or 0.0)
            ),
            "lambda1_hat": (
                None
                if state.get("lambda1_hat") is None
                else float(state.get("lambda1_hat", 0.0) or 0.0)
            ),
            "lambda2_hat": (
                None
                if state.get("lambda2_hat") is None
                else float(state.get("lambda2_hat", 0.0) or 0.0)
            ),
            "selected_r_reach2": (
                None
                if state.get("selected_r_reach2") is None
                else float(state.get("selected_r_reach2", 0.0) or 0.0)
            ),
            "selected_r3": (
                None
                if state.get("selected_r3") is None
                else float(state.get("selected_r3", 0.0) or 0.0)
            ),
            "selected_cost": (
                None
                if state.get("selected_cost") is None
                else float(state.get("selected_cost", 0.0) or 0.0)
            ),
            "certified_candidate_count": int(state.get("certified_candidate_count", 0) or 0),
            "candidate_count": int(state.get("candidate_count", 0) or 0),
            "tau": float(state.get("tau", self.tau) or self.tau),
            "epsilon": float(state.get("epsilon", self.epsilon) or self.epsilon),
            "delta_w": float(state.get("delta_w", self.delta_w) or self.delta_w),
            "n_tracks": int(state.get("n_tracks", self.n_tracks) or self.n_tracks),
            "eta_f": float(
                state.get("eta_f", self.eta_f)
                or self.eta_f
            ),
            "enabled": bool(state.get("enabled", False)),
            "disable_reason": str(state.get("disable_reason", "")),
            "method": str(state.get("method", self.calibration_method)),
            "lambda_search_method": str(
                state.get("lambda_search_method", self.lambda_search_method)
            ),
            "threshold_function": str(state.get("threshold_function", self.threshold_function)),
            "combiner_source": str(state.get("combiner_source", self.combiner_source)),
            "calibration_level": str(state.get("calibration_level", self.calibration_level)),
            "requested_calibration_level": str(
                state.get("requested_calibration_level", self.calibration_level)
            ),
            "population_size": int(state.get("population_size", state.get("fit_support", 0)) or 0),
            "population_group_count": int(
                state.get(
                    "population_group_count",
                    state.get("population_size", state.get("fit_support", 0)),
                )
                or 0
            ),
            "wsr": dict(state.get("wsr", {}) or {}),
            "combiner_params": dict(
                state.get("combiner_params", self._combiner_params_snapshot()) or {}
            ),
            "lambda1_max": (
                None
                if state.get("lambda1_max") is None
                else float(state.get("lambda1_max", 0.0) or 0.0)
            ),
            "I2_max_size": int(state.get("I2_max_size", 0) or 0),
            "I2_max_stage2_prediction_count": int(
                state.get("I2_max_stage2_prediction_count", 0) or 0
            ),
        }

    def snapshot(self) -> Dict[str, Any]:

        threshold_state = self._normalized_threshold_state()
        summary = {
            "enabled": bool((threshold_state or {}).get("enabled", False)),
            "signal_quality_distribution": dict(self._signal_quality_distribution),
            "stage1_neighbor_support_summary": dict(self._stage1_neighbor_support_summary),
            "neighbor_support_summary": dict(self._neighbor_support_summary),
            "stage1_tilde_b_summary": dict(self._stage1_tilde_b_summary),
            "stage2_tilde_b_summary": dict(self._stage2_tilde_b_summary),
        }
        return {
            "is_calibrated": bool(self._calibrated),
            "mode": str(self.mode),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "calibration_method": str(self.calibration_method),
            "tau": float(self.tau),
            "epsilon": float(self.epsilon),
            "delta_w": float(self.delta_w),
            "n_tracks": int(self.n_tracks),
            "eta_f": float(self.eta_f),
            "calibration_level": str(self.calibration_level),
            "combiner_source": str(self.combiner_source),
            "combiner_theta_grid": self._combiner_theta_candidates(),
            "combiner_params": self._combiner_params_snapshot(),
            "neighbor_support_bank": self._stage2_support_bank.snapshot(),
            "neighbor_support_banks": {
                "stage1": self._stage1_support_bank.snapshot(),
                "stage2": self._stage2_support_bank.snapshot(),
            },
            "signal_quality_distribution": dict(self._signal_quality_distribution),
            "stage1_neighbor_support_summary": dict(self._stage1_neighbor_support_summary),
            "neighbor_support_summary": dict(self._neighbor_support_summary),
            "stage1_tilde_b_summary": dict(self._stage1_tilde_b_summary),
            "stage2_tilde_b_summary": dict(self._stage2_tilde_b_summary),
            "threshold_state": threshold_state,
            "summary": summary,
        }

    def _route_stage_fields(self, sample: Sample, *, decision_stage: str = "") -> Dict[str, Any]:
        stage1_pred = self._stage1_prediction(sample)
        stage2_pred = self._stage2_prediction(sample)
        accepted_prediction = None
        if decision_stage == "stage1":
            accepted_prediction = stage1_pred
        elif decision_stage == "stage2":
            accepted_prediction = stage2_pred
        return {
            "stage1_pred": stage1_pred,
            "stage2_pred": stage2_pred,
            "stage2_prediction_available": bool(stage2_pred in {"yes", "no"}),
            "stage2_routing_score": float(self._stage2_base_score(sample)),
            "accepted_prediction": accepted_prediction,
        }

    def _eligibility_status(self, sample: Sample) -> Dict[str, Any]:

        pred = self._stage1_prediction(sample)
        if pred not in {"yes", "no"}:
            return {
                "eligible": False,
                "reason": "invalid_step1_prediction",
                "pred": pred,
                "signal_quality": str(sample.step1_signal_quality or "").strip(),
            }

        signal_quality = str(sample.step1_signal_quality or "").strip()
        if signal_quality not in STEP1_ROUTABLE_SIGNAL_QUALITIES:
            return {
                "eligible": False,
                "reason": "unsupported_signal_quality",
                "pred": pred,
                "signal_quality": signal_quality,
        }

        return {
            "eligible": True,
            "reason": "eligible",
            "pred": pred,
            "signal_quality": signal_quality,
        }

    def _algorithmic_decision(
        self,
        sample: Sample,
        *,
        pred: str,
        signal_quality: str,
        threshold_state: Dict[str, Any],
    ) -> Dict[str, Any]:

        lambda2_hat = threshold_state.get("lambda2_hat", threshold_state.get("lambda_hat"))
        if lambda2_hat is None:
            return {
                **self._route_stage_fields(sample),
                "decision": "defer",
                "reason": "lambda_unavailable",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": pred,
                "signal_quality": signal_quality,
                "routing_score": float(self._clip_unit(sample.step1_routing_score or 0.0)),
                "stage2_routing_score": float(self._stage2_base_score(sample)),
                "neighbor_support": float(self._clip_unit(sample.step2_neighbor_support or 0.5)),
                "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
                "stage2_neighbor_support": float(self._clip_unit(sample.step2_neighbor_support or 0.5)),
                "stage1_neighbor_rho": float(sample.step1_neighbor_rho or 0.0),
                "stage2_neighbor_rho": float(sample.step2_neighbor_rho or 0.0),
                "tilde_b_stage1": float(sample.step1_tilde_b),
                "tilde_b_stage2": float(sample.step2_tilde_b),
                "decision_threshold": None,
                "lambda_hat": None,
                "lambda1_hat": None,
                "lambda2_hat": None,
                "threshold_state": threshold_state,
            }

        decision_details = self._cascade_decision_details(sample, threshold_state)
        decision = "accept" if bool(decision_details["accepted"]) else "defer"
        decision_stage = str(decision_details["decision_stage"])
        reason = (
            f"accepted_{decision_stage}"
            if decision == "accept"
            else (
                "stage2_prediction_unavailable"
                if bool(decision_details["in_I2_max"])
                and not bool(decision_details["stage2_prediction_available"])
                else "tilde_b_below_stage_thresholds"
            )
        )
        return {
            **self._route_stage_fields(
                sample,
                decision_stage=decision_stage if decision == "accept" else "",
            ),
            "decision": decision,
            "reason": reason,
            "decision_source": "algorithmic_rule",
            "step2_eligible": True,
            "eligibility_reason": "eligible",
            "pred": pred,
            "signal_quality": signal_quality,
            "routing_score": float(self._clip_unit(sample.step1_routing_score or 0.0)),
            "stage2_routing_score": float(self._stage2_base_score(sample)),
            "neighbor_support": float(self._clip_unit(sample.step2_neighbor_support or 0.5)),
            "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
            "stage2_neighbor_support": float(self._clip_unit(sample.step2_neighbor_support or 0.5)),
            "stage1_neighbor_k_eff": float(sample.step1_neighbor_k_eff or 0.0),
            "stage2_neighbor_k_eff": float(sample.step2_neighbor_k_eff or 0.0),
            "stage1_neighbor_bar_w": float(sample.step1_neighbor_bar_w or 0.0),
            "stage2_neighbor_bar_w": float(sample.step2_neighbor_bar_w or 0.0),
            "stage1_neighbor_rho": float(sample.step1_neighbor_rho or 0.0),
            "stage2_neighbor_rho": float(sample.step2_neighbor_rho or 0.0),
            "tilde_b_stage1": float(sample.step1_tilde_b),
            "tilde_b_stage2": float(sample.step2_tilde_b),
            "decision_stage": decision_stage,
            "stage1_accept": bool(decision_details["stage1_accept"]),
            "stage2_accept": bool(decision_details["stage2_accept"]),
            "in_I2_max": bool(decision_details["in_I2_max"]),
            "stage2_prediction_available": bool(decision_details["stage2_prediction_available"]),
            "stage1_decision_threshold": decision_details["stage1_decision_threshold"],
            "stage2_decision_threshold": decision_details["stage2_decision_threshold"],
            "decision_threshold": decision_details["decision_threshold"],
            "lambda_hat": float(lambda2_hat),
            "lambda1_hat": (
                None
                if threshold_state.get("lambda1_hat") is None
                else float(threshold_state.get("lambda1_hat", 0.0) or 0.0)
            ),
            "lambda2_hat": float(lambda2_hat),
            "threshold_state": threshold_state,
        }

    def decide(self, sample: Sample) -> Dict[str, Any]:

        self._annotate_tilde_b([sample])
        eligibility = self._eligibility_status(sample)
        routing_score = float(self._clip_unit(sample.step1_routing_score or 0.0))
        neighbor_support = float(self._clip_unit(sample.step2_neighbor_support or 0.5))
        if not bool(eligibility.get("eligible", False)):
            return {
                **self._route_stage_fields(sample),
                "decision": "defer",
                "reason": str(eligibility.get("reason", "step2_ineligible")),
                "decision_source": "eligibility_gate",
                "step2_eligible": False,
                "eligibility_reason": str(eligibility.get("reason", "step2_ineligible")),
                "pred": str(eligibility.get("pred", "") or ""),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "stage2_routing_score": float(self._stage2_base_score(sample)),
                "neighbor_support": neighbor_support,
                "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
                "stage2_neighbor_support": neighbor_support,
                "tilde_b_stage1": float(sample.step1_tilde_b),
                "tilde_b_stage2": float(sample.step2_tilde_b),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        if not self._calibrated:
            return {
                **self._route_stage_fields(sample),
                "decision": "defer",
                "reason": "calibration_required",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": str(eligibility.get("pred", "") or ""),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "stage2_routing_score": float(self._stage2_base_score(sample)),
                "neighbor_support": neighbor_support,
                "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
                "stage2_neighbor_support": neighbor_support,
                "tilde_b_stage1": float(sample.step1_tilde_b),
                "tilde_b_stage2": float(sample.step2_tilde_b),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        threshold_state = self._threshold_state
        if threshold_state is None:
            return {
                **self._route_stage_fields(sample),
                "decision": "defer",
                "reason": "threshold_state_missing",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": str(eligibility.get("pred", "") or ""),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "stage2_routing_score": float(self._stage2_base_score(sample)),
                "neighbor_support": neighbor_support,
                "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
                "stage2_neighbor_support": neighbor_support,
                "tilde_b_stage1": float(sample.step1_tilde_b),
                "tilde_b_stage2": float(sample.step2_tilde_b),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        if not bool(threshold_state.get("enabled", False)):
            return {
                **self._route_stage_fields(sample),
                "decision": "defer",
                "reason": "threshold_state_disabled",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": str(eligibility.get("pred", "") or ""),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "stage2_routing_score": float(self._stage2_base_score(sample)),
                "neighbor_support": neighbor_support,
                "stage1_neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.5)),
                "stage2_neighbor_support": neighbor_support,
                "tilde_b_stage1": float(sample.step1_tilde_b),
                "tilde_b_stage2": float(sample.step2_tilde_b),
                "disable_reason": str(threshold_state.get("disable_reason", "")),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": threshold_state,
            }

        return self._algorithmic_decision(
            sample,
            pred=str(eligibility.get("pred", "") or ""),
            signal_quality=str(eligibility.get("signal_quality", "") or ""),
            threshold_state=threshold_state,
        )


__all__ = [
    "NeighborSupportBank",
    "Step2WSR",
]
