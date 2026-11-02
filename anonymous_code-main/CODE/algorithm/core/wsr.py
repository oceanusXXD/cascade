

from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np
from scipy.stats import hypergeom


_EPS = 1e-12


def hypergeometric_upper_bound(
    x: int,
    *,
    M: int,
    n_A: int,
    alpha: float,
    complement_size: int,
) -> tuple[int, float]:

    x = int(x)
    M = int(M)
    n_A = int(n_A)
    complement_size = int(complement_size)
    alpha = float(alpha)

    lo, hi = x, M - n_A + x

    K_plus = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if float(hypergeom.cdf(x, M, mid, n_A)) >= alpha:
            K_plus = mid
            lo = mid + 1
        else:
            hi = mid - 1

    numerator = max(0, K_plus - x)
    U_A = min(1.0, numerator / complement_size) if complement_size > 0 else 1.0
    return K_plus, U_A


def hypergeometric_population_upper_bound(
    x: int,
    *,
    M: int,
    n: int,
    alpha: float,
) -> tuple[int, float]:


    K_plus, _ = hypergeometric_upper_bound(
        int(x),
        M=int(M),
        n_A=int(n),
        alpha=float(alpha),
        complement_size=max(1, int(M) - int(n)),
    )
    population_size = int(M)
    upper_bound = min(1.0, float(K_plus) / float(population_size)) if population_size > 0 else 1.0
    return K_plus, upper_bound


def _betting_log_capital(
    values: np.ndarray,
    *,
    population_mean_null: float,
    population_size: int,
    betting_clip: float = 0.5,
) -> float:


    sample = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    sample_size = int(sample.size)
    population_n = max(sample_size, int(population_size))
    u = clip_unit(float(population_mean_null))
    t = np.arange(1, sample_size + 1, dtype=np.float64)
    remaining = float(population_n) - t + 1.0
    active = remaining > 0.0
    if not np.any(active):
        return 0.0
    prefix_sum = np.concatenate(([0.0], np.cumsum(sample, dtype=np.float64)[:-1]))
    null_remaining_mean = np.clip(
        ((float(population_n) * u) - prefix_sum[active]) / remaining[active],
        0.0,
        1.0,
    )
    # Paper Eq. (9), predictable aGRAPA moments:
    #   mu_hat_{t-1} = (1/2 + sum_{k<t} X_k) / t
    #   sigma_hat^2_{t-1} = (1/4 + sum_{k<t}(X_k-mu_hat_{k-1})^2) / t.
    # The variance residual is centered at the predictable running mean for
    # observation k, not at the residual null target q_k(u).
    predictable_mu = (0.5 + prefix_sum) / t
    predictable_residual_sq = (sample - predictable_mu) ** 2
    predictable_residual_prefix = np.concatenate(
        ([0.0], np.cumsum(predictable_residual_sq, dtype=np.float64)[:-1])
    )
    active_t = t[active]
    sigma_hat_sq = np.maximum(
        (0.25 + predictable_residual_prefix[active]) / active_t,
        _EPS,
    )
    mu_hat = predictable_mu[active]
    raw_bets = (null_remaining_mean - mu_hat) / sigma_hat_sq
    clip_value = max(0.0, float(betting_clip))
    upper = np.full_like(null_remaining_mean, np.inf, dtype=np.float64)
    not_one = null_remaining_mean < 1.0 - _EPS
    upper[not_one] = clip_value / np.maximum(1.0 - null_remaining_mean[not_one], _EPS)
    bets = np.clip(np.where(np.isfinite(raw_bets), raw_bets, 0.0), 0.0, upper)
    factors = 1.0 + (bets * (null_remaining_mean - sample[active]))
    if np.any(factors <= 0.0):
        return -math.inf
    return float(np.sum(np.log(np.maximum(factors, _EPS)), dtype=np.float64))


def _betting_mixture_log_capital(
    values: np.ndarray,
    *,
    population_mean_null: float,
    population_size: int,
    lambdas: np.ndarray,
) -> float:


    sample = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, 1.0)
    sample_size = int(sample.size)
    population_n = max(sample_size, int(population_size))
    u = clip_unit(float(population_mean_null))
    offsets = np.arange(1, sample_size + 1, dtype=np.float64)
    remaining = float(population_n) - offsets + 1.0
    active = remaining > 0.0
    if not np.any(active):
        return 0.0
    prefix_sum = np.concatenate(([0.0], np.cumsum(sample, dtype=np.float64)[:-1]))
    null_remaining_mean = np.clip(
        ((float(population_n) * u) - prefix_sum[active]) / remaining[active],
        0.0,
        1.0,
    )
    factors = 1.0 + (lambdas[None, :] * (null_remaining_mean[:, None] - sample[active, None]))
    if np.any(factors <= 0.0):
        return -math.inf
    log_capitals = np.sum(np.log(np.maximum(factors, _EPS)), axis=0, dtype=np.float64)

    max_log = float(np.max(log_capitals)) if log_capitals.size else 0.0
    if not math.isfinite(max_log):
        return max_log
    weights = np.full(int(lambdas.size), 1.0 / float(lambdas.size), dtype=np.float64)
    weighted = np.sum(weights * np.exp(np.clip(log_capitals - max_log, -745.0, 700.0)))
    return float(max_log + math.log(max(float(weighted), _EPS)))



def betting_log_wealth(
    values: Iterable[float],
    *,
    M: int,
    u: float,
    lambdas: Iterable[float] | None = None,
) -> float:

    if isinstance(values, np.ndarray):
        sample = values.astype(np.float64, copy=False)
    else:
        sample = np.asarray(list(values), dtype=np.float64)
    sample = np.clip(np.where(np.isfinite(sample), sample, 0.0), 0.0, 1.0)
    population_n = max(int(sample.size), int(M))
    if sample.size <= 0 or population_n <= 0:
        return 0.0
    if lambdas is None:
        return _betting_log_capital(
            sample,
            population_mean_null=float(u),
            population_size=population_n,
        )
    lambda_values = np.asarray(list(lambdas), dtype=np.float64)
    lambda_values = lambda_values[np.isfinite(lambda_values)]
    lambda_values = lambda_values[(lambda_values > 0.0) & (lambda_values < 1.0)]
    if lambda_values.size == 0:
        raise ValueError("betting lambdas must contain at least one finite value in (0, 1)")
    return _betting_mixture_log_capital(
        sample,
        population_mean_null=float(u),
        population_size=population_n,
        lambdas=lambda_values,
    )

def betting_population_upper_bound(
    values: Iterable[float],
    *,
    M: int,
    alpha: float,
    lambdas: Iterable[float] | None = None,
    tol: float = 1e-6,
    max_iter: int = 80,
    scan_points: int | None = None,
) -> tuple[int, float]:


    if isinstance(values, np.ndarray):
        sample = values.astype(np.float64, copy=False)
    else:
        sample = np.asarray(list(values), dtype=np.float64)
    sample = np.clip(np.where(np.isfinite(sample), sample, 0.0), 0.0, 1.0)
    population_n = max(int(sample.size), int(M))
    alpha_value = max(_EPS, min(1.0 - _EPS, float(alpha)))
    if sample.size <= 0 or population_n <= 0:
        return population_n, 1.0

    if lambdas is None:
        def log_capital_at(population_mean_null: float) -> float:
            return _betting_log_capital(
                sample,
                population_mean_null=float(population_mean_null),
                population_size=population_n,
            )

    else:
        lambda_values = np.asarray(list(lambdas), dtype=np.float64)
        lambda_values = lambda_values[np.isfinite(lambda_values)]
        lambda_values = lambda_values[(lambda_values > 0.0) & (lambda_values < 1.0)]
        if lambda_values.size == 0:
            raise ValueError("betting lambdas must contain at least one finite value in (0, 1)")

        def log_capital_at(population_mean_null: float) -> float:
            return _betting_mixture_log_capital(
                sample,
                population_mean_null=float(population_mean_null),
                population_size=population_n,
                lambdas=lambda_values,
            )

    log_boundary = math.log(1.0 / alpha_value)
    if log_capital_at(1.0) < log_boundary:
        return population_n, 1.0

    lo = 0.0
    hi = 1.0
    for _ in range(max(1, int(max_iter))):
        mid = (lo + hi) / 2.0
        log_capital = log_capital_at(mid)
        if log_capital >= log_boundary:
            hi = mid
        else:
            lo = mid
        if hi - lo <= float(tol):
            break

    upper_bound = min(1.0, max(0.0, float(hi)))
    if scan_points is None:
        raw_scan_points = os.environ.get("CASCADE_WSR_INVERSION_SCAN_POINTS", "")
        if str(raw_scan_points or "").strip():
            try:
                scan_points = int(str(raw_scan_points).strip())
            except (TypeError, ValueError):
                scan_points = int(max_iter)
        else:
            scan_points = int(max_iter)
    scan_count = max(2, int(scan_points))
    if scan_count > 2 and upper_bound < 1.0:
        step = 1.0 / float(scan_count - 1)
        scan_start = max(0.0, upper_bound - step)
        last_non_rejected: float | None = None
        for candidate_u in np.linspace(scan_start, 1.0, scan_count, dtype=np.float64):
            if log_capital_at(float(candidate_u)) < log_boundary:
                last_non_rejected = float(candidate_u)
        if last_non_rejected is not None:
            upper_bound = max(upper_bound, min(1.0, last_non_rejected + step))
    upper_count = int(math.ceil(upper_bound * float(population_n)))
    return upper_count, upper_bound


def clip_unit(value: float) -> float:


    item = float(value or 0.0)
    if math.isnan(item) or item <= 0.0:
        return 0.0
    if item >= 1.0:
        return 1.0
    return item


__all__ = [
    "betting_log_wealth",
    "betting_population_upper_bound",
    "clip_unit",
    "hypergeometric_population_upper_bound",
    "hypergeometric_upper_bound",
]
