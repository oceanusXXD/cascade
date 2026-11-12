"""Without-replacement aGRAPA wealth (Appendix: Single-Candidate Betting Test)."""

import math


def log_wealth_path(errors, population_size, epsilon):
    """Yield terminal log wealth after each loss, preserving sample draw order."""
    previous_errors, squared_residuals, log_wealth = 0.0, 0.25, 0.0
    for t, error in enumerate(errors, 1):
        q = min(1.0, max(0.0, (population_size * epsilon - previous_errors) / (population_size - t + 1)))
        mean = (0.5 + previous_errors) / t
        variance = squared_residuals / t
        raw_bet = max(0.0, (q - mean) / (variance + (q - mean) ** 2))
        bet = min(raw_bet, 0.5 / (1 - q)) if q < 1 else raw_bet
        log_wealth += math.log1p(bet * (q - error))
        previous_errors += float(error)
        squared_residuals += (float(error) - mean) ** 2
        yield log_wealth


def betting_log_wealth(errors, population_size, epsilon):
    """Evaluate one fixed candidate at the fixed calibration sample size."""
    log_wealth = 0.0
    for log_wealth in log_wealth_path(errors, population_size, epsilon):
        pass
    return log_wealth


def zero_error_min_size(population_size, epsilon, boundary):
    """Compute n0 before sampling; None means the boundary is unreachable."""
    for size, wealth in enumerate(log_wealth_path((0.0 for _ in range(population_size)), population_size, epsilon), 1):
        if wealth >= boundary:
            return size
    return None
