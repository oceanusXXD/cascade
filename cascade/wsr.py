"""WS&R at epsilon, fixed-sequence within lambda1 tracks, Bonferroni across tracks."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .data import read_json, read_rows, write_json, write_rows


def betting_log_wealth(errors, population_size, epsilon):
    """Paper Eq. (9), with residual centering and predictable aGRAPA bets."""
    errors = np.asarray(errors, dtype=float)
    t = np.arange(1, len(errors) + 1, dtype=float)
    prefix = np.r_[0.0, np.cumsum(errors)[:-1]]
    q = np.clip((population_size * epsilon - prefix) / (population_size - t + 1), 0, 1)
    mu = (0.5 + prefix) / t
    residuals = (errors - mu) ** 2
    variance = (0.25 + np.r_[0.0, np.cumsum(residuals)[:-1]]) / t
    cap = np.divide(0.5, 1 - q, out=np.full_like(q, np.inf), where=q < 1)
    bets = np.minimum(np.maximum((q - mu) / variance, 0), cap)
    return float(np.log1p(bets * (q - errors)).sum())


def aligned_stages(population, stage1, stage2, grid1):
    ids = [row["id"] for row in population]
    first = {row["id"]: row for row in stage1}
    second = {row["id"]: row for row in stage2}
    if set(first) != set(ids):
        raise ValueError("Stage-1 IDs must match the population")
    expected = {key for key in ids if first[key]["score"] is None or first[key]["score"] < max(grid1)}
    if set(second) != expected:
        raise ValueError("Stage-2 IDs must match the maximal pool defined by grid1")

    def arrays(mapping):
        scores = np.array([mapping[key]["score"] if key in mapping and mapping[key]["score"] is not None
                           and mapping[key]["prediction"] in {"yes", "no"} else -np.inf for key in ids])
        predictions = np.array([mapping.get(key, {}).get("prediction") for key in ids], dtype=object)
        return scores, predictions

    return first, second, *arrays(first), *arrays(second)


def model_cost(row, prices):
    return (row["input_tokens"] * prices["input_price"] + row["output_tokens"] * prices["output_price"]) / 1e6


def joint_search(population, stage1, stage2, calibration, oracle, grid1, grid2, config):
    tau, delta = config["tau"], config["delta"]
    if not 0 < tau < 1 or not 0 < delta < 1:
        raise ValueError("tau and delta must be in (0, 1)")
    n_population = len(population)
    if not 1 <= len(calibration) <= n_population:
        raise ValueError("Calibration size must be between 1 and N")
    first, second, score1, pred1, score2, pred2 = aligned_stages(population, stage1, stage2, grid1)
    ids = [row["id"] for row in population]
    positions = {key: i for i, key in enumerate(ids)}
    labels = {row["id"]: row["prediction"] for row in oracle}
    if set(labels) != {row["id"] for row in calibration} or not set(labels).issubset(positions):
        raise ValueError("Oracle labels must cover exactly the calibration sample")
    if any(label not in {"yes", "no"} for label in labels.values()):
        raise ValueError("Each calibration record needs a binary Oracle prediction")
    # Retain the original random draw order, even if the label file was reordered.
    sample = np.array([positions[row["id"]] for row in calibration])
    truth = np.array([labels[row["id"]] for row in calibration])
    wrong1, wrong2 = pred1[sample] != truth, pred2[sample] != truth
    fixed_stage1_cost = sum(model_cost(row, config["slm"]) for row in stage1)
    stage2_costs = np.array([model_cost(second[key], config["slm"]) if key in second else 0 for key in ids])
    # Oracle input lengths use the SLM tokenizer; output lengths are fixed before labels are read.
    prices = config["oracle"]
    oracle_costs = np.array([(first[key]["full_input_tokens"] * prices["input_price"] +
                             prices["expected_output_tokens"] * prices["output_price"]) / 1e6 for key in ids])
    tracks = len(grid1)
    boundary = math.log(tracks / delta)
    candidates, certified = [], []
    for lambda1 in sorted(grid1, reverse=True):
        accept1 = score1 >= lambda1
        for lambda2 in sorted(grid2, reverse=True):
            accept2 = (~accept1) & (score2 >= lambda2)
            defer = ~(accept1 | accept2)
            errors = (accept1[sample] & wrong1) | (accept2[sample] & wrong2)
            wealth = betting_log_wealth(errors, n_population, 1 - tau)
            candidate = {"lambda1": lambda1, "lambda2": lambda2,
                         "certified": wealth >= boundary, "log_wealth": wealth,
                         "sample_error": float(errors.mean()),
                         "stage1_count": int(accept1.sum()), "stage2_count": int(accept2.sum()),
                         "oracle_count": int(defer.sum()), "oracle_rate": float(defer.mean()),
                         "estimated_deployment_cost": float(fixed_stage1_cost + stage2_costs[~accept1].sum() + oracle_costs[defer].sum())}
            candidates.append(candidate)
            if not candidate["certified"]:
                break
            certified.append(candidate)
    selected = min(certified, key=lambda row: (row["estimated_deployment_cost"], row["oracle_count"])) if certified else None
    policy = {"tau": tau, "delta": delta, "population_size": n_population,
              "calibration_size": len(calibration), "tracks": tracks, "level_per_track": delta / tracks,
              "grid_shape": [len(grid1), len(grid2)], "certified_count": len(certified),
              "stage2_search_cost": float(stage2_costs.sum()),
              "calibration_cost": sum(model_cost(row, config["oracle"]) for row in oracle),
              "fallback": "all_oracle" if selected is None else None, "selected": selected}
    return policy, candidates


def route(population, stage1, stage2, policy, oracle_calibration):
    first = {row["id"]: row for row in stage1}
    second = {row["id"]: row for row in stage2}
    oracle = {row["id"]: row["prediction"] for row in oracle_calibration}
    selected = policy["selected"]
    accepted, pending = [], []
    for row in population:
        key = row["id"]
        prediction, stage = None, None
        for name, mapping, threshold_name in (("stage1", first, "lambda1"), ("stage2", second, "lambda2")):
            value = mapping.get(key, {})
            score = value.get("score")
            if selected and score is not None and score >= selected[threshold_name] and value.get("prediction") in {"yes", "no"}:
                prediction, stage = value["prediction"], name
                break
        if prediction is None and key in oracle:
            prediction, stage = oracle[key], "oracle"
        if prediction is None:
            pending.append(row)
        else:
            accepted.append({"id": key, "prediction": prediction, "stage": stage})
    return accepted, pending


def merge(population, accepted, oracle):
    population_ids = {row["id"] for row in population}
    accepted_ids, oracle_ids = {row["id"] for row in accepted}, {row["id"] for row in oracle}
    if accepted_ids & oracle_ids or (accepted_ids | oracle_ids) != population_ids:
        raise ValueError("Accepted and residual Oracle results must partition the population")
    by_id = {row["id"]: row for row in accepted}
    by_id.update({row["id"]: {"id": row["id"], "prediction": row["prediction"], "stage": "oracle"} for row in oracle})
    results = [by_id[row["id"]] for row in population]
    if any(row["prediction"] not in {"yes", "no"} for row in results):
        raise ValueError("Every final record needs a binary prediction")
    summary = {"population_size": len(results), "stage_counts": dict(Counter(row["stage"] for row in results))}
    if population and all(row.get("label") in {"yes", "no"} for row in population):
        truth = np.array([row["label"] for row in population])
        predictions = np.array([row["prediction"] for row in results])
        summary["accuracy"] = float(np.mean(truth == predictions))
        f1 = []
        for label in ("yes", "no"):
            tp = int(((truth == label) & (predictions == label)).sum())
            denominator = int((truth == label).sum() + (predictions == label).sum())
            f1.append(2 * tp / denominator if denominator else 0)
        summary["macro_f1"] = sum(f1) / 2
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Certify tracks and choose their cheapest certified pair")
    fit.add_argument("--population", required=True)
    fit.add_argument("--stage1", required=True)
    fit.add_argument("--stage2", required=True)
    fit.add_argument("--grid1", required=True)
    fit.add_argument("--grid2", required=True)
    fit.add_argument("--calibration", required=True)
    fit.add_argument("--oracle-labels", required=True)
    fit.add_argument("--config", default="config.json")
    fit.add_argument("--output-dir", required=True)
    routing = commands.add_parser("route", help="Apply the policy and prepare residual Oracle inputs")
    routing.add_argument("--population", required=True)
    routing.add_argument("--stage1", required=True)
    routing.add_argument("--stage2", required=True)
    routing.add_argument("--policy", required=True)
    routing.add_argument("--oracle-labels", required=True)
    routing.add_argument("--output-dir", required=True)
    merging = commands.add_parser("merge", help="Assemble final predictions in population order")
    merging.add_argument("--population", required=True)
    merging.add_argument("--accepted", required=True)
    merging.add_argument("--oracle", required=True)
    merging.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    population = read_rows(args.population)
    root = Path(args.output_dir)
    if args.command == "fit":
        grid1, grid2 = read_json(args.grid1)["thresholds"], read_json(args.grid2)["thresholds"]
        policy, candidates = joint_search(population, read_rows(args.stage1), read_rows(args.stage2),
                                         read_rows(args.calibration), read_rows(args.oracle_labels),
                                         grid1, grid2, read_json(args.config))
        write_json(root / "policy.json", policy)
        write_rows(root / "candidates.jsonl", candidates)
        print(f"Certified {policy['certified_count']} candidates; selected: {policy['selected']}")
    elif args.command == "route":
        accepted, pending = route(population, read_rows(args.stage1), read_rows(args.stage2),
                                  read_json(args.policy), read_rows(args.oracle_labels))
        write_rows(root / "accepted.jsonl", accepted)
        write_rows(root / "pending_oracle.jsonl", pending)
        print(f"Resolved {len(accepted)}; residual Oracle calls {len(pending)}")
    else:
        results, summary = merge(population, read_rows(args.accepted), read_rows(args.oracle))
        write_rows(root / "predictions.jsonl", results)
        write_json(root / "summary.json", summary)
        print(summary)


if __name__ == "__main__":
    main()
