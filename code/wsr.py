"""Joint threshold certification and routing."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from betting import betting_log_wealth
from common import read_json, read_rows, write_json, write_rows


def threshold_grid(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("A threshold grid must be a nonempty finite list")
    return np.unique(values).tolist()


def aligned_stages(population, stage1, stage2, grid1):
    ids = [row["id"] for row in population]
    first = {row["id"]: row for row in stage1}
    second = {row["id"]: row for row in stage2}
    if len(first) != len(stage1) or len(second) != len(stage2) or set(first) != set(ids):
        raise ValueError("Stage-1 IDs must match the population")

    def arrays(mapping):
        scores, predictions = [], []
        for key in ids:
            row = mapping.get(key, {})
            score, prediction = row.get("score"), row.get("prediction")
            if score is not None and (not math.isfinite(score) or prediction not in {"yes", "no"}):
                raise ValueError(f"{key}: a finite score requires a binary prediction")
            scores.append(score if score is not None else -np.inf)
            predictions.append(prediction)
        return np.asarray(scores), np.asarray(predictions, dtype=object)

    score1, pred1 = arrays(first)
    lambda1_max = max(threshold_grid(grid1))
    expected = {key for key, score in zip(ids, score1) if score < lambda1_max}
    if set(second) != expected:
        raise ValueError("Stage-2 IDs must match the maximal pool defined by grid1")
    return first, second, score1, pred1, *arrays(second)


def model_cost(row, input_price, output_price):
    return (row["input_tokens"] * input_price + row["output_tokens"] * output_price) / 1e6


def prepare_search(population, stage1, stage2, grid1, grid2, *, slm_input_price=0.2, slm_output_price=0.2,
                   oracle_input_price=5, oracle_output_price=30, oracle_output_tokens=64,
                   oracle_inputs=None):
    grid1, grid2 = threshold_grid(grid1), threshold_grid(grid2)
    if not population:
        raise ValueError("The population is empty")
    first, second, score1, _, score2, _ = aligned_stages(population, stage1, stage2, grid1)
    ids = [row["id"] for row in population]
    fixed_stage1_cost = sum(model_cost(row, slm_input_price, slm_output_price) for row in stage1)
    stage2_costs = np.array([model_cost(second[key], slm_input_price, slm_output_price) if key in second else 0 for key in ids])
    if oracle_inputs is not None:
        input_counts = {row["id"]: row["input_tokens"] for row in oracle_inputs}
        if set(input_counts) != set(ids):
            raise ValueError("Oracle input token estimates must cover the population")
    else:
        input_counts = {key: first[key]["full_input_tokens"] for key in ids}
    oracle_costs = np.array([(input_counts[key] * oracle_input_price +
                             oracle_output_tokens * oracle_output_price) / 1e6 for key in ids])
    if not np.isfinite([fixed_stage1_cost, *stage2_costs, *oracle_costs]).all() or (
            fixed_stage1_cost < 0 or np.any(stage2_costs < 0) or np.any(oracle_costs < 0)):
        raise ValueError("Per-record cost estimates must be finite and nonnegative")
    candidates = []
    for lambda1 in sorted(grid1, reverse=True):
        accept1 = score1 >= lambda1
        for lambda2 in sorted(grid2, reverse=True):
            accept2 = (~accept1) & (score2 >= lambda2)
            defer = ~(accept1 | accept2)
            candidates.append({"lambda1": lambda1, "lambda2": lambda2,
                               "stage1_count": int(accept1.sum()), "stage2_count": int(accept2.sum()),
                               "oracle_count": int(defer.sum()), "oracle_rate": float(defer.mean()),
                               "estimated_plan_cost": float(fixed_stage1_cost + stage2_costs[~accept1].sum()
                                                            + oracle_costs[defer].sum())})
    return {"population_ids": ids, "lambda1_grid": grid1, "lambda2_grid": grid2,
            "stage1_cost": fixed_stage1_cost, "stage2_search_cost": float(stage2_costs.sum()),
            "cost_assumptions": {"slm_input_price": slm_input_price, "slm_output_price": slm_output_price,
                                 "oracle_input_price": oracle_input_price, "oracle_output_price": oracle_output_price,
                                 "oracle_output_tokens": oracle_output_tokens,
                                 "oracle_input_source": "provided_counts" if oracle_inputs is not None else "slm_tokenizer_proxy"},
            "candidates": candidates}


def joint_search(population, stage1, stage2, calibration, oracle, prepared, *, tau=0.9, delta=0.05):
    if not 0 < tau < 1 or not 0 < delta < 1:
        raise ValueError("tau and delta must be in (0, 1)")
    grid1 = threshold_grid(prepared["lambda1_grid"])
    grid2 = threshold_grid(prepared["lambda2_grid"])
    n_population = len(population)
    if not 1 <= len(calibration) <= n_population:
        raise ValueError("Calibration size must be between 1 and N")
    _, _, score1, pred1, score2, pred2 = aligned_stages(population, stage1, stage2, grid1)
    ids = [row["id"] for row in population]
    if prepared["population_ids"] != ids:
        raise ValueError("The population IDs and order must match preparation")
    fixed_candidates = {(row["lambda1"], row["lambda2"]): row for row in prepared["candidates"]}
    if set(fixed_candidates) != {(a, b) for a in grid1 for b in grid2}:
        raise ValueError("Prepared candidates must cover the frozen grid")
    positions = {key: i for i, key in enumerate(ids)}
    labels = {row["id"]: row["prediction"] for row in oracle}
    if set(labels) != {row["id"] for row in calibration} or not set(labels).issubset(positions):
        raise ValueError("Oracle labels must cover exactly the calibration sample")
    if any(label not in {"yes", "no"} for label in labels.values()):
        raise ValueError("Each calibration record needs a binary Oracle prediction")
    for row in calibration:
        original = population[positions[row["id"]]]
        if any(row.get(field) != original.get(field) for field in ("query", "document")):
            raise ValueError(f"{row['id']}: calibration text differs from the population")
    sample = np.array([positions[row["id"]] for row in calibration])
    truth = np.array([labels[row["id"]] for row in calibration])
    sample_score1, sample_score2 = score1[sample], score2[sample]
    wrong1, wrong2 = pred1[sample] != truth, pred2[sample] != truth
    tracks = len(grid1)
    boundary = math.log(tracks / delta)
    candidates, certified = [], []
    for lambda1 in sorted(grid1, reverse=True):
        accept1 = sample_score1 >= lambda1
        for lambda2 in sorted(grid2, reverse=True):
            accept2 = (~accept1) & (sample_score2 >= lambda2)
            errors = (accept1 & wrong1) | (accept2 & wrong2)
            wealth = betting_log_wealth(errors, n_population, 1 - tau)
            candidate = {**fixed_candidates[lambda1, lambda2],
                         "certified": wealth >= boundary, "log_wealth": wealth,
                         "sample_error": float(errors.mean())}
            candidates.append(candidate)
            if not candidate["certified"]:
                break
            certified.append(candidate)
    selected = min(certified, key=lambda row: (row["estimated_plan_cost"], row["oracle_count"])) if certified else None
    prices = prepared["cost_assumptions"]
    policy = {"tau": tau, "delta": delta, "population_size": n_population,
              "population_ids": ids,
              "calibration_size": len(calibration), "tracks": tracks, "level_per_track": delta / tracks,
              "calibration_ids": [row["id"] for row in calibration],
              "calibration_predictions": labels,
              "lambda1_grid": grid1, "lambda2_grid": grid2,
              "grid_shape": [len(grid1), len(grid2)], "certified_count": len(certified),
              "cost_objective": "selected_plan", "stage1_cost": prepared["stage1_cost"],
              "stage2_search_cost": prepared["stage2_search_cost"],
              "calibration_cost": sum(model_cost(row, prices["oracle_input_price"], prices["oracle_output_price"])
                                      for row in oracle),
              "cost_assumptions": prices,
              "fallback": "all_oracle" if selected is None else None, "selected": selected}
    return policy, candidates


def route(population, stage1, stage2, policy, oracle_calibration):
    if [row["id"] for row in population] != policy["population_ids"]:
        raise ValueError("Routing requires the fitted population in its original order")
    _, _, score1, pred1, score2, pred2 = aligned_stages(population, stage1, stage2, policy["lambda1_grid"])
    oracle = {row["id"]: row["prediction"] for row in oracle_calibration}
    if set(oracle) != set(policy["calibration_ids"]) or any(label not in {"yes", "no"} for label in oracle.values()):
        raise ValueError("Routing must reuse the fitted calibration sample's binary Oracle labels")
    if oracle != policy["calibration_predictions"]:
        raise ValueError("Routing must reuse the same Oracle labels used during fitting")
    selected = policy["selected"]
    if selected is None:
        accept1 = accept2 = np.zeros(len(population), dtype=bool)
    else:
        if (not selected["certified"] or selected["lambda1"] not in policy["lambda1_grid"]
                or selected["lambda2"] not in policy["lambda2_grid"]):
            raise ValueError("The selected pair must be certified and belong to the fitted grids")
        accept1 = score1 >= selected["lambda1"]
        accept2 = (~accept1) & (score2 >= selected["lambda2"])
    accepted, pending = [], []
    for i, row in enumerate(population):
        key = row["id"]
        prediction, stage = None, None
        if accept1[i]:
            prediction, stage = pred1[i], "stage1"
        elif accept2[i]:
            prediction, stage = pred2[i], "stage2"
        if prediction is None and key in oracle:
            prediction, stage = oracle[key], "oracle"
        if prediction is None:
            pending.append(row)
        else:
            accepted.append({"id": key, "prediction": prediction, "stage": stage})
    return accepted, pending


def merge(population, accepted, oracle):
    if not population:
        raise ValueError("The population is empty")
    population_ids = {row["id"] for row in population}
    accepted_ids, oracle_ids = {row["id"] for row in accepted}, {row["id"] for row in oracle}
    if accepted_ids & oracle_ids or (accepted_ids | oracle_ids) != population_ids:
        raise ValueError("Accepted and residual Oracle results must partition the population")
    by_id = {row["id"]: row for row in accepted}
    by_id.update({row["id"]: {"id": row["id"], "prediction": row["prediction"], "stage": "oracle"} for row in oracle})
    results = [by_id[row["id"]] for row in population]
    if any(row["prediction"] not in {"yes", "no"} for row in results):
        raise ValueError("Every final record needs a binary prediction")
    counts = Counter(row["stage"] for row in results)
    summary = {"population_size": len(results),
               "stage_counts": {stage: counts[stage] for stage in ("stage1", "stage2", "oracle")},
               "stage_rates": {stage: counts[stage] / len(results) for stage in ("stage1", "stage2", "oracle")}}
    labels = [row.get("label") for row in population]
    if any(label is not None for label in labels):
        if any(label not in {"yes", "no"} for label in labels):
            raise ValueError("Evaluation requires yes/no ground truth for every population record")
        truth = np.array(labels)
        predictions = np.array([row["prediction"] for row in results])
        summary["accuracy"] = float(np.mean(truth == predictions))
        f1 = []
        for label in sorted(set(truth) | set(predictions)):
            tp = int(((truth == label) & (predictions == label)).sum())
            denominator = int((truth == label).sum() + (predictions == label).sum())
            f1.append(2 * tp / denominator if denominator else 0)
        summary["macro_f1"] = sum(f1) / len(f1)
    return results, summary


def execution_summary(results, oracle_residual, oracle_calibration, policy, *, prep_cost=None, oracle_reference=None):
    """Account for each member of S union R once, at the frozen nominal prices."""
    if [row["id"] for row in results] != policy["population_ids"]:
        raise ValueError("Final results must match the fitted population")
    calibration = {row["id"]: row for row in oracle_calibration}
    residual = {row["id"]: row for row in oracle_residual}
    if {key: row["prediction"] for key, row in calibration.items()} != policy["calibration_predictions"]:
        raise ValueError("Cost accounting requires the fitted calibration labels")
    if calibration.keys() & residual.keys():
        raise ValueError("Residual Oracle calls must exclude cached calibration records")
    oracle = {**calibration, **residual}
    for row in results:
        if row["stage"] == "oracle" and (row["id"] not in oracle or oracle[row["id"]]["prediction"] != row["prediction"]):
            raise ValueError("Oracle fallback must return its cached reference label")
    prices = policy["cost_assumptions"]
    input_price, output_price = prices["oracle_input_price"], prices["oracle_output_price"]
    calibration_cost = sum(model_cost(row, input_price, output_price) for row in calibration.values())
    residual_cost = sum(model_cost(row, input_price, output_price) for row in residual.values())
    inference_cost = policy["stage1_cost"] + policy["stage2_search_cost"] + calibration_cost + residual_cost
    total_cost = inference_cost + prep_cost if prep_cost is not None else None
    selected = policy["selected"]
    summary = {
        "tau": policy["tau"], "delta": policy["delta"],
        "calibration_size": policy["calibration_size"],
        "fallback": policy["fallback"], "certified_count": policy["certified_count"],
        "oracle_usage": {"calibration_records": len(calibration), "residual_records": len(residual),
                         "unique_records": len(oracle),
                         "input_tokens": sum(row["input_tokens"] for row in oracle.values()),
                         "output_tokens": sum(row["output_tokens"] for row in oracle.values())},
        "costs_usd": {"estimated_selected_plan": selected["estimated_plan_cost"] if selected else None,
                      "stage1": policy["stage1_cost"], "stage2_preparation": policy["stage2_search_cost"],
                      "calibration_oracle": calibration_cost, "residual_oracle": residual_cost,
                      "prepared_batch_inference": inference_cost,
                      "preprocessing": prep_cost, "prepared_batch_total": total_cost,
                      "prepared_batch_inference_per_record": inference_cost / len(results),
                      "prepared_batch_total_per_record": total_cost / len(results) if total_cost is not None else None},
    }
    if oracle_reference is not None:
        reference = {row["id"]: row["prediction"] for row in oracle_reference}
        if set(reference) != {row["id"] for row in results} or any(label not in {"yes", "no"} for label in reference.values()):
            raise ValueError("Oracle agreement requires fixed binary reference labels for the full population")
        if any(reference[key] != row["prediction"] for key, row in oracle.items()):
            raise ValueError("Evaluation reference must use the same Oracle labels as calibration and fallback")
        agreement = sum(row["prediction"] == reference[row["id"]] for row in results) / len(results)
        summary.update(oracle_agreement=agreement, oracle_violation=agreement < policy["tau"])
    return summary


def summarize_runs(summaries):
    """Mean and sample standard deviation across independent calibration draws."""
    if len(summaries) < 2:
        raise ValueError("Repeated-draw reporting requires at least two summaries")
    settings = [(row["population_size"], row["tau"], row["delta"]) for row in summaries]
    if len(set(settings)) != 1:
        raise ValueError("Aggregate draws of one population and operating point at a time")
    result = {"draws": len(summaries), "population_size": settings[0][0],
              "tau": settings[0][1], "delta": settings[0][2]}
    for key in ("accuracy", "macro_f1", "oracle_agreement"):
        if all(key in row for row in summaries):
            values = np.array([row[key] for row in summaries])
            result[key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    if all("oracle_violation" in row for row in summaries):
        result["oracle_violation_count"] = sum(row["oracle_violation"] for row in summaries)
    for stage in ("stage1", "stage2", "oracle"):
        values = np.array([row["stage_rates"][stage] for row in summaries])
        result.setdefault("stage_rates", {})[stage] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    for key in ("prepared_batch_inference_per_record", "prepared_batch_total_per_record"):
        values = [row["costs_usd"][key] for row in summaries]
        if all(value is not None for value in values):
            result.setdefault("costs_usd", {})[key] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze the grid and label-free candidate plan costs before sampling",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prepare.add_argument("--population", required=True)
    prepare.add_argument("--stage1", required=True)
    prepare.add_argument("--stage2", required=True)
    prepare.add_argument("--grid1", required=True)
    prepare.add_argument("--grid2", required=True)
    prepare.add_argument("--slm-input-price", type=float, default=0.2, help="SLM price per million input tokens")
    prepare.add_argument("--slm-output-price", type=float, default=0.2, help="SLM price per million output tokens")
    prepare.add_argument("--oracle-input-price", type=float, default=5, help="Oracle price per million input tokens")
    prepare.add_argument("--oracle-output-price", type=float, default=30, help="Oracle price per million output tokens")
    prepare.add_argument("--oracle-output-tokens", type=float, default=64, help="Expected Oracle label + justification tokens, fixed before sampling")
    prepare.add_argument("--oracle-inputs", help="Optional label-free JSONL: id and Oracle-prompt input_tokens, fixed before sampling")
    prepare.add_argument("--output", required=True)
    fit = commands.add_parser("fit", help="Certify tracks and choose their cheapest certified pair",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_argument("--population", required=True)
    fit.add_argument("--stage1", required=True)
    fit.add_argument("--stage2", required=True)
    fit.add_argument("--prepared", required=True)
    fit.add_argument("--calibration", required=True)
    fit.add_argument("--oracle-labels", required=True)
    fit.add_argument("--tau", type=float, default=0.9, help="Target population agreement with the Oracle")
    fit.add_argument("--delta", type=float, default=0.05, help="Failure probability across all lambda1 tracks")
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
    merging.add_argument("--oracle-labels", required=True, help="Calibration Oracle results used during fitting")
    merging.add_argument("--policy", required=True)
    merging.add_argument("--prep-cost", type=float, help="Measured retrieval, embedding, reranking, indexing and search cost in USD for this population")
    merging.add_argument("--oracle-reference", help="Evaluation only: full-population fixed Oracle labels, never used for selection")
    merging.add_argument("--output-dir", required=True)
    aggregate = commands.add_parser("summarize", help="Aggregate metrics over independent calibration draws of one setting")
    aggregate.add_argument("--summaries", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        write_json(args.output, summarize_runs([read_json(path) for path in args.summaries]))
        print(f"Aggregated {len(args.summaries)} calibration draws -> {args.output}")
        return
    population = read_rows(args.population)
    if args.command == "prepare":
        prices = [args.slm_input_price, args.slm_output_price, args.oracle_input_price,
                  args.oracle_output_price, args.oracle_output_tokens]
        if not np.isfinite(prices).all() or min(prices) < 0:
            parser.error("prices and expected output tokens must be nonnegative")
        grid1, grid2 = read_json(args.grid1)["thresholds"], read_json(args.grid2)["thresholds"]
        prepared = prepare_search(population, read_rows(args.stage1), read_rows(args.stage2), grid1, grid2,
                                  slm_input_price=args.slm_input_price, slm_output_price=args.slm_output_price,
                                  oracle_input_price=args.oracle_input_price, oracle_output_price=args.oracle_output_price,
                                  oracle_output_tokens=args.oracle_output_tokens,
                                  oracle_inputs=read_rows(args.oracle_inputs) if args.oracle_inputs else None)
        write_json(args.output, prepared)
        print(f"Frozen {len(prepared['candidates'])} candidate plan costs -> {args.output}")
        return
    root = Path(args.output_dir)
    if args.command == "fit":
        policy, candidates = joint_search(population, read_rows(args.stage1), read_rows(args.stage2),
                                         read_rows(args.calibration), read_rows(args.oracle_labels),
                                         read_json(args.prepared), tau=args.tau, delta=args.delta)
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
        if args.prep_cost is not None and (not math.isfinite(args.prep_cost) or args.prep_cost < 0):
            parser.error("prep-cost must be finite and nonnegative")
        oracle = read_rows(args.oracle)
        results, summary = merge(population, read_rows(args.accepted), oracle)
        summary.update(execution_summary(results, oracle, read_rows(args.oracle_labels), read_json(args.policy),
                                         prep_cost=args.prep_cost,
                                         oracle_reference=read_rows(args.oracle_reference) if args.oracle_reference else None))
        write_rows(root / "predictions.jsonl", results)
        write_json(root / "summary.json", summary)
        print(summary)


if __name__ == "__main__":
    main()
