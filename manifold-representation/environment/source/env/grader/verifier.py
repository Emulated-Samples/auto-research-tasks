#!/usr/bin/env python3
"""Production verifier for manifold-bench."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grader.metrics import match_factors, score_prediction  # noqa: E402
from grader.protocol import (  # noqa: E402
    additive_error,
    permutation_error,
    presence_contribution_agreement,
)
from grader.runner import (  # noqa: E402
    PreparedSubmission,
    classify_fault,
    preflight_sandbox,
    SubmissionError,
    prepare_submission,
    run_suite,
)
from grader.scoring import (  # noqa: E402
    CATASTROPHIC_FLOORS,
    CATEGORY_NAMES,
    CATEGORY_WEIGHTS,
    PASS_THRESHOLDS,
    TAIL_QUALITY_THRESHOLD,
    Integrity,
    aggregate_categories,
    benchmark_pass,
    breadth_adjusted_categories,
    breadth_factor,
    category_pass,
    lower_tail_quality,
    score_suite,
    suite_quality,
    weighted_category_mean,
)
from grader.specs import SCORING_VERSION, suites_for_problem  # noqa: E402
from grader.targets import (  # noqa: E402
    ENVELOPE_SIGMA,
    attainable_target,
    calibration_id,
    non_solution_floor,
    scored_metrics,
)
from grader.zoo import build_dataset  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
HARNESS_CONFIG = REPO / "grader" / "harness.json"


def grade(config: dict) -> dict:
    problem_id = str(config["problem_id"])
    submission = Path(config["submission"]).resolve()
    sandboxed = bool(config.get("sandboxed", True))
    specs = suites_for_problem(problem_id)
    prepared: PreparedSubmission | None = None
    suite_details: list[dict] = []
    all_valid = True
    try:
        if sandboxed:
            preflight_sandbox()
        try:
            prepared = prepare_submission(submission, sandboxed=sandboxed)
        except SubmissionError as error:
            all_valid = False
            for spec in specs:
                suite_details.append(
                    {
                        "suite": spec.name,
                        "suite_category": spec.category,
                        "weight": spec.weight,
                        "valid": False,
                        "error_type": type(error).__name__,
                        "contract_fault": classify_fault(str(error)),
                        "error": str(error),
                        "categories": {category: 0.0 for category in CATEGORY_NAMES},
                        "reward": 0.0,
                        "duration_seconds": 0.0,
                    }
                )
            prepared = None
        if prepared is None:
            amortized_build = 0.0
        else:
            amortized_build = prepared.build_seconds / len(specs)
        for spec in () if prepared is None else specs:
            started = time.monotonic()
            # Trusted data/reference work stays outside the submission error
            # boundary.  A fault here is a grader failure, never a candidate
            # failure silently converted into a zero.
            dataset = build_dataset(spec)
            target_metrics = attainable_target(spec)
            floor_metrics = non_solution_floor(spec)
            suite_scored = scored_metrics(spec)
            try:
                run = run_suite(prepared, dataset, sandboxed=sandboxed)
            except SubmissionError as error:
                all_valid = False
                detail = {
                    "suite": spec.name,
                    "suite_category": spec.category,
                    "weight": spec.weight,
                    "valid": False,
                    "error_type": type(error).__name__,
                    "contract_fault": classify_fault(str(error)),
                    "error": str(error),
                    "categories": {category: 0.0 for category in CATEGORY_NAMES},
                    "reward": 0.0,
                    "duration_seconds": time.monotonic() - started,
                }
                suite_details.append(detail)
                continue
            # From this point onward inputs have passed the strict protocol.
            # Scoring failures are trusted grader failures and must propagate.
            candidate_match = match_factors(dataset.match, run.match)
            candidate_metrics = score_prediction(dataset.score, run.score, candidate_match)
            integrity = Integrity(
                additive_error=additive_error(run.score, dataset.score.x),
                support_agreement=presence_contribution_agreement(run.score),
                permutation_error=permutation_error(
                    run.score, run.permuted, run.permutation_inverse
                ),
            )
            candidate_seconds = amortized_build + run.fit_seconds + run.transform_seconds
            scored = score_suite(
                candidate_metrics,
                floor_metrics,
                target_metrics,
                integrity,
                candidate_seconds,
                spec.full_credit_seconds,
                suite_scored,
            )
            detail = scored.to_json()
            # A gross additive-identity or permutation-equivariance violation is a
            # hard-contract fault, not soft partial credit: the prompt states these
            # invariants must hold, so a suite that breaks them is invalid and must
            # not report as contract_met on the strength of its category scores.
            contract_breach = integrity.gross_contract_breach()
            detail.update(
                {
                    "suite": spec.name,
                    "suite_category": spec.category,
                    "weight": spec.weight,
                    "valid": contract_breach is None,
                    "duration_seconds": time.monotonic() - started,
                    "log_tail": run.log_tail,
                }
            )
            if contract_breach is not None:
                detail["contract_fault"] = contract_breach
            suite_details.append(detail)
    finally:
        if prepared is not None:
            prepared.close()

    suite_categories = [detail["categories"] for detail in suite_details]
    raw_categories = aggregate_categories(
        [(detail["categories"], detail["weight"]) for detail in suite_details]
    )
    tail_quality = lower_tail_quality([suite_quality(scores) for scores in suite_categories])
    breadth = breadth_factor(suite_categories)
    categories = breadth_adjusted_categories(raw_categories, suite_categories)
    reward = weighted_category_mean(categories)
    # Say WHY a zero is a zero. A submission that never ran the contract and a
    # submission that ran it perfectly and recovered nothing both score 0.0, and
    # they are opposite verdicts: one is fix-your-program, the other is
    # the-science-is-hard. Report the distinction rather than making a reader
    # reconstruct it from a log tail.
    faults = [detail["contract_fault"] for detail in suite_details if not detail["valid"]]
    contract_status = {
        "valid_suites": sum(1 for detail in suite_details if detail["valid"]),
        "total_suites": len(suite_details),
        "faults": sorted(set(faults)),
        "fault_counts": {name: faults.count(name) for name in sorted(set(faults))},
    }
    if not faults:
        contract_status["verdict"] = "contract_met"
    elif len(faults) == len(suite_details):
        contract_status["verdict"] = "contract_invalid"
    else:
        contract_status["verdict"] = "contract_partial"
    passed = benchmark_pass(
        categories,
        suite_categories,
        all_valid,
    )
    return {
        "schema_version": "1.0",
        "scoring_version": SCORING_VERSION,
        # Identifies the exact frozen targets these scores are relative to.
        "calibration_id": calibration_id(),
        "envelope_sigma": ENVELOPE_SIGMA,
        "status": "ok",
        "contract_status": contract_status,
        "problem_id": problem_id,
        "score": reward,
        "reward": reward,
        "passed": passed,
        "category_pass": category_pass(categories),
        "pass_threshold": {
            **PASS_THRESHOLDS,
            "lower_tail_quality": TAIL_QUALITY_THRESHOLD,
            "catastrophic_floors": CATASTROPHIC_FLOORS,
            "all_suites_valid": True,
        },
        "category_weights": CATEGORY_WEIGHTS,
        "subscores": [
            {"name": f"category:{name}", "score": value} for name, value in categories.items()
        ],
        "additional_data": {
            "included_suites": len(specs),
            "category_scores": categories,
            "raw_category_scores": raw_categories,
            "category_weights": CATEGORY_WEIGHTS,
            "lower_tail_quality": tail_quality,
            "breadth_factor": breadth,
            "suite_details": suite_details,
        },
    }


def main() -> int:
    config = json.loads(HARNESS_CONFIG.read_text())
    reward_path = Path(config["reward_json"])
    detail_path = reward_path.with_name("reward_detail.json")
    try:
        detail = grade(config)
    except Exception as error:
        detail = {
            "schema_version": "1.0",
            "scoring_version": SCORING_VERSION,
            "status": "grader_error",
            "score": 0.0,
            "reward": 0.0,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps({"reward": detail["reward"]}, sort_keys=True) + "\n")
    detail_path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
    return 0 if detail["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
