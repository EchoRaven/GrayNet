"""Plan and run a paired, model-free comparison of throughput probe access."""

import random
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from graynet.agents.base import FlowDiagnosis
from graynet.agents.flow_baseline import BASELINE_CALLS, PROBE_CALLS, FlowProbeBaseline
from graynet.artifacts.records import RunArtifacts, utc_now, write_json
from graynet.artifacts.trajectory import save_trajectory
from graynet.environments.partial_path import PartialPathBackend
from graynet.evaluation.flow_diagnosis import score_flow_diagnosis
from graynet.evaluation.integrity import audit_observations
from graynet.evaluation.judge import identity_digest
from graynet.execution.experiments import FAULT_SEED_LIMIT, case_fingerprint, runtime_fingerprint
from graynet.execution.runner import run_case
from graynet.scenarios.definitions import load_case
from graynet.tools.network import NetworkTools

ARMS = ("fixed", "choice")
FIELDS = {
    "schema_version",
    "name",
    "cases",
    "fixed_flow",
    "repetitions",
    "order_seed",
    "fault_seed",
    "max_tool_calls",
    "timeout_seconds",
}


def plan_probe_experiment(path: Path) -> dict[str, Any]:
    """Freeze matched cases, tool access, budgets, and counterbalanced arm order."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise ValueError("invalid probe experiment fields")
    for key, minimum in (
        ("schema_version", 1),
        ("repetitions", 2),
        ("order_seed", 0),
        ("fault_seed", 0),
        ("max_tool_calls", BASELINE_CALLS),
        ("timeout_seconds", 1),
    ):
        if type(raw[key]) is not int or raw[key] < minimum:
            raise ValueError(f"invalid probe experiment {key}")
    if (
        raw["schema_version"] != 1
        or not isinstance(raw["name"], str)
        or not raw["name"].strip()
        or raw["repetitions"] % 2
        or raw["fault_seed"] + raw["repetitions"] > FAULT_SEED_LIMIT
    ):
        raise ValueError("expected an even number of repetitions and valid unsigned loss seeds")
    if not isinstance(raw["cases"], list) or not all(isinstance(x, str) for x in raw["cases"]):
        raise ValueError("cases must list scenario directories")
    cases = [load_case(path.resolve().parent / directory) for directory in raw["cases"]]
    if len(cases) != 7 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("the comparison needs seven distinct cases")
    if any(case.partial_path is None for case in cases):
        raise ValueError("every comparison case must use the partial path profile")
    backends = [PartialPathBackend(case) for case in cases]
    if len({backend.environment_sha256 for backend in backends}) != 1:
        raise ValueError("all arms and cases need the same public environment and workload")
    first = backends[0]
    flows = tuple(first.flows)
    fixed = raw["fixed_flow"]
    if fixed not in flows or raw["timeout_seconds"] < PROBE_CALLS * first.duration + 10:
        raise ValueError("invalid fixed flow or insufficient time to complete the probes")
    if sum(backend.selection.flow is None for backend in backends) != 1:
        raise ValueError("the plan needs one healthy control")
    bands = {
        flow: sorted(
            (
                backend.selection.min_slowdown,
                backend.selection.max_slowdown,
                backend.case.ground_truth.anomaly,
            )
            for backend in backends
            if backend.selection.flow == flow
        )
        for flow in flows
    }
    left, right = bands.values()
    if (
        len(left) != 3
        or left != right
        or [violated for _, _, violated in left] != [False, True, True]
        or any(
            high > next_low for (_, high, _), (next_low, _, _) in zip(left, left[1:], strict=False)
        )
    ):
        raise ValueError("both paths need matching, disjoint mild, near, and clear severity bands")

    rng = random.Random(raw["order_seed"])
    pairs = []
    for backend in backends:
        case = backend.case
        initial_order = rng.randrange(2)
        for index in range(raw["repetitions"]):
            pairs.append(
                {
                    "id": f"{case.case_id}-r{index + 1:03d}",
                    "case_id": case.case_id,
                    "case_directory": str(case.directory),
                    "repetition": index + 1,
                    "fault_seed": None if backend.is_control else raw["fault_seed"] + index,
                    "fixed_probe_intersects_fault": None
                    if backend.is_control
                    else backend.selection.flow == fixed,
                    "arm_order": list(ARMS if (index + initial_order) % 2 == 0 else ARMS[::-1]),
                }
            )
    rng.shuffle(pairs)
    plan = {
        **{key: raw[key] for key in FIELDS - {"cases"}},
        "flows": list(flows),
        "targets": {"fixed": [fixed], "choice": list(flows)},
        "runtime_sha256": runtime_fingerprint(),
        "case_sha256": {case.case_id: case_fingerprint(case.directory) for case in cases},
        "pairs": pairs,
    }
    return {**plan, "sha256": identity_digest(plan)}


def _run_arm(plan: dict, pair: dict, arm: str, directory: Path) -> dict:
    case = load_case(pair["case_directory"])
    if pair["fault_seed"] is not None:
        case = replace(case, fault={**case.fault, "seed": pair["fault_seed"]})
    requested = {
        "pair": pair["id"],
        "arm": arm,
        "fault_seed": pair["fault_seed"],
        "plan_sha256": plan["sha256"],
    }
    artifacts = RunArtifacts.create(
        case.case_id, "run-probe-experiment", output=directory, requested=requested
    )
    try:
        backend = PartialPathBackend(case)
        tools = NetworkTools(
            case.environment,
            lab_name=backend.LAB_NAME,
            workload_lock=backend.workload_lock,
            policy=backend.network.policy,
            probe=backend.probe_for_agent,
            reference=backend.healthy_reference,
            probe_targets=tuple(plan["targets"][arm]),
        )
        agent = FlowProbeBaseline(
            tools, max_calls=plan["max_tool_calls"], timeout_seconds=plan["timeout_seconds"]
        )
        metadata = {
            "driver": "flow-probe-baseline",
            "version": "1",
            "uses_model": False,
            "arm": arm,
            "max_tool_calls": plan["max_tool_calls"],
            "timeout_seconds": plan["timeout_seconds"],
            "tool_definitions": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    },
                }
                for tool in tools.definitions()
            ],
        }
        write_json(
            artifacts.agent / "input" / "context.json",
            {
                "incident": case.incident,
                "environment": case.environment,
                "tools": tools.definitions(),
            },
        )
        artifacts.stage("evaluation")
        result = run_case(case, backend, agent)
        score = None
        if result.status == "valid" and result.agent_status == "completed":
            assert isinstance(result.agent_answer, FlowDiagnosis)
            score = asdict(
                score_flow_diagnosis(case, backend.flows, result.agent_answer, agent.trace)
            )
            write_json(artifacts.agent / "diagnosis.json", asdict(result.agent_answer))
        metadata.update(
            status=result.agent_status, visibility_audit=audit_observations(agent.trace)
        )
        save_trajectory(artifacts.agent, trace=agent.trace, metadata=metadata)
        artifacts.finish(
            {
                "run": {
                    **asdict(result),
                    "evaluation": {
                        "mode": "flow_diagnosis",
                        "status": "completed" if score is not None else "not_run",
                        "version": "1",
                    },
                    "flow_evaluation": score,
                },
                "ground_truth": {
                    **asdict(case.ground_truth),
                    "affected_flow": backend.selection.flow,
                },
                "agent": metadata,
                "evidence": backend.evidence,
            },
            trace=agent.trace,
        )
    except (Exception, KeyboardInterrupt) as exc:
        artifacts.fail(exc)
        raise
    return {
        "directory": str(directory),
        "case_status": result.status,
        "agent_status": result.agent_status,
        "validation_reasons": list(result.validation.reasons) if result.validation else [],
        "tool_calls": len(agent.trace),
        "score": score,
    }


def _compare_pairs(pairs: list[dict], outcomes: dict) -> dict:
    groups: dict[str, dict] = {}
    for pair in pairs:
        arms = outcomes[pair["id"]]
        if set(arms) != set(ARMS) or any(row["score"] is None for row in arms.values()):
            continue
        group = {
            None: "control",
            True: "fixed_probe_sees_fault",
            False: "fixed_probe_misses_fault",
        }[pair["fixed_probe_intersects_fault"]]
        scores = [arms[arm]["score"] for arm in ARMS]
        for dimension in ("detection", "affected_flow", "location"):
            if all(score[dimension] == "not_applicable" for score in scores):
                continue
            qualified = tuple(
                score[dimension] == "correct" and score["evidence_support"][dimension] is True
                for score in scores
            )
            outcome = {
                (True, True): "both",
                (True, False): "fixed_only",
                (False, True): "choice_only",
                (False, False): "neither",
            }[qualified]
            counts = groups.setdefault(group, {}).setdefault(
                dimension, dict.fromkeys(("both", "fixed_only", "choice_only", "neither"), 0)
            )
            counts[outcome] += 1
    return groups


def run_probe_experiment(
    plan: dict,
    output: Path,
    *,
    case_ids: set[str] | None = None,
    repetition: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Run both arms of each selected pair on separate fresh deployments."""
    if identity_digest({key: value for key, value in plan.items() if key != "sha256"}) != plan.get(
        "sha256"
    ):
        raise ValueError("probe experiment plan hash does not match")
    if (
        case_ids
        and not case_ids <= set(plan["case_sha256"])
        or repetition is not None
        and not 1 <= repetition <= plan["repetitions"]
    ):
        raise ValueError("unknown case or repetition in probe experiment selection")
    pairs = [
        pair
        for pair in plan["pairs"]
        if (not case_ids or pair["case_id"] in case_ids)
        and (repetition is None or pair["repetition"] == repetition)
    ]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    outcomes = {pair["id"]: {} for pair in pairs}
    report = {
        "status": "running",
        "pairs": outcomes,
        "plan_sha256": plan["sha256"],
        "started_at": utc_now(),
    }
    write_json(output / "summary.json", report)
    try:
        for pair in pairs:
            for arm in pair["arm_order"]:
                if (
                    runtime_fingerprint() != plan["runtime_sha256"]
                    or case_fingerprint(Path(pair["case_directory"]))
                    != plan["case_sha256"][pair["case_id"]]
                ):
                    raise ValueError("frozen runtime or scenario changed during the experiment")
                progress(f"{pair['id']} {arm}")
                directory = output / "runs" / pair["id"] / arm
                outcome = _run_arm(plan, pair, arm, directory)
                outcomes[pair["id"]][arm] = outcome
                write_json(output / "summary.json", report)
                if outcome["case_status"] == "error":
                    raise RuntimeError(f"backend error; see {directory}")
        report["status"] = "finished"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
        )
        raise
    finally:
        report["complete_pairs"] = sum(
            set(arms) == set(ARMS) and all(row["score"] is not None for row in arms.values())
            for arms in outcomes.values()
        )
        report["supported_correct_comparisons"] = _compare_pairs(pairs, outcomes)
        report["finished_at"] = utc_now()
        write_json(output / "summary.json", report)
    return report
