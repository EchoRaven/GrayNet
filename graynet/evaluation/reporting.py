"""Aggregate only admitted cases; valid agent failures remain in the denominator."""

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

NORMAL_95_PERCENT_Z = 1.959963984540054


def wilson_interval(successes: int, total: int) -> list[float] | None:
    """95% binomial Wilson interval; repeated live trials are not deterministic."""
    if not total:
        return None
    z = NORMAL_95_PERCENT_Z
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _classification_score(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("classification", run.get("score")) or {}


def _classification_correct(record: dict[str, Any]) -> bool:
    if record["run"]["agent_status"] != "completed":
        return False
    score = _classification_score(record["run"])
    return score.get("detection_correct") is True and (
        not record["ground_truth"]["anomaly"]
        or score.get("localization_correct") is True
        and score.get("mechanism_correct") is True
    )


def _semantic(record: dict[str, Any]) -> bool:
    return (record["run"].get("evaluation") or {}).get("mode") == "semantic"


def _eligible(record: dict[str, Any]) -> bool:
    if not _semantic(record) or record["run"]["agent_status"] != "completed":
        return True
    return record["run"]["evaluation"]["status"] == "completed"


def _correct(record: dict[str, Any]) -> bool:
    if not _semantic(record):
        return _classification_correct(record)
    evaluation = record["run"]["evaluation"]
    return (
        record["run"]["agent_status"] == "completed"
        and evaluation["status"] == "completed"
        and (evaluation.get("judgment") or {}).get("passed") is True
    )


def _success_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [episode for episode in episodes if _eligible(episode)]
    passed = sum(_correct(episode) for episode in eligible)
    return {
        "scored_episodes": len(eligible),
        "correct_diagnoses": passed,
        "success_rate": passed / len(eligible) if eligible else None,
        "success_rate_ci95": wilson_interval(passed, len(eligible)),
        "evaluation_errors": sum(
            (episode["run"].get("evaluation") or {}).get("status") == "error"
            for episode in episodes
        ),
        "unjudged_completed_episodes": sum(
            episode["run"]["agent_status"] == "completed"
            and _semantic(episode)
            and (episode["run"].get("evaluation") or {}).get("status") == "not_run"
            for episode in episodes
        ),
    }


def _measurements(episodes: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [episode.get("agent", {}).get(field) for episode in episodes]
    known = [
        value
        for value in values
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
    ]
    return {
        "observed_total": sum(known),
        "observed_mean": statistics.mean(known) if known else None,
        "observed_median": statistics.median(known) if known else None,
        "known_episodes": len(known),
        "missing_episodes": len(episodes) - len(known),
    }


def _agent_failure_reason(agent: dict[str, Any]) -> str:
    if agent.get("status") == "timeout":
        return "timeout"
    if agent.get("evidence_audit", {}).get("passed") is False:
        return "invalid_evidence"
    return "agent_error"


def _failures(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        run, agent = episode["run"], episode.get("agent", {})
        if run["agent_status"] != "completed":
            counts[_agent_failure_reason(agent)] += 1
            continue
        if _semantic(episode):
            evaluation = run["evaluation"]
            if evaluation["status"] != "completed":
                counts["evaluation_error" if evaluation["status"] == "error" else "unjudged"] += 1
                continue
            for dimension, judgment in evaluation["judgment"]["dimensions"].items():
                if judgment["status"] != "correct":
                    counts[f"{dimension}:{judgment['status']}"] += 1
            continue
        score = _classification_score(run)
        if score.get("detection_correct") is not True:
            counts[
                "missed_degradation" if episode["ground_truth"]["anomaly"] else "false_alarm"
            ] += 1
        elif episode["ground_truth"]["anomaly"]:
            if score.get("localization_correct") is not True:
                counts["wrong_component"] += 1
            if score.get("mechanism_correct") is not True:
                counts["wrong_mechanism"] += 1
    return dict(counts)


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate valid episodes without mixing incompatible evaluation cohorts."""
    groups = defaultdict(list)
    excluded = {"invalid": 0, "error": 0, "no_agent": 0}
    for record in records:
        run = record["run"]
        if "flow_evaluation" in run:
            excluded["paired_probe"] = excluded.get("paired_probe", 0) + 1
            continue
        if run["status"] != "valid":
            excluded[run["status"]] += 1
            continue
        if run["agent_status"] == "not_run":
            excluded["no_agent"] += 1
            continue
        reference = record.get("ground_truth")
        if not isinstance(reference, dict) or type(reference.get("anomaly")) is not bool:
            excluded["missing_reference"] = excluded.get("missing_reference", 0) + 1
            continue
        agent = record.get("agent", {})
        key = (
            agent.get("driver", "unknown"),
            agent.get("model"),
            agent.get("reasoning_effort"),
            agent.get("timeout_seconds"),
            agent.get("max_tool_calls"),
            agent.get("answer_schema_sha256"),
            agent.get("environment_sha256"),
            agent.get("runtime_sha256"),
            agent.get("tool_definitions_sha256"),
            agent.get("image_id"),
            agent.get("environment_runtime_sha256"),
            (run.get("evaluation") or {}).get("identity_sha256", "label-metrics-legacy"),
        )
        groups[key].append(record)
    summaries = []
    for (
        driver,
        model,
        effort,
        timeout,
        tool_budget,
        schema_hash,
        environment_hash,
        runtime_hash,
        tools_hash,
        image_id,
        live_environment_hash,
        evaluation_hash,
    ), episodes in groups.items():
        total = len(episodes)
        completed = 0
        healthy = completed_healthy = false_alarms = 0
        detection = localization = mechanism = anomalous = 0
        tokens = {key: 0 for key in ("input_tokens", "cached_input_tokens", "output_tokens")}
        tokens_available = False
        missing_usage = {key: 0 for key in tokens}
        for episode in episodes:
            run, agent = episode["run"], episode.get("agent", {})
            score = _classification_score(run)
            completed += run["agent_status"] == "completed"
            detection += score.get("detection_correct") is True
            anomaly = episode["ground_truth"]["anomaly"]
            if anomaly:
                anomalous += 1
                localization += score.get("localization_correct") is True
                mechanism += score.get("mechanism_correct") is True
            else:
                healthy += 1
                completed_healthy += run["agent_status"] == "completed"
                answer = run.get("agent_answer") or {}
                false_alarms += answer.get("detected") is True
            for key in tokens:
                value = (agent.get("usage") or {}).get(key)
                if type(value) is int and value >= 0:
                    tokens_available = True
                    tokens[key] += value
                else:
                    missing_usage[key] += 1
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for episode in episodes:
            by_case[episode["run"].get("case_id", "unknown")].append(episode)
        classification_matches = sum(_classification_correct(episode) for episode in episodes)
        measurements = {
            field: _measurements(episodes, field) for field in ("tool_calls", "duration_seconds")
        }
        summaries.append(
            {
                "driver": driver,
                "model": model,
                "reasoning_effort": effort,
                "timeout_seconds": timeout,
                "max_tool_calls": tool_budget,
                "answer_schema_sha256": schema_hash,
                "environment_sha256": environment_hash,
                "runtime_sha256": runtime_hash,
                "tool_definitions_sha256": tools_hash,
                "image_id": image_id,
                "environment_runtime_sha256": live_environment_hash,
                "evaluation_identity_sha256": evaluation_hash,
                "evaluation_mode": "semantic" if _semantic(episodes[0]) else "classification",
                "valid_episodes": total,
                "completed": completed,
                "agent_errors": total - completed,
                **_success_metrics(episodes),
                "classification_matches": classification_matches,
                "classification_success_rate": classification_matches / total,
                "detection_accuracy": detection / total,
                "localization_accuracy": localization / anomalous if anomalous else None,
                "mechanism_accuracy": mechanism / anomalous if anomalous else None,
                "healthy_episodes": healthy,
                "false_alarms": false_alarms,
                "completed_healthy_episodes": completed_healthy,
                "healthy_agent_errors": healthy - completed_healthy,
                "false_alarm_rate": false_alarms / completed_healthy if completed_healthy else None,
                "false_alarm_rate_ci95": wilson_interval(false_alarms, completed_healthy),
                "tool_calls": measurements["tool_calls"]["observed_total"],
                "duration_seconds": measurements["duration_seconds"]["observed_total"],
                "measurements": measurements,
                "failure_modes": _failures(episodes),
                "per_case": [
                    {
                        "case_id": case_id,
                        "valid_episodes": len(case_episodes),
                        **_success_metrics(case_episodes),
                        "failure_modes": _failures(case_episodes),
                    }
                    for case_id, case_episodes in sorted(by_case.items())
                ],
                "usage": {
                    key: value if not missing_usage[key] else None for key, value in tokens.items()
                }
                if tokens_available
                else None,
                "observed_usage": tokens if tokens_available else None,
                "episodes_with_missing_usage": missing_usage,
                "estimated_cost_usd": None,
            }
        )
    return {
        "schema_version": 2,
        "excluded": excluded,
        "agents": summaries,
        "interpretation": "Classification cohorts report label agreement, not semantic diagnosis quality. Semantic cohorts exclude evaluator failures from scored_episodes and report them separately. Valid agent failures remain unsuccessful episodes. Intervals assume independent trials within the selected cohort; small integration pilots do not estimate general benchmark performance.",
    }
