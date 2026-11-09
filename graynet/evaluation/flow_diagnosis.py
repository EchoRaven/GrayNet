"""Score structured flow diagnoses and check the observations behind them."""

import json
import statistics
from dataclasses import dataclass
from typing import Literal

from graynet.agents.base import FlowDiagnosis
from graynet.environments.network import ServiceFlow
from graynet.evaluation.integrity import audit_evidence
from graynet.scenarios.definitions import CaseDefinition

MIN_PROBES = 2
AGREEMENT = 0.75
MIN_SLOWDOWN = 0.04
Outcome = Literal["correct", "incorrect", "uncertain", "not_applicable"]


@dataclass(frozen=True)
class FlowSignals:
    service_violation: bool | None = None
    slowdown: bool | None = None
    loss_location: str | None = None


@dataclass(frozen=True)
class FlowDiagnosisScore:
    detection: Outcome
    affected_flow: Outcome
    location: Outcome
    evidence_support: dict[str, bool | None]


def queue_command(component: str) -> dict:
    """Request sanitized counters for a named egress interface."""
    node, interface = component.split(":")
    return {"node": node, "argv": ["tc", "-s", "qdisc", "show", "dev", interface]}


def _queue_drops(entry: dict) -> int | None:
    output = entry.get("result", {}).get("stdout")
    if not isinstance(output, str):
        return None
    rows = json.loads(output).get("queues", [])
    return sum(row["dropped"] for row in rows) if rows else None


def _drops_during_probe(trace: list[dict], component: str, probe_indices: list[int]) -> bool:
    counters = []
    for index, entry in enumerate(trace):
        if entry["tool"] == "run_command" and entry["arguments"] == queue_command(component):
            drops = _queue_drops(entry)
            if drops is not None:
                counters.append((index, drops))
    return any(
        before < probe < after and end > start
        for before, start in counters
        for after, end in counters
        for probe in probe_indices
    )


def _consensus(votes: list[bool]) -> bool | None:
    if len(votes) < MIN_PROBES:
        return None
    positive = sum(votes) / len(votes)
    if positive >= AGREEMENT:
        return True
    if 1 - positive >= AGREEMENT:
        return False
    return None


def flow_signals(trace: list[dict]) -> dict[str, FlowSignals]:
    """Interpret only successful public observations in the supplied trace."""
    visible = [
        entry
        for entry in trace
        if entry["status"] == "completed" and entry["result"].get("exit_code", 0) == 0
    ]
    references = next(
        (row["result"]["flows"] for row in visible if row["tool"] == "healthy_reference"),
        {},
    )
    signals = {}
    for name, reference in references.items():
        indices = [
            index
            for index, row in enumerate(visible)
            if row["tool"] == "throughput_probe" and row["result"]["flow"] == name
        ]
        values = [visible[index]["result"]["throughput_mbps"] for index in indices]
        healthy = reference["throughput_mbps"]
        median = statistics.median(healthy)
        variation = statistics.pstdev(healthy) / statistics.mean(healthy)
        slowdown_limit = median * (1 - max(MIN_SLOWDOWN, 3 * variation))
        violation = _consensus([value < reference["service_min_mbps"] for value in values])
        slowdown = _consensus([value <= slowdown_limit for value in values])
        egress = reference["preferred_egress"]
        location = egress if slowdown and _drops_during_probe(visible, egress, indices) else None
        signals[name] = FlowSignals(violation, slowdown, location)
    return signals


def service_state(signals: dict[str, FlowSignals]) -> bool | None:
    """A repeatable violation on either flow suffices; clearing both needs both."""
    if any(signal.service_violation is True for signal in signals.values()):
        return True
    if len(signals) == 2 and all(signal.service_violation is False for signal in signals.values()):
        return False
    return None


def _finding(actual: str | None, expected: tuple[str, ...]) -> Outcome:
    if not expected:
        return "incorrect" if actual is not None else "not_applicable"
    if actual is None:
        return "uncertain"
    return "correct" if actual in expected else "incorrect"


def score_flow_diagnosis(
    case: CaseDefinition,
    flows: dict[str, ServiceFlow],
    answer: FlowDiagnosis,
    trace: list[dict],
) -> FlowDiagnosisScore:
    """Separate truth agreement from the support for each structured claim."""
    if case.partial_path is None or not audit_evidence(answer, trace)["passed"]:
        raise ValueError("flow scoring needs a partial path case and valid evidence references")
    cited = [entry for entry in trace if entry.get("observation_id") in answer.evidence]
    signals = flow_signals(cited)
    affected = case.partial_path.flow
    expected = (
        (":".join(flows[affected].sender), ":".join(flows[affected].receiver))
        if affected is not None
        else ()
    )
    supported_components = {
        component
        for name, signal in signals.items()
        if signal.loss_location is not None
        for component in (signal.loss_location, ":".join(flows[name].receiver))
    }
    detection = (
        "uncertain"
        if answer.detected is None
        else "correct"
        if answer.detected == case.ground_truth.anomaly
        else "incorrect"
    )
    return FlowDiagnosisScore(
        detection,
        _finding(answer.affected_flow, (affected,) if affected else ()),
        _finding(answer.component, expected),
        {
            "detection": None
            if answer.detected is None
            else answer.detected == service_state(signals),
            "affected_flow": None
            if answer.affected_flow is None
            else signals.get(answer.affected_flow, FlowSignals()).slowdown is True,
            "location": None
            if answer.component is None
            else answer.component in supported_components,
        },
    )
