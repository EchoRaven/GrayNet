"""Driver-independent diagnosis and agent interface."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Diagnosis:
    detected: bool
    component: str | None = None
    mechanism: str | None = None
    explanation: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlowDiagnosis:
    """Separate a service violation from a slowdown and its location."""

    detected: bool | None
    affected_flow: str | None
    component: str | None
    explanation: str
    evidence: tuple[str, ...]


def validate_submission(answer: Diagnosis) -> None:
    """Every driver must satisfy the same answer contract."""
    if not isinstance(answer, Diagnosis) or type(answer.detected) is not bool:
        raise ValueError("agent must submit a Diagnosis with boolean detection")
    for label in (answer.component, answer.mechanism):
        if answer.detected:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("a detected incident needs component and mechanism")
        elif label is not None:
            raise ValueError("a healthy answer must use null component and mechanism")
    _validate_evidence_fields(answer)


def validate_flow_submission(answer: FlowDiagnosis) -> None:
    """Allow unknown service state, affected flow, or location independently."""
    if not isinstance(answer, FlowDiagnosis) or (
        answer.detected is not None and type(answer.detected) is not bool
    ):
        raise ValueError("flow diagnosis needs a boolean or unknown service state")
    for label in (answer.affected_flow, answer.component):
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError("flow and component must be names or null")
    _validate_evidence_fields(answer)


def _validate_evidence_fields(answer: Diagnosis | FlowDiagnosis) -> None:
    if not isinstance(answer.explanation, str) or not answer.explanation.strip():
        raise ValueError("agent must explain its diagnosis")
    if (
        not isinstance(answer.evidence, tuple)
        or not answer.evidence
        or any(not isinstance(ref, str) or not ref for ref in answer.evidence)
    ):
        raise ValueError("agent must cite observation IDs")


class Agent(Protocol):
    """An agent receives the public incident report; tools are configured out of band."""

    def diagnose(self, incident: str) -> Diagnosis | FlowDiagnosis: ...
