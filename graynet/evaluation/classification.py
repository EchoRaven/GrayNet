"""Exact classification metrics; these do not judge explanation quality."""

from dataclasses import dataclass

from graynet.agents.base import Diagnosis
from graynet.scenarios.definitions import GroundTruth


@dataclass(frozen=True)
class ClassificationScore:
    detection_correct: bool
    localization_correct: bool | None
    mechanism_correct: bool | None


def _same_label(actual: str | None, expected: str | None) -> bool:
    return (
        isinstance(actual, str)
        and isinstance(expected, str)
        and actual.strip().casefold() == expected.casefold()
    )


def score_classification(
    case_valid: bool, truth: GroundTruth, answer: Diagnosis
) -> ClassificationScore:
    """An invalid or broken episode never enters an agent's denominator."""
    if not case_valid:
        raise ValueError("cannot score an agent on an invalid case run")
    if not isinstance(answer, Diagnosis):
        raise ValueError("agent answer must be a Diagnosis")
    if not isinstance(answer.detected, bool):
        raise ValueError("detected must be a boolean")
    if not truth.anomaly:
        return ClassificationScore(not answer.detected, None, None)
    return ClassificationScore(
        detection_correct=answer.detected,
        localization_correct=answer.detected
        and any(
            _same_label(answer.component, label)
            for label in (truth.component, *truth.component_aliases)
        ),
        mechanism_correct=answer.detected
        and any(
            _same_label(answer.mechanism, label)
            for label in (truth.mechanism, *truth.mechanism_aliases)
        ),
    )
