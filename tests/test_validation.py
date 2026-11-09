from dataclasses import replace
from pathlib import Path

from graynet.evaluation.validation import validate_sustained_case
from graynet.scenarios.definitions import load_case

FIXTURE = Path(__file__).parent / "fixtures" / "sustained_congestion"
SPEC = load_case(FIXTURE).validation
BASELINE = [100, 101, 99, 100, 100]
INCIDENT = [60, 61, 62, 59, 60]
RECOVERY = [99, 100, 100, 101, 100]


def test_admit_repeatable_degradation_and_recovery() -> None:
    report = validate_sustained_case(SPEC, BASELINE, INCIDENT, RECOVERY)
    assert report.valid
    assert report.relative_change == 0.40
    assert report.affected_fraction == 1.0


def test_pre_agent_admission_does_not_require_recovery_yet() -> None:
    report = validate_sustained_case(SPEC, BASELINE, INCIDENT)
    assert report.valid
    assert report.recovery_median is None


def test_reject_no_effect() -> None:
    report = validate_sustained_case(SPEC, BASELINE, [98, 100, 99, 100, 99])
    assert not report.valid
    assert "no_repeatable_degradation" in report.reasons


def test_reject_single_large_outlier() -> None:
    report = validate_sustained_case(SPEC, BASELINE, [50, 100, 100, 100, 100])
    assert not report.valid
    assert report.affected_fraction == 0.2


def test_reject_unstable_baseline() -> None:
    report = validate_sustained_case(SPEC, [50, 100, 150, 100, 100], INCIDENT)
    assert "unstable_baseline" in report.reasons


def test_reject_failed_recovery() -> None:
    report = validate_sustained_case(SPEC, BASELINE, INCIDENT, [60, 61, 59, 60, 60])
    assert "recovery_not_healthy" in report.reasons


def test_reject_insufficient_or_invalid_samples() -> None:
    assert "insufficient_incident_samples" in validate_sustained_case(SPEC, BASELINE, [60]).reasons
    assert (
        "invalid_incident_samples"
        in validate_sustained_case(SPEC, BASELINE, [60, 61, -1, 59, 60]).reasons
    )


def test_healthy_control_requires_stability() -> None:
    control = replace(SPEC, effect="stable", direction=None, threshold=0.1)
    assert validate_sustained_case(control, BASELINE, RECOVERY, BASELINE).valid
    assert "unexpected_degradation" in validate_sustained_case(control, BASELINE, INCIDENT).reasons
