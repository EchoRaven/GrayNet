from dataclasses import replace
from pathlib import Path

import pytest

from graynet.agents.base import Diagnosis
from graynet.evaluation.classification import score_classification
from graynet.execution.runner import run_case, validate_case
from graynet.scenarios.definitions import GroundTruth, load_case

CASE = load_case(Path(__file__).parent / "fixtures" / "sustained_congestion")


class FakeBackend:
    def __init__(self, *, incident=None, recovery=None, fail_at=None):
        self.events = []
        self.incident = incident if incident is not None else [60, 61, 59, 60, 60]
        self.recovery = recovery if recovery is not None else [100, 101, 99, 100, 100]
        self.fail_at = fail_at

    def _event(self, name):
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(name)

    def deploy(self, case):
        self._event("deploy")

    def measure(self, case, phase):
        self._event(f"measure:{phase}")
        return {
            "baseline": [100, 101, 99, 100, 100],
            "incident": self.incident,
            "recovery": self.recovery,
        }[phase]

    def inject(self, case):
        self._event("inject")

    def recover(self, case):
        self._event("recover")

    def teardown(self, case):
        self._event("teardown")


class FakeAgent:
    def __init__(self, *, fail=False):
        self.prompts = []
        self.fail = fail
        self.trace = [
            {
                "observation_id": "obs-0001",
                "tool": "run_command",
                "status": "completed",
                "result": {"exit_code": 0},
            }
        ]

    def diagnose(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("agent unavailable")
        return Diagnosis(
            True, "r1-r2", "congestion", "The primary path is congested.", ("obs-0001",)
        )


def test_valid_case_scores_agent_and_cleans_up() -> None:
    backend = FakeBackend()
    agent = FakeAgent()
    result = run_case(CASE, backend, agent)
    assert result.status == "valid"
    assert result.agent_status == "completed"
    assert result.classification is not None
    assert result.classification.detection_correct
    assert result.classification.localization_correct
    assert result.classification.mechanism_correct
    assert agent.prompts == [CASE.incident]
    assert backend.events == [
        "deploy",
        "measure:baseline",
        "inject",
        "measure:incident",
        "recover",
        "measure:recovery",
        "teardown",
    ]


def test_invalid_incident_skips_agent() -> None:
    backend = FakeBackend(incident=[99, 100, 100, 100, 99])
    agent = FakeAgent()
    result = run_case(CASE, backend, agent)
    assert result.status == "invalid"
    assert result.agent_status == "not_run"
    assert result.classification is None
    assert agent.prompts == []
    assert backend.events[-2:] == ["measure:recovery", "teardown"]


def test_failed_recovery_excludes_agent_score() -> None:
    result = run_case(CASE, FakeBackend(recovery=[60, 60, 61, 59, 60]), FakeAgent())
    assert result.status == "invalid"
    assert result.agent_status == "completed"
    assert result.classification is None
    assert "recovery_not_healthy" in result.validation.reasons


def test_agent_failure_does_not_invalidate_case() -> None:
    result = run_case(CASE, FakeBackend(), FakeAgent(fail=True))
    assert result.status == "valid"
    assert result.agent_status == "error"
    assert result.classification is None
    assert "agent unavailable" in result.agent_error


def test_exposed_injector_artifact_excludes_episode_not_agent_failure():
    agent = FakeAgent()
    agent.trace[0]["result"]["stdout"] = "qdisc netem"
    backend = FakeBackend()
    result = run_case(CASE, backend, agent)
    assert result.status == "invalid" and result.agent_status == "completed"
    assert result.classification is None and "private_artifact_exposed" in result.validation.reasons
    assert backend.events[-2:] == ["measure:recovery", "teardown"]


@pytest.mark.parametrize("fail_at", ["deploy", "inject", "measure:incident", "recover"])
def test_backend_failure_attempts_cleanup(fail_at) -> None:
    backend = FakeBackend(fail_at=fail_at)
    result = run_case(CASE, backend)
    assert result.status == "error"
    assert "teardown" in backend.events
    if fail_at in ("inject", "measure:incident", "recover"):
        assert "recover" in backend.events


def test_teardown_failure_marks_case_error() -> None:
    result = run_case(CASE, FakeBackend(fail_at="teardown"), FakeAgent())
    assert result.status == "error"
    assert result.classification is None
    assert result.cleanup_errors[0].startswith("teardown:")


def test_validation_only_runs_without_an_agent() -> None:
    result = validate_case(CASE, FakeBackend())
    assert result.status == "valid"
    assert result.agent_status == "not_run"
    assert result.classification is None


def test_invalid_case_cannot_be_scored_directly() -> None:
    with pytest.raises(ValueError, match="invalid case"):
        score_classification(False, CASE.ground_truth, Diagnosis(True, "r1-r2", "congestion"))


def test_healthy_control_scores_false_alarms() -> None:
    truth = GroundTruth(False, None, None)
    assert score_classification(True, truth, Diagnosis(False)).detection_correct
    assert not score_classification(True, truth, Diagnosis(True)).detection_correct


def test_fault_expiry_during_agent_run_invalidates_the_episode() -> None:
    class MonitoredBackend(FakeBackend):
        def start_agent_window(self, case):
            self._event("start-window")

        def finish_agent_window(self, case, baseline):
            self._event("finish-window")
            raise RuntimeError("fault expired during diagnosis")

    backend = MonitoredBackend()
    result = run_case(CASE, backend, FakeAgent())
    assert result.status == "error"
    assert result.agent_status == "completed"
    assert result.classification is None
    assert backend.events[-2:] == ["recover", "teardown"]


def test_measured_agent_window_failure_is_invalid_and_still_measures_recovery():
    from dataclasses import replace

    from graynet.evaluation.validation import validate_sustained_case

    class Backend(FakeBackend):
        def finish_agent_window(self, case, baseline):
            report = validate_sustained_case(case.validation, baseline, self.incident)
            return replace(report, valid=False, reasons=("no_repeatable_degradation",))

    backend = Backend()
    result = run_case(CASE, backend, FakeAgent())
    assert (
        result.status == "invalid"
        and result.agent_status == "completed"
        and result.classification is None
    )
    assert (
        result.error is None
        and "agent_window:no_repeatable_degradation" in result.validation.reasons
    )
    assert result.recovery and backend.events[-2:] == ["measure:recovery", "teardown"]


def test_equivalent_interface_labels_do_not_require_the_private_case_label():
    truth = GroundTruth(
        True, "r1-r2-primary", "congestion", ("r1:eth3", "r2:eth1"), ("queue contention",)
    )
    score = score_classification(True, truth, Diagnosis(True, "r1:eth3", "queue contention"))
    assert score.localization_correct and score.mechanism_correct


def test_semantic_judge_runs_after_teardown_and_does_not_change_case_validity(tmp_path):
    import json

    from graynet.evaluation.judge import SemanticDiagnosisEvaluator
    from graynet.evaluation.models import Completion

    backend = FakeBackend()
    seen = []
    case = replace(
        CASE,
        ground_truth=replace(
            CASE.ground_truth, diagnosis="Competing traffic congests the primary path."
        ),
    )

    class Model:
        identity = {"model": "test"}

        def complete(self, prompt, schema):
            seen.append((backend.events[-1], json.loads(prompt.split("Dossier:\n", 1)[1])))
            return Completion("malformed judge output")

    result = run_case(
        case,
        backend,
        FakeAgent(),
        evaluator=SemanticDiagnosisEvaluator(Model(), tmp_path / "judge"),
    )
    assert result.status == "valid" and result.agent_status == "completed"
    assert result.classification.mechanism_correct and result.evaluation.status == "error"
    assert result.evaluation.judgment is None
    assert len(seen) == 1 and seen[0][0] == "teardown"
    assert seen[0][1]["incident"] == case.incident


@pytest.mark.parametrize(
    "answer",
    [
        None,
        Diagnosis(True, "r1-r2", "congestion"),
        Diagnosis(True, "r1-r2", "congestion", "Explained.", ("obs-9999",)),
    ],
)
def test_all_drivers_share_the_submission_and_evidence_contract(answer):
    agent = FakeAgent()
    agent.diagnose = lambda incident: answer
    result = run_case(CASE, FakeBackend(), agent)
    assert (
        result.status == "valid"
        and result.agent_status == "error"
        and result.classification is None
    )
