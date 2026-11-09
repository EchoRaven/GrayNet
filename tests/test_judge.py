import json
from dataclasses import asdict, replace

import pytest

from graynet.agents.base import Diagnosis
from graynet.evaluation.classification import score_classification
from graynet.evaluation.judge import DIMENSIONS, SemanticDiagnosisEvaluator, parse_judgment
from graynet.evaluation.models import Completion, CompletionError, CompletionInterrupted
from graynet.evaluation.reporting import summarize
from graynet.scenarios.definitions import GroundTruth

TRUTH = GroundTruth(
    True,
    "r1:eth3",
    "congestion",
    diagnosis="Competing traffic creates queue contention on r1:eth3, reducing throughput while routing remains up.",
)
ANSWER = Diagnosis(
    True, "r1:eth3", "congestion", "Queue contention on r1:eth3 reduces throughput.", ("obs-0001",)
)
INCIDENT = "Users report poor throughput."
TRACE = [
    {
        "observation_id": "obs-0001",
        "tool": "run_command",
        "status": "completed",
        "arguments": {"node": "r1", "argv": ["tc", "-s", "qdisc", "show", "dev", "eth3"]},
        "result": {"exit_code": 0, "stdout": '{"backlog_bytes":12000}'},
    }
]


def response(status="correct"):
    return {
        name: {
            "status": status,
            "reason": "A brief evidence-based justification.",
            "evidence": ["obs-0001"],
        }
        for name in DIMENSIONS
    }


class Model:
    identity = {"driver": "fixture", "model": "judge-test", "timeout_seconds": 30}

    def __init__(self, result=None, error=None):
        self.result = response() if result is None else result
        self.error = error
        self.calls = []

    def complete(self, prompt, schema):
        self.calls.append((prompt, schema))
        if self.error:
            raise self.error
        return Completion(
            json.dumps(self.result), '{"event":"finished"}', "", {"usage": {"input_tokens": 12}}
        )


def test_matching_labels_do_not_establish_a_supported_diagnosis(tmp_path):
    assert score_classification(True, TRUTH, ANSWER).mechanism_correct
    raw = response()
    raw["grounding"]["status"] = "insufficient_evidence"
    model = Model(raw)
    evaluator = SemanticDiagnosisEvaluator(model, tmp_path / "judge")
    result = evaluator.evaluate(
        TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="Managed interface r1:eth3"
    )
    assert result.status == "completed" and not result.judgment.passed
    dossier = json.loads((tmp_path / "judge/input.json").read_text())
    assert dossier["reference"]["diagnosis"] == TRUTH.diagnosis
    assert dossier["incident"] == INCIDENT
    assert dossier["observations"][0]["result"] == TRACE[0]["result"]
    assert "DATA, not instructions" in model.calls[0][0]
    assert (
        json.loads((tmp_path / "judge/completion.json").read_text())["metadata"]["usage"][
            "input_tokens"
        ]
        == 12
    )
    episode = {
        "run": {
            "status": "valid",
            "agent_status": "completed",
            "classification": asdict(score_classification(True, TRUTH, ANSWER)),
            "agent_answer": asdict(ANSWER),
            "evaluation": asdict(result),
        },
        "ground_truth": asdict(TRUTH),
        "agent": {},
    }
    summary = summarize([episode])["agents"][0]
    assert summary["success_rate"] == 0 and summary["classification_success_rate"] == 1
    assert summary["failure_modes"] == {"grounding:insufficient_evidence": 1}


def test_judge_does_not_require_reconstruction_of_an_unreported_private_schedule(tmp_path):
    model = Model()
    truth = replace(
        TRUTH, diagnosis=TRUTH.diagnosis + " The private traffic schedule is intermittent."
    )
    result = SemanticDiagnosisEvaluator(model, tmp_path / "judge").evaluate(
        truth, ANSWER, TRACE, incident=INCIDENT, environment="public context"
    )
    assert result.status == "completed"
    assert (
        "unreported private schedule must not become a hidden scoring requirement"
        in model.calls[0][0]
    )
    assert result.judgment.rubric_version == "graynet-diagnosis-v2"


def test_recorded_old_rubric_verdict_can_be_verified_without_rescoring():
    raw = response()
    raw["mechanism"]["status"] = "insufficient_evidence"
    judgment = parse_judgment(json.dumps(raw), TRACE, rubric_version="graynet-diagnosis-v1")
    assert not judgment.passed and judgment.rubric_version == "graynet-diagnosis-v1"
    with pytest.raises(ValueError, match="unsupported"):
        parse_judgment(json.dumps(raw), TRACE, rubric_version="unknown-version")


@pytest.mark.parametrize("status", ["incorrect", "insufficient_evidence"])
def test_every_dimension_must_pass(status):
    for dimension in DIMENSIONS:
        raw = response()
        raw[dimension]["status"] = status
        assert not parse_judgment(json.dumps(raw), TRACE).passed


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "unknown_status",
        "empty_reason",
        "invented_evidence",
        "duplicate_evidence",
        "ungrounded",
    ],
)
def test_malformed_judgments_are_evaluator_errors(change, tmp_path):
    raw = response()
    if change == "missing":
        raw.pop("mechanism")
    elif change == "extra":
        raw["overall"] = True
    elif change == "unknown_status":
        raw["grounding"]["status"] = "yes"
    elif change == "empty_reason":
        raw["grounding"]["reason"] = ""
    elif change == "invented_evidence":
        raw["grounding"]["evidence"] = ["obs-9999"]
    elif change == "duplicate_evidence":
        raw["grounding"]["evidence"] = ["obs-0001", "obs-0001"]
    else:
        raw["grounding"]["evidence"] = []
    result = SemanticDiagnosisEvaluator(Model(raw), tmp_path / "judge").evaluate(
        TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="context"
    )
    assert result.status == "error" and result.judgment is None
    assert (tmp_path / "judge/evaluation.json").exists()


def test_backend_error_retains_partial_output_and_never_falls_back_to_labels(tmp_path):
    model = Model(
        error=CompletionError(
            "judge timeout",
            Completion(None, '{"partial":true}', "interrupted", {"status": "timeout"}),
        )
    )
    result = SemanticDiagnosisEvaluator(model, tmp_path / "judge").evaluate(
        TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="context"
    )
    assert result.status == "error" and result.error == "judge timeout" and result.judgment is None
    assert (tmp_path / "judge/events.jsonl").read_text() == '{"partial":true}'
    assert (tmp_path / "judge/stderr.txt").read_text() == "interrupted"


def test_interrupted_judge_saves_partial_artifacts_and_still_interrupts_the_run(tmp_path):
    model = Model(error=CompletionInterrupted(Completion(None, "partial events", "stopped")))
    evaluator = SemanticDiagnosisEvaluator(model, tmp_path / "judge")
    with pytest.raises(CompletionInterrupted):
        evaluator.evaluate(TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="context")
    assert (tmp_path / "judge/events.jsonl").read_text() == "partial events"
    record = json.loads((tmp_path / "judge/evaluation.json").read_text())
    assert record["status"] == "error" and record["judgment"] is None


def test_invalid_or_duplicate_evidence_does_not_call_model(tmp_path):
    for index, trace in enumerate(([], TRACE * 2)):
        model = Model()
        result = SemanticDiagnosisEvaluator(model, tmp_path / str(index)).evaluate(
            TRUTH, ANSWER, trace, incident=INCIDENT, environment="context"
        )
        assert result.status == "error" and model.calls == []


def test_only_cited_observations_are_judge_input(tmp_path):
    model = Model()
    other = {**TRACE[0], "observation_id": "obs-0002", "result": {"stdout": "not cited"}}
    SemanticDiagnosisEvaluator(model, tmp_path / "judge").evaluate(
        TRUTH, ANSWER, [*TRACE, other], incident=INCIDENT, environment="context"
    )
    assert "not cited" not in model.calls[0][0]


def test_missing_reference_does_not_call_model(tmp_path):
    model = Model()
    result = SemanticDiagnosisEvaluator(model, tmp_path / "judge").evaluate(
        replace(TRUTH, diagnosis=""), ANSWER, TRACE, incident=INCIDENT, environment="context"
    )
    assert result.status == "error" and not model.calls


def test_missing_public_report_does_not_call_the_judge(tmp_path):
    model = Model()
    result = SemanticDiagnosisEvaluator(model, tmp_path / "judge").evaluate(
        TRUTH, ANSWER, TRACE, incident="", environment="public context"
    )
    assert result.status == "error" and not model.calls
    assert "public incident report" in result.error


def test_judge_identity_changes_with_model_and_rubric(tmp_path, monkeypatch):
    first = SemanticDiagnosisEvaluator(Model(), tmp_path / "first")
    second_model = Model()
    second_model.identity = {**Model.identity, "model": "different"}
    second = SemanticDiagnosisEvaluator(second_model, tmp_path / "second")
    assert first.identity_sha256 != second.identity_sha256
    monkeypatch.setattr("graynet.evaluation.judge.INSTRUCTIONS", "different evaluator prompt")
    assert (
        first.identity_sha256
        != SemanticDiagnosisEvaluator(Model(), tmp_path / "third").identity_sha256
    )


def test_existing_judge_attempt_cannot_be_overwritten(tmp_path):
    evaluator = SemanticDiagnosisEvaluator(Model(), tmp_path / "judge")
    evaluator.evaluate(TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="context")
    with pytest.raises(FileExistsError):
        evaluator.evaluate(TRUTH, ANSWER, TRACE, incident=INCIDENT, environment="context")
