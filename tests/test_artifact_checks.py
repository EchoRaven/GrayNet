import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from graynet.agents.base import Diagnosis
from graynet.artifacts.records import write_json
from graynet.evaluation.judge import DIMENSIONS, SemanticDiagnosisEvaluator
from graynet.evaluation.models import Completion
from graynet.evaluation.reporting import summarize
from graynet.scenarios.definitions import GroundTruth
from scripts.check_experiment_artifacts import (
    verify_evaluation,
    verify_planned_evaluation,
    verify_summary,
)


def test_legacy_summary_metrics_remain_verifiable_without_rescoring():
    record = {
        "run": {
            "status": "valid",
            "agent_status": "completed",
            "score": {"detection_correct": True},
        },
        "ground_truth": {"anomaly": False},
        "agent": {},
    }
    computed = summarize([record])
    legacy = {
        "agents": [{"success_rate": 1, "driver": "unknown"}],
        "excluded": computed["excluded"],
        "interpretation": "historical wording",
    }
    verify_summary(legacy, computed, [record])
    corrupted = deepcopy(legacy)
    corrupted["agents"][0]["success_rate"] = 0
    with pytest.raises(ValueError, match="historical"):
        verify_summary(corrupted, computed, [record])
    with pytest.raises(ValueError, match="legacy"):
        verify_summary(legacy, computed, [{"run": {"evaluation": {"mode": "semantic"}}}])


def test_versioned_summary_must_match_every_computed_metric():
    computed = summarize([])
    verify_summary(computed, computed, [])
    changed = {**computed, "excluded": {}}
    with pytest.raises(ValueError):
        verify_summary(changed, computed, [])


@pytest.fixture
def semantic_record(tmp_path):
    truth = GroundTruth(True, "r1:eth3", "congestion", diagnosis="Traffic contends on r1:eth3.")
    answer = Diagnosis(True, "r1:eth3", "congestion", "Traffic contends on r1:eth3.", ("obs-0001",))
    trace = [
        {
            "observation_id": "obs-0001",
            "tool": "run_command",
            "status": "completed",
            "arguments": {"node": "r1"},
            "result": {"stdout": "sample", "exit_code": 0},
        }
    ]

    class Model:
        identity = {"model": "fixture", "reasoning_effort": "low", "timeout_seconds": 30}

        def complete(self, prompt, schema):
            raw = {
                name: {"status": "correct", "reason": "Fixture.", "evidence": ["obs-0001"]}
                for name in DIMENSIONS
            }
            return Completion(json.dumps(raw), "events", "stderr")

    result = SemanticDiagnosisEvaluator(Model(), tmp_path / "evaluator/judge").evaluate(
        truth, answer, trace, incident="Fixture incident report.", environment="public context"
    )
    record = json.loads(
        json.dumps(
            {
                "run": {"agent_answer": asdict(answer), "evaluation": asdict(result)},
                "ground_truth": asdict(truth),
            }
        )
    )
    (tmp_path / "agent/input").mkdir(parents=True)
    (tmp_path / "agent/input/environment.md").write_text("public context")
    (tmp_path / "agent/input/incident.md").write_text("Fixture incident report.")
    write_json(tmp_path / "agent/tools.json", trace)
    return tmp_path, record


def test_semantic_verdict_is_checked_against_raw_completion_and_actual_trace(semantic_record):
    directory, record = semantic_record
    assert verify_evaluation(directory, record)


@pytest.mark.parametrize("change", ["verdict", "dossier", "incident", "events", "identity"])
def test_semantic_artifact_corruption_is_rejected(semantic_record, change):
    directory, record = semantic_record
    if change == "verdict":
        record["run"]["evaluation"]["judgment"]["passed"] = False
        write_json(directory / "evaluator/judge/evaluation.json", record["run"]["evaluation"])
    elif change == "dossier":
        path = directory / "evaluator/judge/input.json"
        dossier = json.loads(path.read_text())
        dossier["observations"][0]["result"]["stdout"] = "fabricated observation"
        write_json(path, dossier)
    elif change == "events":
        (directory / "evaluator/judge/events.jsonl").write_text("different raw response")
    elif change == "incident":
        (directory / "agent/input/incident.md").write_text("A different reported symptom.")
    else:
        record["run"]["evaluation"]["identity"]["model"]["model"] = "different judge"
    with pytest.raises(ValueError):
        verify_evaluation(directory, record)


@pytest.mark.parametrize("changed", [None, "mode", "model", "reasoning_effort", "timeout_seconds"])
def test_actual_evaluator_must_match_the_frozen_plan(semantic_record, changed):
    _, record = semantic_record
    settings = {
        "mode": "semantic",
        "model": "fixture",
        "reasoning_effort": "low",
        "timeout_seconds": 30,
    }
    if changed == "mode":
        settings = {"mode": "classification"}
    elif changed:
        settings[changed] = 60 if changed == "timeout_seconds" else "different"
    if changed:
        with pytest.raises(ValueError, match="frozen experiment"):
            verify_planned_evaluation({"evaluation": settings}, record)
    else:
        verify_planned_evaluation({"evaluation": settings}, record)
