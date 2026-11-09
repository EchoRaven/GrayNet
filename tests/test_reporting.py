import pytest

from graynet.evaluation.reporting import summarize


def record(status="valid", agent_status="completed", correct=True, anomaly=True):
    return {
        "run": {
            "status": status,
            "agent_status": agent_status,
            "score": {
                "detection_correct": correct,
                "localization_correct": correct,
                "mechanism_correct": correct,
            }
            if agent_status == "completed"
            else None,
            "agent_answer": {"detected": anomaly if correct else not anomaly},
        },
        "agent": {"driver": "codex", "model": "test", "tool_calls": 2},
        "ground_truth": {"anomaly": anomaly},
    }


def test_valid_agent_errors_stay_in_denominator_broken_cases_do_not():
    result = summarize(
        [
            record(),
            record(agent_status="error"),
            record(status="invalid"),
            record(status="error"),
            record(agent_status="not_run"),
        ]
    )
    agent = result["agents"][0]
    assert agent["valid_episodes"] == 2
    assert agent["success_rate"] == 0.5
    assert agent["agent_errors"] == 1
    assert result["excluded"] == {"invalid": 1, "error": 1, "no_agent": 1}


def test_healthy_controls_count_false_alarms_and_not_localization():
    agent = summarize([record(anomaly=False), record(anomaly=False, correct=False)])["agents"][0]
    assert agent["false_alarm_rate"] == 0.5
    assert agent["localization_accuracy"] is None
    assert agent["estimated_cost_usd"] is None


def test_healthy_timeout_is_not_mistaken_for_an_observed_negative():
    agent = summarize(
        [record(anomaly=False, correct=False), record(anomaly=False, agent_status="error")]
    )["agents"][0]
    assert agent["valid_episodes"] == 2
    assert agent["healthy_agent_errors"] == 1
    assert agent["false_alarm_rate"] == 1
    assert agent["success_rate"] == 0


def test_partial_usage_is_not_reported_as_a_complete_total():
    known = record()
    known["agent"]["usage"] = {"input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 2}
    agent = summarize([known, record(agent_status="error")])["agents"][0]
    assert agent["usage"]["input_tokens"] is None
    assert agent["observed_usage"]["input_tokens"] == 10
    assert agent["episodes_with_missing_usage"]["input_tokens"] == 1


def test_different_budgets_and_answer_contracts_are_not_pooled():
    first, second, third = record(), record(), record()
    first["agent"].update(timeout_seconds=180, answer_schema_sha256="contract-a")
    second["agent"].update(timeout_seconds=5, answer_schema_sha256="contract-a")
    third["agent"].update(timeout_seconds=180, answer_schema_sha256="contract-b")
    assert len(summarize([first, second, third])["agents"]) == 3


def test_historical_missing_reference_is_not_backfilled_or_scored():
    incomplete = record()
    incomplete["ground_truth"] = None
    result = summarize([record(), incomplete])
    assert result["excluded"]["missing_reference"] == 1
    assert result["agents"][0]["valid_episodes"] == 1


def test_environment_changes_are_not_pooled_and_small_samples_have_wide_intervals():
    first, second = record(), record()
    first["agent"]["environment_sha256"] = "first"
    second["agent"]["environment_sha256"] = "second"
    result = summarize([first, second])
    assert len(result["agents"]) == 2
    lo, hi = result["agents"][0]["success_rate_ci95"]
    assert lo < 0.3 and hi > 0.99


def test_runtime_tool_and_image_revisions_are_not_pooled_with_legacy_results():
    records = [record() for _ in range(5)]
    for episode, field in zip(
        records[1:],
        ("runtime_sha256", "tool_definitions_sha256", "image_id", "environment_runtime_sha256"),
        strict=True,
    ):
        episode["agent"][field] = "changed"
    assert len(summarize(records)["agents"]) == 5


def test_per_case_metrics_and_failure_modes_include_timeouts():
    first, timeout = record(), record(agent_status="error")
    first["run"]["case_id"] = "loss"
    timeout["run"]["case_id"] = "congestion"
    timeout["agent"]["status"] = "timeout"
    result = summarize([first, timeout])["agents"][0]
    assert len(result["per_case"]) == 2 and result["failure_modes"] == {"timeout": 1}
    assert result["measurements"]["duration_seconds"]["missing_episodes"] == 2


def test_semantic_evaluator_errors_are_not_agent_failures_or_silent_successes():
    good, judge_error, agent_error = record(), record(), record(agent_status="error")
    for episode in (good, judge_error, agent_error):
        episode["run"]["evaluation"] = {
            "mode": "semantic",
            "identity_sha256": "judge-one",
            "status": "completed",
            "judgment": {"passed": True, "dimensions": {}},
        }
    judge_error["run"]["evaluation"].update(status="error", judgment=None)
    agent_error["run"]["evaluation"].update(status="not_run", judgment=None)
    summary = summarize([good, judge_error, agent_error])["agents"][0]
    assert summary["valid_episodes"] == 3 and summary["scored_episodes"] == 2
    assert summary["evaluation_errors"] == 1 and summary["agent_errors"] == 1
    assert summary["success_rate"] == 0.5


def test_semantic_cohorts_and_legacy_label_cohorts_are_never_pooled():
    one, two = record(), record()
    one["run"]["evaluation"] = {
        "mode": "semantic",
        "status": "completed",
        "identity_sha256": "one",
        "judgment": {"passed": True, "dimensions": {}},
    }
    two["run"]["evaluation"] = {
        "mode": "semantic",
        "status": "completed",
        "identity_sha256": "two",
        "judgment": {"passed": True, "dimensions": {}},
    }
    assert len(summarize([record(), one, two])["agents"]) == 3


@pytest.mark.parametrize(
    "status,evidence_passed,reason",
    [
        ("timeout", False, "timeout"),
        ("error", False, "invalid_evidence"),
        ("error", True, "agent_error"),
    ],
)
def test_agent_failure_categories_keep_their_precedence(status, evidence_passed, reason):
    episode = record(agent_status="error")
    episode["agent"].update(status=status, evidence_audit={"passed": evidence_passed})
    assert summarize([episode])["agents"][0]["failure_modes"] == {reason: 1}


@pytest.mark.parametrize("score_field", ["score", "classification"])
def test_classification_metrics_accept_legacy_and_current_fields(score_field):
    episode = record()
    episode["run"][score_field] = episode["run"].pop("score")
    summary = summarize([episode])["agents"][0]
    assert summary["classification_matches"] == 1
    assert summary["classification_success_rate"] == 1
    assert summary["tool_calls"] == summary["measurements"]["tool_calls"]["observed_total"] == 2
