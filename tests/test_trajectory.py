import json

import pytest
from pydantic import ValidationError

from graynet.artifacts.atif import Trajectory
from graynet.artifacts.atif.validator import TrajectoryValidator
from graynet.artifacts.trajectory import normalize_trajectory, render_trajectory


def test_trajectory_retains_provider_order_tool_errors_usage_and_source_locations():
    call = {
        "id": "item_1",
        "type": "mcp_tool_call",
        "tool": "run_command",
        "arguments": {"node": "r1", "argv": ["ps"]},
    }
    events = [
        {"type": "thread.started", "thread_id": "test"},
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "Check counters"},
        },
        {"type": "item.started", "item": {**call, "status": "in_progress"}},
        {
            "type": "item.completed",
            "item": {**call, "status": "completed", "error": "denied", "result": None},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 50}},
    ]
    trace = [
        {
            "tool": "run_command",
            "arguments": call["arguments"],
            "status": "error",
            "error": "denied",
            "timestamp": "2026-09-30T01:02:03+00:00",
            "duration_seconds": 0.1,
        }
    ]
    trajectory = normalize_trajectory(
        "\n".join(json.dumps(event) for event in events),
        prompt="Investigate",
        trace=trace,
        metadata={"driver": "codex", "model": "test", "status": "completed"},
    )
    steps = trajectory["steps"]
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == "test"
    assert [step["source"] for step in steps] == ["user", "agent", "agent"]
    assert [step["step_id"] for step in steps] == [1, 2, 3]
    assert steps[2]["timestamp"] == trace[0]["timestamp"]
    assert steps[2]["extra"]["graynet"]["broker_status"] == "error"
    result = steps[2]["observation"]["results"][0]
    assert result["source_call_id"] == steps[2]["tool_calls"][0]["tool_call_id"]
    assert result["content"] == "denied"
    assert result["extra"]["graynet"]["raw_source"] == "logs/events.jsonl:4"
    assert steps[1].get("timestamp") is None
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100
    assert trajectory["final_metrics"]["total_cached_tokens"] == 50
    assert "total_completion_tokens" not in trajectory["final_metrics"]
    assert all("llm_call_count" not in step for step in steps)
    assert TrajectoryValidator().validate(trajectory)
    assert "Check counters" in render_trajectory(trajectory)


def test_timeout_preserves_partial_calls_and_unmatched_broker_observations():
    events = '{"type":"item.started","item":{"id":"item_0","type":"mcp_tool_call","tool":"ping","arguments":{}}}\ntruncated'
    trace = [{"tool": "ping", "arguments": {}, "status": "completed", "result": "reply"}]
    trajectory = normalize_trajectory(
        events, prompt=None, trace=trace, metadata={"status": "timeout"}
    )
    assert trajectory["extra"]["graynet"]["status"] == "timeout"
    assert len(trajectory["extra"]["graynet"]["warnings"]) == 1
    assert len(trajectory["steps"]) == 1
    assert trajectory["steps"][0]["extra"]["graynet"]["provider_completion_missing"]
    observation = trajectory["steps"][0]["observation"]["results"][0]
    assert observation["content"] == "reply"
    assert observation["extra"]["graynet"]["delivery_to_agent"] == "unconfirmed"
    assert TrajectoryValidator().validate(trajectory)


def test_repeated_identical_tools_match_distinct_broker_calls():
    item = {
        "type": "mcp_tool_call",
        "tool": "throughput_probe",
        "arguments": {},
        "status": "completed",
    }
    raw = "\n".join(
        json.dumps({"type": "item.completed", "item": {**item, "id": str(index)}})
        for index in range(2)
    )
    trace = [
        {
            "tool": "throughput_probe",
            "arguments": {},
            "timestamp": f"2026-10-03T01:02:0{index}+00:00",
        }
        for index in range(2)
    ]
    steps = normalize_trajectory(raw, prompt=None, trace=trace, metadata={})["steps"]
    assert [step["timestamp"] for step in steps] == [record["timestamp"] for record in trace]
    assert steps[0]["tool_calls"][0]["tool_call_id"] != steps[1]["tool_calls"][0]["tool_call_id"]


def test_readable_trajectory_does_not_break_fences_on_agent_text():
    trajectory = normalize_trajectory(
        "", prompt="```\n# malicious formatting\n```", trace=[], metadata={}
    )
    rendered = render_trajectory(trajectory)
    assert "````\n```\n# malicious formatting\n```\n````" in rendered


def test_broker_only_calls_are_retained_without_claiming_model_turns():
    trace = [{"tool": "probe", "arguments": {}, "status": "completed", "result": {"throughput": 2}}]
    trajectory = normalize_trajectory(
        "", prompt=None, trace=trace, metadata={"driver": "codex", "status": "timeout"}
    )
    step = trajectory["steps"][0]
    assert "unknown" in step["extra"]["graynet"]["ordering"]
    assert "llm_call_count" not in step
    assert (
        step["observation"]["results"][0]["extra"]["graynet"]["delivery_to_agent"] == "unconfirmed"
    )
    smoke = normalize_trajectory("", prompt=None, trace=trace, metadata={"driver": "smoke"})
    assert smoke["steps"][0]["llm_call_count"] == 0
    assert "metrics" not in smoke["steps"][0]


def test_recorded_reasoning_is_separate_from_message_and_never_fabricated():
    raw = json.dumps(
        {
            "type": "item.completed",
            "item": {"id": "reasoning", "type": "reasoning", "text": "Compare both paths."},
        }
    )
    trajectory = normalize_trajectory(raw, prompt=None, trace=[], metadata={"driver": "codex"})
    step = trajectory["steps"][0]
    assert step["reasoning_content"] == "Compare both paths."
    assert step["message"] == ""
    prompt_only = normalize_trajectory("", prompt="Diagnose", trace=[], metadata={})
    assert "reasoning_content" not in prompt_only["steps"][0]


def test_empty_timeout_produces_valid_partial_record_not_fake_agent_answer():
    trajectory = normalize_trajectory("", prompt=None, trace=[], metadata={"status": "timeout"})
    assert trajectory["steps"][0]["source"] == "system"
    assert trajectory["final_metrics"]["extra"]["coverage"] == "partial_or_unknown"
    assert "total_prompt_tokens" not in trajectory["final_metrics"]
    assert TrajectoryValidator().validate(trajectory)


def test_zero_usage_is_preserved_and_cache_not_added_to_prompt():
    raw = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "turn.started"},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 0},
            },
        ]
    )
    metrics = normalize_trajectory(
        raw, prompt="Diagnose", trace=[], metadata={"status": "completed"}
    )["final_metrics"]
    assert metrics["total_prompt_tokens"] == 100 and metrics["total_cached_tokens"] == 50
    assert metrics["total_completion_tokens"] == 0
    assert "total_cost_usd" not in metrics
    assert metrics["extra"]["coverage"] == "complete"


def test_missing_usage_in_one_turn_is_not_reported_as_a_complete_total():
    raw = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "turn.completed", "usage": {"input_tokens": 100}},
            {"type": "turn.completed", "usage": {}},
        ]
    )
    metrics = normalize_trajectory(raw, prompt="Diagnose", trace=[], metadata={})["final_metrics"]
    assert "total_prompt_tokens" not in metrics
    assert metrics["extra"]["observed_usage"]["input_tokens"] == 100


def test_partial_messages_invalid_timestamps_unknown_items_and_runtime_error_survive():
    raw = "\n".join(
        json.dumps(event)
        for event in [
            {
                "type": "item.started",
                "timestamp": "not-a-time",
                "item": {"id": "partial", "type": "agent_message", "text": "Checking"},
            },
            {
                "type": "item.completed",
                "item": {"id": "other", "type": "future_item", "value": "keep me"},
            },
            {"type": "turn.failed", "error": {"message": "deadline"}},
        ]
    )
    trajectory = normalize_trajectory(raw, prompt=None, trace=[], metadata={})
    assert any(step["extra"]["graynet"].get("incomplete") for step in trajectory["steps"])
    assert any(step["source"] == "system" for step in trajectory["steps"])
    assert any(
        entry["event"].get("item", {}).get("value") == "keep me"
        for entry in trajectory["extra"]["graynet"]["provider_events"]
    )
    assert TrajectoryValidator().validate(trajectory)


def test_duplicate_completion_does_not_duplicate_call_or_consume_another_record():
    item = {
        "id": "same",
        "type": "mcp_tool_call",
        "tool": "probe",
        "arguments": {},
        "status": "completed",
        "result": "reply",
    }
    raw = "\n".join([json.dumps({"type": "item.completed", "item": item})] * 2)
    trace = [{"tool": "probe", "arguments": {}, "result": "reply"}]
    trajectory = normalize_trajectory(raw, prompt=None, trace=trace, metadata={})
    assert len(trajectory["steps"]) == 1
    assert trajectory["steps"][0]["extra"]["graynet"]["broker_source"] == "tools.json[0]"


def test_multiple_turns_can_reuse_codex_item_ids_without_merging_actions():
    item = {
        "id": "item_0",
        "type": "mcp_tool_call",
        "tool": "probe",
        "arguments": {},
        "status": "completed",
        "result": "reply",
    }
    raw = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "turn.started"},
            {"type": "item.completed", "item": item},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 10},
            },
            {"type": "turn.started"},
            {"type": "item.completed", "item": item},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 200, "cached_input_tokens": 100, "output_tokens": 20},
            },
        ]
    )
    trajectory = normalize_trajectory(raw, prompt=None, trace=[], metadata={"status": "completed"})
    assert len(trajectory["steps"]) == 2
    ids = [step["tool_calls"][0]["tool_call_id"] for step in trajectory["steps"]]
    assert len(set(ids)) == 2
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 300
    assert trajectory["final_metrics"]["total_completion_tokens"] == 30


def test_mcp_output_becomes_tool_content_not_an_escaped_envelope():
    output = {
        "content": [{"type": "text", "text": '{"stdout":"packet counters","exit_code":0}'}],
        "structured_content": None,
    }
    raw = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "id": "call",
                "type": "mcp_tool_call",
                "tool": "show",
                "arguments": {},
                "result": output,
            },
        }
    )
    trajectory = normalize_trajectory(raw, prompt=None, trace=[], metadata={})
    content = trajectory["steps"][0]["observation"]["results"][0]["content"]
    assert json.loads(content) == {"stdout": "packet counters", "exit_code": 0}


@pytest.mark.parametrize("mutation", ["schema", "step_id", "reference", "unexpected"])
def test_schema_validators_reject_non_atif_documents(mutation):
    trajectory = normalize_trajectory(
        "", prompt=None, trace=[{"tool": "probe", "arguments": {}, "result": "reply"}], metadata={}
    )
    if mutation == "schema":
        trajectory["schema_version"] = 1
    elif mutation == "step_id":
        trajectory["steps"][0]["step_id"] = 2
    elif mutation == "reference":
        trajectory["steps"][0]["observation"]["results"][0]["source_call_id"] = "missing-call"
    else:
        trajectory["steps"][0]["kind"] = "custom"
    with pytest.raises(ValidationError):
        Trajectory.model_validate(trajectory)
    assert not TrajectoryValidator().validate(trajectory)
