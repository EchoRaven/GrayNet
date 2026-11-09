"""Codex exec events and broker records -> validated ATIF v1.7.

Adapted from SREGym's standalone Codex converter's linked call/observation
pattern. Exec events do not expose individual model-request boundaries. We do
not invent those boundaries, hidden reasoning, timestamps or token allocations.
Model provenance: atif/UPSTREAM.md.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from graynet.artifacts.atif import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from graynet.artifacts.records import write_json


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _provider_output(value: Any) -> str:
    """ATIF content is the tool's text, not an escaped MCP envelope."""
    if isinstance(value, dict) and isinstance(value.get("content"), list) and value["content"]:
        parts = value["content"]
        if all(
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            for part in parts
        ):
            return "\n".join(part["text"] for part in parts)
    return _stringify(value)


def _timestamp(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        except ValueError:
            pass
    return None


def _known_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def normalize_trajectory(
    raw: str,
    *,
    prompt: str | None,
    trace: list[dict[str, Any]],
    metadata: dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    steps: list[Step] = []
    calls: dict[str, Step] = {}
    messages: dict[str, Step] = {}
    partial_messages: dict[str, tuple[dict[str, Any], str]] = {}
    matched: set[int] = set()
    warnings: list[str] = []
    provider_events: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    turns_started = turns_completed = 0
    driver = metadata.get("driver") or "unknown"
    model = metadata.get("model")
    scripted = metadata.get("uses_model") is False

    def append(source: str, message: str, *, extra: dict[str, Any], **fields: Any) -> Step:
        step = Step(
            step_id=len(steps) + 1,
            source=source,
            message=message,
            extra={"graynet": extra},
            **fields,
        )
        steps.append(step)
        return step

    def broker_match(tool: str, arguments: dict[str, Any]) -> int | None:
        return next(
            (
                index
                for index, record in enumerate(trace)
                if index not in matched
                and record.get("tool") == tool
                and record.get("arguments", {}) == arguments
            ),
            None,
        )

    def attach_broker(step: Step, index: int, *, provider_completed: bool) -> None:
        matched.add(index)
        record = trace[index]
        details = step.extra["graynet"]
        details.update(
            broker_source=f"tools.json[{index}]",
            broker_status=record.get("status"),
            duration_seconds=record.get("duration_seconds"),
            broker_match="tool_and_arguments_fifo",
        )
        if step.timestamp is None and _timestamp(record.get("timestamp")):
            step.timestamp = record["timestamp"]
            details["timestamp_source"] = "broker_tool_start"
        if step.observation is None:
            value = record.get("result", record.get("error"))
            if value is not None:
                step.observation = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=step.tool_calls[0].tool_call_id,
                            content=_stringify(value),
                            extra={
                                "graynet": {
                                    "raw_source": f"tools.json[{index}]",
                                    "observed_by": "broker",
                                    "delivery_to_agent": "unconfirmed",
                                }
                            },
                        )
                    ]
                )
        if not provider_completed:
            details["provider_completion_missing"] = True

    if prompt is not None:
        append("user", prompt, extra={"raw_source": "input/prompt.txt", "kind": "prompt"})

    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        location = f"logs/events.jsonl:{line_number}"
        try:
            event = json.loads(line)
        except ValueError:
            warnings.append(f"Unparsed provider output at {location}")
            continue
        if not isinstance(event, dict):
            warnings.append(f"Non-object provider output at {location}")
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            if isinstance(event.get("thread_id"), str):
                session_id = event["thread_id"]
        elif event_type == "turn.started":
            turns_started += 1
        elif event_type == "turn.completed":
            turns_completed += 1
            usages.append(event.get("usage") if isinstance(event.get("usage"), dict) else {})
        elif event_type in ("error", "turn.failed"):
            append(
                "system",
                _stringify(event.get("message", event.get("error", event))),
                extra={"kind": "runtime_error", "raw_source": location, "event": event},
            )
        item = event.get("item")
        if not isinstance(item, dict):
            provider_events.append({"raw_source": location, "event": event})
            continue
        item_type = item.get("type")
        item_id = item.get("id") if isinstance(item.get("id"), str) else f"line-{line_number}"
        item_key = f"{turns_started}:{item_id}"
        if item_type in ("agent_message", "reasoning"):
            if event_type != "item.completed":
                partial_messages[item_key] = (item, location)
                continue
            partial_messages.pop(item_key, None)
            text = item.get("text") if isinstance(item.get("text"), str) else ""
            kind = "assistant_message" if item_type == "agent_message" else "reasoning_summary"
            step = messages.get(item_key)
            if step is None:
                step = append(
                    "agent",
                    text if item_type == "agent_message" else "",
                    extra={"kind": kind, "raw_source": location, "item_id": item_id},
                    model_name=model,
                    reasoning_content=text if item_type == "reasoning" else None,
                    timestamp=_timestamp(event.get("timestamp")),
                )
                messages[item_key] = step
            else:
                step.message = text if item_type == "agent_message" else ""
                step.reasoning_content = text if item_type == "reasoning" else None
                step.extra["graynet"]["raw_source"] = location
            continue
        if item_type != "mcp_tool_call":
            provider_events.append({"raw_source": location, "event": event})
            warnings.append(
                f"Unmapped item type {item_type!r} at {location}; retained in extra.graynet.provider_events"
            )
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            warnings.append(f"Non-object tool arguments at {location}; original retained")
            arguments = {} if arguments is None else {"value": arguments}
        step = calls.get(item_key)
        if step is None:
            tool = item.get("tool") if isinstance(item.get("tool"), str) else "unknown"
            step = append(
                "agent",
                "",
                extra={"kind": "tool_call", "raw_sources": [], "item_id": item_id},
                model_name=model,
                timestamp=_timestamp(event.get("timestamp")),
                tool_calls=[
                    ToolCall(
                        tool_call_id=f"codex:turn-{turns_started}:{item_id}",
                        function_name=tool,
                        arguments=arguments,
                        extra={
                            "server": item.get("server"),
                            "original_arguments": item.get("arguments"),
                        },
                    )
                ],
            )
            calls[item_key] = step
        elif isinstance(item.get("arguments"), dict):
            step.tool_calls[0].arguments = arguments
        details = step.extra["graynet"]
        details["raw_sources"].append(location)
        details["provider_status"] = item.get("status")
        details["turn_index"] = turns_started
        if event_type == "item.completed":
            details["provider_completed"] = True
            output = item.get("result")
            error = item.get("error")
            if output is not None or error is not None:
                step.observation = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=step.tool_calls[0].tool_call_id,
                            content=_provider_output(output if output is not None else error),
                            extra={
                                "graynet": {
                                    "raw_source": location,
                                    "provider_error": error,
                                    "observed_by": "provider",
                                }
                            },
                        )
                    ]
                )
            if "broker_source" not in details:
                index = broker_match(step.tool_calls[0].function_name, step.tool_calls[0].arguments)
                if index is not None:
                    attach_broker(step, index, provider_completed=True)

    # A broker may finish a call without Codex reporting completion before timeout.
    for step in calls.values():
        if "broker_source" not in step.extra["graynet"]:
            index = broker_match(step.tool_calls[0].function_name, step.tool_calls[0].arguments)
            if index is not None:
                attach_broker(
                    step,
                    index,
                    provider_completed=step.extra["graynet"].get("provider_completed", False),
                )
    for item_id, (item, location) in partial_messages.items():
        if not isinstance(item.get("text"), str) or not item["text"]:
            provider_events.append({"raw_source": location, "item": item})
            continue
        append(
            "agent",
            item["text"] if item["type"] == "agent_message" else "",
            extra={
                "kind": "partial_message",
                "item_id": item_id,
                "raw_source": location,
                "incomplete": True,
                "ordering": "position relative to completed provider steps unknown",
            },
            model_name=model,
            reasoning_content=item["text"] if item["type"] == "reasoning" else None,
        )
    for index, record in enumerate(trace):
        if index in matched:
            continue
        arguments = record.get("arguments", {})
        arguments = arguments if isinstance(arguments, dict) else {"value": arguments}
        step = append(
            "agent",
            "",
            extra={
                "kind": "broker_only_tool_call",
                "raw_source": f"tools.json[{index}]",
                "ordering": "broker order; position relative to provider messages unknown",
            },
            model_name=model,
            llm_call_count=0 if scripted or driver == "smoke" else None,
            tool_calls=[
                ToolCall(
                    tool_call_id=f"broker:{index + 1}",
                    function_name=record.get("tool") or "unknown",
                    arguments=arguments,
                    extra={"identifier_source": "broker_record_index"},
                )
            ],
        )
        attach_broker(step, index, provider_completed=False)

    if not steps:
        append("system", "No agent interaction was recorded.", extra={"kind": "empty_recording"})

    usage = {}
    observed_usage = {}
    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        known = [entry[name] for entry in usages if _known_count(entry.get(name))]
        observed_usage[name] = sum(known) if known else None
        usage[name] = sum(known) if usages and len(known) == len(usages) else None
        if not usages:
            value = (metadata.get("usage") or {}).get(name)
            usage[name] = value if _known_count(value) else None
    completeness = (
        "complete"
        if metadata.get("status") == "completed"
        and (scripted or turns_started > 0 and turns_started == turns_completed)
        else "partial_or_unknown"
    )
    metrics = FinalMetrics(
        total_prompt_tokens=usage["input_tokens"],
        total_cached_tokens=usage["cached_input_tokens"],
        total_completion_tokens=usage["output_tokens"],
        total_steps=len(steps),
        extra={
            "token_metrics_version": 2,
            "reasoning_output_tokens": usage["reasoning_output_tokens"],
            "usage_source": "turn.completed"
            if usages
            else "metadata"
            if metadata.get("usage")
            else "unknown",
            "coverage": completeness,
            "observed_usage": observed_usage,
        },
    )
    digest = hashlib.sha256(
        json.dumps({"raw": raw, "prompt": prompt, "trace": trace}, sort_keys=True).encode()
    ).hexdigest()
    trajectory = Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        trajectory_id=f"graynet-{digest}",
        agent=Agent(
            name=driver,
            version=metadata.get("codex_version") or metadata.get("version") or "unknown",
            model_name=model,
            tool_definitions=metadata.get("tool_definitions"),
            extra={
                "reasoning_effort": metadata.get("reasoning_effort"),
                "image_id": metadata.get("image_id"),
            },
        ),
        steps=steps,
        final_metrics=metrics,
        notes=(
            "Recorded from scripted calls to the network tool broker. No model was called."
            if scripted
            else "Converted from recorded exec events and broker observations. Per-inference boundaries and hidden reasoning are unavailable; llm_call_count is unknown for Codex steps. Broker-only results do not confirm delivery to the agent."
        ),
        extra={
            "graynet": {
                "status": metadata.get("status", "unknown"),
                "warnings": warnings,
                "provider_events": provider_events,
                "conversion_version": 1,
            }
        },
    )
    return Trajectory.model_validate(trajectory.to_json_dict()).to_json_dict()


def _block(value: Any) -> str:
    content = value if isinstance(value, str) else json.dumps(value, indent=2)
    longest = max((len(match) for match in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{content.rstrip()}\n{fence}\n"


def _observation(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return _block(value)
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        return "\n".join(
            _observation(part.get("text", part)) if isinstance(part, dict) else _block(part)
            for part in value["content"]
        )
    if isinstance(value, dict) and "stdout" in value:
        parts = [f"Exit code: {value.get('exit_code')}\n", _observation(value["stdout"])]
        if value.get("stderr"):
            parts.extend(["Stderr:\n", _block(value["stderr"])])
        return "\n".join(parts)
    return _block(value)


def render_trajectory(trajectory: dict[str, Any]) -> str:
    trajectory = Trajectory.model_validate(trajectory).to_json_dict()
    agent = trajectory["agent"]
    details = trajectory.get("extra", {}).get("graynet", {})
    lines = [
        "# Agent trajectory (ATIF v1.7)\n",
        f"Agent: {agent['name']}; version: {agent['version']}; model: {agent.get('model_name', 'unknown')}; status: {details.get('status', 'unknown')}.\n",
        trajectory.get("notes", "") + "\n",
    ]
    lines.extend(warning + "\n" for warning in details.get("warnings", []))
    for step in trajectory["steps"]:
        extra = step.get("extra", {}).get("graynet", {})
        lines.append(
            f"## {step['step_id']}. {step['source']} — {extra.get('kind', 'interaction')}\n"
        )
        source = extra.get("raw_source") or ", ".join(extra.get("raw_sources", []))
        if source:
            lines.append(f"Source: {source}\n")
        if step.get("timestamp"):
            lines.append(
                f"Recorded time: {step['timestamp']} ({extra.get('timestamp_source', 'provider')}).\n"
            )
        if extra.get("ordering"):
            lines.append(extra["ordering"] + "\n")
        if step["message"]:
            lines.append(_block(step["message"]))
        if step.get("reasoning_content"):
            lines.extend(["Recorded reasoning summary:\n", _block(step["reasoning_content"])])
        for call in step.get("tool_calls", []):
            lines.extend(
                [
                    f"Tool: {call['function_name']} ({call['tool_call_id']}); status: {extra.get('broker_status', extra.get('provider_status', 'unknown'))}.\n",
                    "Arguments:\n",
                    _block(call["arguments"]),
                ]
            )
        for result in step.get("observation", {}).get("results", []):
            observation_extra = result.get("extra", {}).get("graynet", {})
            if observation_extra.get("delivery_to_agent") == "unconfirmed":
                lines.append("Broker observation; delivery to agent is unconfirmed.\n")
            lines.extend(
                [
                    f"Observation for {result.get('source_call_id', 'system')}:\n",
                    _observation(result.get("content")),
                ]
            )
    lines.extend(["## Aggregate metrics\n", _block(trajectory.get("final_metrics"))])
    return "\n".join(lines)


def save_trajectory(
    directory: Path, *, trace: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    raw = directory / "logs/events.jsonl"
    prompt = directory / "input/prompt.txt"
    trajectory = normalize_trajectory(
        raw.read_text(encoding="utf-8") if raw.exists() else "",
        prompt=prompt.read_text(encoding="utf-8") if prompt.exists() else None,
        trace=trace,
        metadata=metadata,
        session_id=directory.parent.name,
    )
    write_json(directory / "trajectory.json", trajectory)
    (directory / "trajectory.md").write_text(render_trajectory(trajectory), encoding="utf-8")
