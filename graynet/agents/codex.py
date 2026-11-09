"""Codex CLI adapter with a disposable container and evaluator-side MCP tools."""

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graynet.agents.base import Diagnosis
from graynet.artifacts.records import write_json
from graynet.artifacts.trajectory import save_trajectory
from graynet.evaluation.integrity import MECHANISMS, audit_evidence, audit_observations
from graynet.runtime.catalog import preflight_model
from graynet.runtime.codex import (
    CodexRuntimeError,
    IsolatedCodexSession,
    SessionError,
    SessionInterrupted,
    SessionOutput,
    codex_config,
    parse_events,
)
from graynet.tools.broker import ToolBroker
from graynet.tools.network import MANAGED_INTERFACES, NetworkTools, tool_definitions

COMPONENT_LABELS = [
    *MANAGED_INTERFACES,
    *[
        f"{node}:{interface}"
        for node, interfaces in MANAGED_INTERFACES.items()
        for interface in interfaces
    ],
]
MECHANISM_LABELS = list(MECHANISMS)

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "detected": {"type": "boolean"},
        "component": {
            "type": ["string", "null"],
            "enum": [*COMPONENT_LABELS, None],
            "description": "Choose the managed node or interface where the problem is located, without extra prose.",
        },
        "mechanism": {
            "type": ["string", "null"],
            "enum": [*MECHANISM_LABELS, None],
            "description": "Choose a fault category; explain the detailed cause in explanation.",
        },
        "explanation": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cite unique observation_id values (obs-0001 etc.) from successful network tools, not inventory or invented observations.",
        },
    },
    "required": ["detected", "component", "mechanism", "explanation", "evidence"],
    "additionalProperties": False,
}


CodexAgentError = CodexRuntimeError


def diagnosis_schema(components: list[str]) -> dict[str, Any]:
    schema = deepcopy(DIAGNOSIS_SCHEMA)
    schema["properties"]["component"]["enum"] = [*components, None]
    return schema


def parse_diagnosis(text: str, schema: dict[str, Any] = DIAGNOSIS_SCHEMA) -> Diagnosis:
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise CodexAgentError("Codex did not submit valid diagnosis JSON") from exc
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise CodexAgentError("Codex diagnosis has missing or unexpected fields")
    if type(value["detected"]) is not bool:
        raise CodexAgentError("detected must be a boolean")
    for field in ("component", "mechanism"):
        label = value[field]
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise CodexAgentError(f"{field} must be a nonempty string or null")
        if value["detected"] != (label is not None):
            raise CodexAgentError(
                "a detected incident needs component and mechanism; a healthy answer needs null labels"
            )
        if label not in schema["properties"][field]["enum"]:
            raise CodexAgentError(f"{field} is not in the public answer vocabulary")
    if not isinstance(value["explanation"], str) or not value["explanation"].strip():
        raise CodexAgentError("diagnosis must explain its conclusion")
    evidence = value["evidence"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise CodexAgentError("diagnosis must include observed evidence")
    return Diagnosis(
        value["detected"],
        value["component"],
        value["mechanism"],
        value["explanation"],
        tuple(evidence),
    )


class CodexDriver:
    def __init__(
        self,
        tools: NetworkTools,
        *,
        auth_file: Path,
        artifacts: Path,
        model: str | None = None,
        effort: str | None = None,
        timeout_seconds: int = 300,
        max_tool_calls: int = 64,
        image: str = "graynet-codex:0.1",
        runtime_context: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if (
            type(timeout_seconds) is not int
            or timeout_seconds < 1
            or type(max_tool_calls) is not int
            or max_tool_calls < 1
        ):
            raise ValueError("agent time and tool budgets must be positive integers")
        self.tools = tools
        self.schema = diagnosis_schema(tools.policy.components)
        self.auth_file = auth_file
        self.artifacts = artifacts
        self.model = model
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.max_tool_calls = max_tool_calls
        self.image = image
        self.runtime_context = runtime_context
        self.trace: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "driver": "codex",
            "model": model,
            "reasoning_effort": effort,
            "timeout_seconds": timeout_seconds,
            "max_tool_calls": max_tool_calls,
            "estimated_cost_usd": None,
            "tool_definitions": [
                {
                    "type": "function",
                    "function": {
                        "name": definition["name"],
                        "description": definition["description"],
                        "parameters": definition["inputSchema"],
                    },
                }
                for definition in tool_definitions(tools.policy)
            ],
            "answer_schema_sha256": hashlib.sha256(
                json.dumps(self.schema, sort_keys=True).encode()
            ).hexdigest(),
        }
        self.metadata["tool_definitions_sha256"] = hashlib.sha256(
            json.dumps(self.metadata["tool_definitions"], sort_keys=True).encode()
        ).hexdigest()

    def preflight(self) -> None:
        runtime = preflight_model(self.image, self.auth_file, self.model, self.effort)
        self.image, self.model, self.effort = runtime.image_id, runtime.model, runtime.effort
        self.provider_hosts = runtime.provider_hosts
        self.metadata.update(
            image_id=runtime.image_id,
            codex_version=runtime.cli_version,
            model=self.model,
            reasoning_effort=self.effort,
            model_catalog=runtime.catalog,
        )

    def diagnose(self, incident: str) -> Diagnosis:
        if self.runtime_context is not None:
            self.metadata.update(self.runtime_context())
        if "image_id" not in self.metadata:
            self.preflight()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        inputs, logs = self.artifacts / "input", self.artifacts / "logs"
        inputs.mkdir(exist_ok=True)
        logs.mkdir(exist_ok=True)
        instruction = (
            "You are a network operations agent. Diagnose the reported incident using the network MCP tools. "
            "Do not change network configuration. Work autonomously. Investigate before concluding, and "
            "submit a structured diagnosis with observations supporting your conclusion. Do not assume "
            "that a reported slowdown establishes a fault: a healthy network is also a possible answer. "
            "Use healthy_reference to compare observations under the same workload. Idle ping RTT is not the normal RTT under a saturating TCP workload. "
            f"You have {self.timeout_seconds} seconds and at most {self.max_tool_calls} network tool calls.\n\n"
            f"Incident report:\n{incident}\n\nOperational context:\n{self.tools.inventory}"
            f"\n\nAnswer format: choose component from {json.dumps(self.tools.policy.components)} and mechanism from "
            f"{json.dumps(MECHANISM_LABELS)}. Put additional detail in explanation and evidence. "
            "For a healthy network, detected must be false and both labels must be null."
            " In evidence, cite observation_id values returned by the network tools. Explain how those observations support your conclusion in explanation."
        )
        (inputs / "prompt.txt").write_text(instruction, encoding="utf-8")
        (inputs / "incident.md").write_text(incident, encoding="utf-8")
        (inputs / "environment.md").write_text(self.tools.inventory, encoding="utf-8")
        (inputs / "config.toml").write_text(codex_config(self.model, self.effort), encoding="utf-8")
        write_json(inputs / "diagnosis.schema.json", self.schema)
        broker = ToolBroker(
            self.tools, max_calls=self.max_tool_calls, timeout_seconds=self.timeout_seconds + 30
        )
        session = IsolatedCodexSession(
            image=self.metadata["image_id"],
            model=self.model,
            effort=self.effort,
            auth_file=self.auth_file,
            provider_hosts=self.provider_hosts,
            timeout_seconds=self.timeout_seconds,
            broker=broker,
        )
        output = SessionOutput("", "", "", {"status": "error"})
        try:
            try:
                output = session.run(
                    instruction,
                    self.schema,
                    files={"incident.md": incident, "environment.md": self.tools.inventory},
                )
            except (SessionError, SessionInterrupted) as exc:
                output = exc.output
                raise
            self.metadata.update(output.metadata)
            events, _ = parse_events(output.stdout)
            self.metadata["event_count"] = len(events)
            self.metadata["turns"] = sum(event.get("type") == "turn.completed" for event in events)
            final = next(
                (
                    event["item"].get("text")
                    for event in reversed(events)
                    if event.get("type") == "item.completed"
                    and isinstance(event.get("item"), dict)
                    and event["item"].get("type") == "agent_message"
                ),
                None,
            )
            if not any(
                entry["status"] == "completed"
                and entry["tool"] not in ("network_inventory", "healthy_reference")
                and entry["result"].get("exit_code", 0) == 0
                for entry in broker.trace
            ):
                raise CodexAgentError("Codex submitted without inspecting the network")
            answer = parse_diagnosis(final, self.schema)
            write_json(self.artifacts / "diagnosis.json", asdict(answer))
            self.metadata["evidence_audit"] = audit_evidence(answer, broker.trace)
            if not self.metadata["evidence_audit"]["passed"]:
                raise CodexAgentError(
                    "diagnosis cites missing, duplicate, or non-network observation IDs"
                )
            self.metadata["status"] = "completed"
            return answer
        except BaseException:
            self.metadata["status"] = output.metadata.get("status")
            if self.metadata["status"] not in ("timeout", "interrupted"):
                self.metadata["status"] = "error"
            raise
        finally:
            status = self.metadata.get("status", "error")
            self.metadata.update(output.metadata)
            self.metadata["status"] = status
            self.trace = broker.trace
            self.metadata["visibility_audit"] = audit_observations(self.trace)
            self.metadata["tool_calls"] = len(self.trace)
            self.metadata["successful_tool_calls"] = sum(
                entry["status"] == "completed" for entry in self.trace
            )
            _, self.metadata["usage"] = parse_events(output.stdout)
            (logs / "events.jsonl").write_text(output.stdout, encoding="utf-8")
            (logs / "stderr.txt").write_text(output.stderr, encoding="utf-8")
            (logs / "egress.jsonl").write_text(output.proxy_log, encoding="utf-8")
            write_json(self.artifacts / "tools.json", self.trace)
            write_json(self.artifacts / "metadata.json", self.metadata)
            save_trajectory(self.artifacts, trace=self.trace, metadata=self.metadata)
