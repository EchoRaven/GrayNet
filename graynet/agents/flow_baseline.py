"""A deterministic comparison policy that uses only the public tool broker."""

import json
from collections import Counter
from typing import Any

from graynet.agents.base import FlowDiagnosis
from graynet.evaluation.flow_diagnosis import flow_signals, queue_command, service_state
from graynet.tools.broker import ToolBroker
from graynet.tools.network import NetworkTools

PROBE_CALLS = 4
BASELINE_CALLS = 9


class FlowProbeBaseline:
    """Measure available flows evenly and abstain when evidence is incomplete."""

    def __init__(self, tools: NetworkTools, *, max_calls: int, timeout_seconds: int) -> None:
        self.tools = tools
        self.max_calls = max_calls
        self.timeout_seconds = timeout_seconds
        self.trace: list[dict[str, Any]] = []

    def diagnose(self, incident: str) -> FlowDiagnosis:
        broker = ToolBroker(
            self.tools, max_calls=self.max_calls, timeout_seconds=self.timeout_seconds
        )
        self.trace = broker.trace

        def request(method: str, **params: Any) -> dict:
            response = broker.dispatch(
                {"jsonrpc": "2.0", "id": len(self.trace) + 1, "method": method, "params": params}
            )
            if response is None or "error" in response or response["result"].get("isError"):
                raise RuntimeError(f"diagnostic request failed: {method}")
            return response["result"]

        def call(name: str, arguments: dict) -> dict:
            result = request("tools/call", name=name, arguments=arguments)
            return json.loads(result["content"][0]["text"])

        definitions = request("tools/list")["tools"]
        probe = next(tool for tool in definitions if tool["name"] == "throughput_probe")
        targets = probe["inputSchema"]["properties"]["flow"]["enum"]
        reference = call("healthy_reference", {})["flows"]
        commands = [queue_command(row["preferred_egress"]) for row in reference.values()]
        for command in commands:
            call("run_command", command)
        counts: Counter[str] = Counter()
        for _ in range(PROBE_CALLS):
            target = min(targets, key=lambda name: (counts[name], name))
            call("throughput_probe", {"flow": target})
            counts[target] += 1
        for command in commands:
            call("run_command", command)

        signals = flow_signals(self.trace)
        detected = service_state(signals)
        slowed = [name for name, signal in signals.items() if signal.slowdown is True]
        affected = slowed[0] if len(slowed) == 1 else None
        component = signals[affected].loss_location if affected is not None else None
        state = {
            True: "Repeated throughput samples show a service violation.",
            False: "Repeated throughput samples meet the requirement on both flows.",
            None: "The probes do not establish whether either service violates its requirement.",
        }[detected]
        effect = (
            f" Repeated samples show a slowdown on {affected}."
            if affected
            else " No affected flow is established by these observations."
        )
        location = (
            f" The queue on {component} registered drops during the measurements."
            if component
            else " The observations do not establish a location."
        )
        return FlowDiagnosis(
            detected,
            affected,
            component,
            state + effect + location,
            tuple(row["observation_id"] for row in self.trace if row["status"] == "completed"),
        )
