"""Evaluator-side network tools, with argument validation before Docker access."""

import json
import re
import threading
import time
from collections.abc import Callable
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Any

from graynet.environments.network import DiagnosticPolicy
from graynet.runtime.commands import run_process

MANAGED_INTERFACES = {
    "r1": ("eth1", "eth2", "eth3", "eth4"),
    "r2": ("eth1", "eth2", "eth3", "eth4"),
    "user-client": ("eth1",),
    "user-server": ("eth1",),
}
PUBLIC_ADDRESSES = {
    "10.1.0.1",
    "10.1.0.2",
    "10.2.0.1",
    "10.2.0.2",
    "10.3.0.1",
    "10.4.0.1",
    "10.12.0.1",
    "10.12.0.2",
    "10.12.1.1",
    "10.12.1.2",
}
FRR_SHOW_COMMANDS = {
    "show bgp ipv4 unicast summary json",
    "show bgp ipv4 unicast json",
    "show ip route json",
    "show interface json",
}
DEFAULT_POLICY = DiagnosticPolicy(
    MANAGED_INTERFACES,
    frozenset({"r1", "r2"}),
    frozenset(PUBLIC_ADDRESSES),
    "user-client",
    "10.2.0.2",
)


def queue_statistics(raw: str) -> str:
    """Expose operational counters, never simulator names, handles or parameters.

    This projection is identical in healthy and faulty runs. The privileged
    evaluator keeps raw output separately for case validation.
    """
    rows = []
    for block in re.split(r"(?=qdisc )", raw):
        if not block.startswith("qdisc "):
            continue
        sent = re.search(r"Sent (\d+) bytes (\d+) pkt.*?dropped (\d+), overlimits (\d+)", block)
        backlog = re.search(r"backlog (\d+)b (\d+)p", block)
        if not sent:
            raise ValueError("queue statistics unavailable")
        rows.append(
            {
                "sent_bytes": int(sent[1]),
                "sent_packets": int(sent[2]),
                "dropped": int(sent[3]),
                "overlimits": int(sent[4]),
                "backlog_bytes": int(backlog[1]) if backlog else 0,
                "backlog_packets": int(backlog[2]) if backlog else 0,
            }
        )
    if not rows:
        raise ValueError("queue statistics unavailable")
    return json.dumps({"queues": rows})


def interface_observations(raw: str, policy: DiagnosticPolicy, node: str) -> str:
    """Data-plane state only; no emulator queues, namespaces, or lab aliases."""
    data = json.loads(raw)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("interface telemetry unavailable")
    fields = {
        "ifname",
        "flags",
        "mtu",
        "operstate",
        "link_type",
        "address",
        "broadcast",
        "addr_info",
        "stats64",
        "stats",
    }
    return json.dumps(
        [
            {key: value for key, value in item.items() if key in fields}
            for item in data
            if item.get("ifname") in policy.interfaces[node]
        ]
    )


def frr_interface_observations(raw: str, policy: DiagnosticPolicy, node: str) -> str:
    data = json.loads(raw)
    if not isinstance(data, dict) or any(not isinstance(item, dict) for item in data.values()):
        raise ValueError("interface telemetry unavailable")
    fields = {
        "administrativeStatus",
        "operationalStatus",
        "linkDetection",
        "linkUps",
        "linkDowns",
        "lastLinkUp",
        "lastLinkDown",
        "vrfName",
        "mtu",
        "speed",
        "txqlen",
        "flags",
        "type",
        "hardwareAddress",
        "ipAddresses",
        "metric",
        "protodown",
        "mplsEnabled",
    }
    return json.dumps(
        {
            interface: {key: value for key, value in item.items() if key in fields}
            for interface, item in data.items()
            if interface in policy.interfaces[node]
        }
    )


def _public_prefix(prefix: str, policy: DiagnosticPolicy) -> bool:
    try:
        network = IPv4Network(prefix, strict=False)
        return any(IPv4Address(address) in network for address in policy.addresses)
    except ValueError:
        return False


def route_observations(raw: str, policy: DiagnosticPolicy, node: str, *, frr: bool = False) -> str:
    data = json.loads(raw)
    if frr:
        if not isinstance(data, dict) or any(not isinstance(rows, list) for rows in data.values()):
            raise ValueError("route telemetry unavailable")
        visible = {}
        for prefix, rows in data.items():
            if not _public_prefix(prefix, policy):
                continue
            projected = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("route telemetry unavailable")
                hops = [
                    hop
                    for hop in row.get("nexthops", [])
                    if hop.get("interfaceName") in policy.interfaces[node]
                ]
                if hops:
                    projected.append({**row, "nexthops": hops})
            if projected:
                visible[prefix] = projected
        return json.dumps(visible)
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("route telemetry unavailable")
    return json.dumps(
        [
            item
            for item in data
            if item.get("dev") in policy.interfaces[node]
            and _public_prefix(
                "0.0.0.0/0" if item.get("dst") == "default" else item.get("dst", ""), policy
            )
        ]
    )


def socket_observations(raw: str, policy: DiagnosticPolicy) -> str:
    """Keep TCP state for public addresses, not management-plane connections."""
    lines = raw.splitlines()
    if not lines:
        return ""
    output = [lines[0]]
    visible = False
    for line in lines[1:]:
        if not line[:1].isspace():
            addresses = re.findall(r"(?<![\d.])((?:\d{1,3}\.){3}\d{1,3}):", line)
            visible = len(addresses) == 2 and all(
                address in policy.addresses for address in addresses
            )
        if visible:
            output.append(line)
    return "\n".join(output) + "\n"


def cpu_statistics(
    executor: Callable[..., Any], container: str, *, sample_seconds: float = 1.0
) -> dict[str, Any]:
    pid = executor(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container], timeout=5
    ).stdout.strip()
    if not pid.isdigit() or int(pid) <= 0:
        raise ValueError("endpoint resource namespace unavailable")

    def read():
        raw = _read_cgroup(pid, "cpu.stat")
        values = {
            parts[0]: int(parts[1]) for line in raw.splitlines() if len(parts := line.split()) == 2
        }
        if not {"usage_usec", "nr_throttled", "throttled_usec"} <= set(values):
            raise ValueError("endpoint CPU telemetry unavailable")
        return values

    before = read()
    started = time.monotonic()
    time.sleep(sample_seconds)
    after = read()
    elapsed = time.monotonic() - started
    quota = _read_cgroup(pid, "cpu.max").split()
    if len(quota) != 2:
        raise ValueError("endpoint CPU allocation unavailable")
    cores = None if quota[0] == "max" else int(quota[0]) / int(quota[1])
    used = (after["usage_usec"] - before["usage_usec"]) / elapsed / 1e6
    return {
        "sample_seconds": elapsed,
        "cpu_usage_cores": used,
        "allocated_cores": cores,
        "allocation_utilization": used / cores if cores else None,
        "throttled_periods": after["nr_throttled"] - before["nr_throttled"],
        "throttled_seconds": (after["throttled_usec"] - before["throttled_usec"]) / 1e6,
    }


def _read_cgroup(pid: str, name: str) -> str:
    # Out-of-band read: a saturated endpoint must not starve the diagnostic
    # process. Neither path nor PID is supplied by the evaluated agent.
    return Path(f"/proc/{pid}/root/sys/fs/cgroup/{name}").read_text(encoding="utf-8")


def validate_command(node: str, argv: list[str], policy: DiagnosticPolicy = DEFAULT_POLICY) -> None:
    """No shell, files, processes, management network, or config changes."""
    if node not in policy.interfaces:
        raise ValueError("node is not a managed device")
    if not isinstance(argv, list) or not argv or any(not isinstance(part, str) for part in argv):
        raise ValueError("argv must be a nonempty list of strings")
    allowed = argv in (["ip", "-j", "addr", "show"], ["ip", "-j", "route", "show"], ["ss", "-tin"])
    if argv[:4] == ["ip", "-j", "-s", "link"]:
        allowed = argv == ["ip", "-j", "-s", "link", "show"] or (
            len(argv) == 7 and argv[4:6] == ["show", "dev"] and argv[6] in policy.interfaces[node]
        )
    if argv[:3] == ["ip", "route", "get"]:
        allowed = len(argv) == 4 and argv[3] in policy.addresses
    if argv[:5] == ["tc", "-s", "qdisc", "show", "dev"]:
        allowed = len(argv) == 6 and argv[5] in policy.interfaces[node]
    if argv[:2] == ["vtysh", "-c"]:
        allowed = node in policy.routers and len(argv) == 3 and argv[2] in FRR_SHOW_COMMANDS
    if argv[:1] == ["ping"]:
        allowed = (
            len(argv) == 6
            and argv[:5] == ["ping", "-c", "3", "-W", "1"]
            and argv[5] in policy.addresses
        )
    if not allowed:
        raise ValueError("command is not an allowed read-only network operation")


def tool_definitions(
    policy: DiagnosticPolicy = DEFAULT_POLICY, *, probe_targets: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    targets = probe_targets or policy.flow_names
    probe_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    if targets:
        probe_schema["properties"]["flow"] = {"type": "string", "enum": list(targets)}
        probe_schema["required"] = ["flow"]
    return [
        {
            "name": "network_inventory",
            "description": "Read the operational topology, service expectation, and available diagnostic commands.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "run_command",
            "description": (
                "Run an allowed network command on any managed device. argv is an argument array, not a shell. "
                "Supported: ip -j addr show; ip -j route show; ip -j -s link show [dev ethN]; "
                "ip route get ADDRESS; tc -s qdisc show dev ethN; ss -tin; ping -c 3 -W 1 ADDRESS; "
                "on either router, vtysh -c with show bgp ipv4 unicast summary json, "
                "show bgp ipv4 unicast json, show ip route json, or show interface json. "
                "Queue observations contain operational counters, not queue configuration. "
                "Interface observations cover managed data-plane interfaces only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string", "enum": list(policy.interfaces)},
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                },
                "required": ["node", "argv"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "throughput_probe",
            "description": (
                "Measure TCP throughput at the receiver and mean TCP round trip time for the selected service flow under the standard workload."
                if targets
                else "Measure receiver-side TCP throughput and mean TCP RTT under the configured standard service workload. "
                "The probe uses the standard service workload and is serialized with monitoring "
                "so a second test flow does not create its own slowdown."
            ),
            "inputSchema": probe_schema,
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "healthy_reference",
            "description": (
                "Read normal historical throughput, the throughput requirement, and the preferred path for each service flow. This is history, not current network health."
                if targets
                else "Read five normal historical observations under the same standard workload: throughput, loaded TCP RTT, idle ping RTT, and queue counters for that reference window. Compare like measurement types and workloads. This is history, not current network health."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "resource_probe",
            "description": "Sample endpoint CPU usage and allocation throttling for one second. No process lists or configuration changes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "enum": [node for node in policy.interfaces if node not in policy.routers],
                    }
                },
                "required": ["node"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
    ]


class NetworkTools:
    def __init__(
        self,
        inventory: str,
        *,
        lab_name: str,
        workload_lock: threading.Lock,
        executor: Callable[..., Any] = run_process,
        policy: DiagnosticPolicy = DEFAULT_POLICY,
        probe: Callable[..., dict[str, Any]] | None = None,
        reference: Callable[[], dict[str, Any]] | None = None,
        probe_targets: tuple[str, ...] = (),
    ) -> None:
        self.inventory = inventory
        self.lab_name = lab_name
        self.workload_lock = workload_lock
        self.executor = executor
        self.policy = policy
        self.probe = probe
        self.reference = reference
        self.probe_targets = probe_targets or policy.flow_names
        if not set(self.probe_targets) <= set(policy.flow_names) or len(
            set(self.probe_targets)
        ) != len(self.probe_targets):
            raise ValueError("probe targets must be distinct public flow names")

    def definitions(self) -> list[dict[str, Any]]:
        return tool_definitions(self.policy, probe_targets=self.probe_targets)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        if name == "network_inventory" and not arguments:
            return {
                "context": self.inventory,
                "commands": self.definitions()[1]["description"],
            }
        if name == "healthy_reference" and not arguments:
            if self.reference is None:
                raise ValueError("normal workload reference is unavailable")
            return self.reference()
        if name == "throughput_probe" and self.probe_targets:
            if set(arguments) != {"flow"} or arguments["flow"] not in self.probe_targets:
                raise ValueError("flow is not an available probe target")
            if self.probe is None:
                raise ValueError("named throughput probe is unavailable")
            return self.probe(arguments["flow"])
        if name == "throughput_probe" and not arguments:
            if self.probe is not None:
                return self.probe()
            with self.workload_lock:
                result = self.executor(
                    [
                        "docker",
                        "exec",
                        f"clab-{self.lab_name}-{self.policy.client}",
                        "iperf3",
                        "-c",
                        self.policy.server_ip,
                        "-t",
                        "3",
                        "-J",
                    ],
                    timeout=15,
                )
            data = json.loads(result.stdout)
            if "error" in data:
                raise RuntimeError("service throughput probe failed")
            return {"throughput_mbps": data["end"]["sum_received"]["bits_per_second"] / 1e6}
        if name == "run_command" and set(arguments) == {"node", "argv"}:
            node, argv = arguments["node"], arguments["argv"]
            if not isinstance(node, str):
                raise ValueError("node must be a string")
            validate_command(node, argv, self.policy)
            result = self.executor(
                [
                    "docker",
                    "exec",
                    f"clab-{self.lab_name}-{node}",
                    *argv,
                ],
                timeout=12,
                check=False,
            )
            output = (
                queue_statistics(result.stdout)
                if argv[0] == "tc" and result.returncode == 0
                else result.stdout
            )
            if (
                result.returncode == 0
                and argv[:2] == ["ip", "-j"]
                and ("addr" in argv or "link" in argv)
            ):
                output = interface_observations(result.stdout, self.policy, node)
            if result.returncode == 0 and argv == ["ip", "-j", "route", "show"]:
                output = route_observations(result.stdout, self.policy, node)
            if result.returncode == 0 and argv == ["vtysh", "-c", "show interface json"]:
                output = frr_interface_observations(result.stdout, self.policy, node)
            if result.returncode == 0 and argv == ["vtysh", "-c", "show ip route json"]:
                output = route_observations(result.stdout, self.policy, node, frr=True)
            if result.returncode == 0 and argv == ["ss", "-tin"]:
                output = socket_observations(result.stdout, self.policy)
            # Docker/runtime error messages can contain private container names
            # and paths. Ping's loss/RTT output remains useful on a failed probe.
            if result.returncode != 0 and argv[0] != "ping":
                output = ""
            return {
                "stdout": output[:24000],
                "stderr": "network command failed" if result.returncode else "",
                "exit_code": result.returncode,
            }
        if name == "resource_probe" and set(arguments) == {"node"}:
            node = arguments["node"]
            if (
                not isinstance(node, str)
                or node not in self.policy.interfaces
                or node in self.policy.routers
            ):
                raise ValueError("resource observations are available only on managed endpoints")
            return cpu_statistics(self.executor, f"clab-{self.lab_name}-{node}")
        raise ValueError("unknown tool or unexpected arguments")
