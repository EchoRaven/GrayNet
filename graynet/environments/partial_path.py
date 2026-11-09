"""Measure two services on separate routed links under the same workload."""

import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any

from graynet.environments.base import Phase
from graynet.environments.containerlab import ContainerlabBackend, _positive_number
from graynet.environments.faults import loss_parameters
from graynet.environments.network import ServiceFlow
from graynet.evaluation.validation import validate_partial_paths
from graynet.runtime.commands import CommandError, run_process
from graynet.scenarios.definitions import CaseDefinition


class PartialPathBackend(ContainerlabBackend):
    """Run parallel workload pairs and keep path and injector evidence private."""

    def __init__(self, case: CaseDefinition):
        super().__init__(case)
        assert self.network.flows and case.partial_path is not None
        self.flows = self.network.flows
        self.selection = case.partial_path
        self.report_flow = self.selection.flow or next(iter(self.flows))
        self.pacing_mbps = _positive_number(case.workload.get("rate_mbps"), "workload.rate_mbps")
        self._flow_tests = dict.fromkeys(self.flows, 1)

    def _service_servers(self) -> tuple[str, ...]:
        return tuple(flow.server for flow in self.network.flows.values())

    def _configure_endpoint_resources(self) -> None:
        self.evidence["endpoint_resources"] = {
            name: self._configure_endpoint(flow.server) for name, flow in self.flows.items()
        }

    def _checks(self) -> dict[str, Any]:
        checks = super()._checks()
        checks["reachability"] = {
            name: (
                {"ping": checks["ping"], "rtt_avg_ms": checks["rtt_avg_ms"]}
                if flow.client == self.client
                else self._ping(flow.client, flow.server_ip)
            )
            for name, flow in self.flows.items()
        }
        return checks

    def _sample_flow(self, name: str) -> dict[str, Any]:
        flow = self.flows[name]
        ready = self._wait_for_workload_ready(server=flow.server, expected=self._flow_tests[name])
        started = datetime.now(UTC).isoformat()
        start = time.monotonic()
        raw = self._docker(
            flow.client,
            "iperf3",
            "-c",
            flow.server_ip,
            "-B",
            flow.client_ip,
            "-b",
            f"{self.pacing_mbps:g}M",
            "-t",
            str(self.duration),
            "-J",
            timeout=self.duration + 15,
        )
        end = time.monotonic()
        data = json.loads(raw)
        if "error" in data:
            raise CommandError(f"{name} iperf3 error: {data['error']}")
        self._flow_tests[name] += 1
        return {
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "started_monotonic": start,
            "finished_monotonic": end,
            "throughput_mbps": data["end"]["sum_received"]["bits_per_second"] / 1e6,
            "iperf": data,
            "server_readiness": ready,
        }

    def _capture_flow(self, flow: ServiceFlow) -> dict[str, str]:
        node, interface = flow.receiver
        pid = run_process(
            ["docker", "inspect", "--format", "{{.State.Pid}}", self._container(node)], timeout=5
        ).stdout.strip()
        if not pid.isdecimal() or int(pid) <= 0:
            raise CommandError("router namespace is unavailable for path verification")
        packet = run_process(
            [
                "nsenter",
                "-t",
                pid,
                "-n",
                "tcpdump",
                "-n",
                "-l",
                "-Q",
                "in",
                "-i",
                interface,
                "-c",
                "1",
                "-s",
                "96",
                "tcp",
                "and",
                "src",
                "host",
                flow.client_ip,
                "and",
                "dst",
                "host",
                flow.server_ip,
                "and",
                "dst",
                "port",
                "5201",
                "and",
                "greater",
                "200",
            ],
            timeout=self.duration + 8,
        ).stdout.strip()
        if not packet:
            raise CommandError(f"no workload packet observed at {node}:{interface}")
        return {"receiver": f"{node}:{interface}", "packet": packet}

    def _sample_pair(self, *, prove_path: bool = False) -> dict[str, dict]:
        with self.workload_lock, ThreadPoolExecutor(max_workers=2 * len(self.flows)) as pool:
            captures = {
                name: pool.submit(self._capture_flow, flow)
                for name, flow in self.flows.items()
                if prove_path
            }
            samples = {name: pool.submit(self._sample_flow, name) for name in self.flows}
            pair = {name: sample.result() for name, sample in samples.items()}
            overlap = min(row["finished_monotonic"] for row in pair.values()) - max(
                row["started_monotonic"] for row in pair.values()
            )
            if overlap < self.duration * 0.75:
                raise CommandError("the two service measurements did not sufficiently overlap")
            for name, capture in captures.items():
                pair[name]["path_observation"] = capture.result()
            return pair

    def _impairment(self, phase: str) -> dict[str, dict]:
        self.fault.verify(phase)
        snapshots = {}
        for name, flow in self.flows.items():
            node, interface = flow.sender
            raw = self._docker(node, "tc", "-d", "-s", "qdisc", "show", "dev", interface)
            if "qdisc netem" not in raw:
                raise CommandError(f"expected service queue is absent at {node}:{interface}")
            loss, seed = loss_parameters(raw)
            expected = phase == "incident" and name == self.selection.flow
            if expected:
                if (
                    not math.isclose(loss, self.fault.value, abs_tol=0.0001)
                    or seed != self.fault.seed
                ):
                    raise CommandError(
                        f"active loss rate or seed did not match at {node}:{interface}"
                    )
            elif loss != 0:
                raise CommandError(f"unexpected link impairment at {node}:{interface}")
            drops = re.search(r"\bdropped (\d+)\b", raw)
            snapshots[name] = {"raw": raw.strip(), "drops": int(drops[1]) if drops else 0}
        return snapshots

    def _measure_pairs(self, evidence: dict, phase: str) -> dict[str, list[float]]:
        if "flows" not in evidence:
            self._start_measurement(evidence, phase)
        records = evidence["flows"]
        while len(records[self.report_flow]) < self.case.validation.min_samples:
            self._record_pair(evidence, self._sample_pair())
        evidence["impairment_after"] = self._impairment(phase)
        evidence["checks"] = self._checks()
        affected = self.selection.flow
        if (
            phase == "incident"
            and affected is not None
            and (
                evidence["impairment_after"][affected]["drops"]
                <= evidence["impairment_before"][affected]["drops"]
            )
        ):
            raise CommandError("the configured impairment dropped no packets during measurement")
        return {
            name: [record["throughput_mbps"] for record in samples]
            for name, samples in records.items()
        }

    def _start_measurement(self, evidence: dict, phase: str) -> None:
        evidence["flows"] = {name: [] for name in self.flows}
        evidence["impairment_before"] = self._impairment(phase)
        self._record_pair(evidence, self._sample_pair(prove_path=True))

    @staticmethod
    def _record_pair(evidence: dict, pair: dict[str, dict]) -> None:
        for name, record in pair.items():
            evidence["flows"][name].append(record)

    def measure(self, case: CaseDefinition, phase: Phase) -> list[float]:
        evidence = self.evidence["phases"][phase] = {}
        evidence["throughput_mbps"] = self._measure_pairs(evidence, phase)
        return evidence["throughput_mbps"][self.report_flow]

    def start_agent_window(self, case: CaseDefinition) -> None:
        evidence = self.evidence["agent_window"] = {}
        evidence["initial_checks"] = self._checks()
        self._start_measurement(evidence, "incident")

    def finish_agent_window(self, case: CaseDefinition, baseline: tuple[float, ...]):
        evidence = self.evidence["agent_window"]
        evidence["throughput_mbps"] = self._measure_pairs(evidence, "incident")
        return self.validate(
            baseline, evidence["throughput_mbps"][self.report_flow], phase="agent_window"
        )

    def probe_for_agent(self, flow: str) -> dict[str, Any]:
        """Run the same workload pair and return only the selected flow."""
        evidence = self.evidence["agent_window"]
        pair = self._sample_pair()
        self._record_pair(evidence, pair)
        record = pair[flow]
        return {
            "flow": flow,
            "throughput_mbps": record["throughput_mbps"],
            "duration_seconds": self.duration,
            "tcp_mean_rtt_ms": self._tcp_mean_rtt(record),
            "measurement_started_at": record["started_at"],
            "measurement_finished_at": record["finished_at"],
        }

    def healthy_reference(self) -> dict[str, Any]:
        """Return measured healthy history for both services in either arm."""
        phases = self.evidence.get("phases", {})
        baseline = phases.get("baseline", {}).get("throughput_mbps", {})
        if not baseline:
            raise ValueError("normal workload reference is unavailable")
        fraction = self.network.service_min_fraction
        return {
            "flows": {
                name: {
                    "throughput_mbps": baseline[name][:5],
                    "service_min_mbps": fraction * statistics.median(baseline[name][:5]),
                    "preferred_egress": ":".join(flow.sender),
                    "preferred_ingress": ":".join(flow.receiver),
                }
                for name, flow in self.flows.items()
            },
            "workload_duration_seconds": self.duration,
        }

    def validate(self, baseline, incident, recovery=None, *, phase="incident"):
        phases = self.evidence["phases"]
        baseline_flows = phases["baseline"]["throughput_mbps"]
        incident_evidence = phases["incident"] if phase == "incident" else self.evidence[phase]
        recovery_flows = phases["recovery"]["throughput_mbps"] if recovery is not None else None
        reports = validate_partial_paths(
            self.case.validation,
            self.selection,
            baseline_flows,
            incident_evidence["throughput_mbps"],
            recovery_flows,
        )
        incident_evidence["flow_validation"] = {
            name: asdict(report) for name, report in reports.items()
        }
        reasons = tuple(
            f"{name}:{reason}" for name, row in reports.items() for reason in row.result.reasons
        )
        return replace(reports[self.report_flow].result, valid=not reasons, reasons=reasons)
