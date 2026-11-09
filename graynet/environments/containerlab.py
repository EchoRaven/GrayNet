"""Profile-driven live backend for GrayNet's Linux/FRR cases.

Run this on a Linux host with Docker and Containerlab. Only the host-side
validator imports it; none of these controls belong in an agent workspace.
"""

import hashlib
import json
import math
import os
import platform
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graynet.environments.base import Phase
from graynet.environments.faults import ScheduledTraffic, build_fault
from graynet.environments.network import load_network
from graynet.evaluation.validation import validate_sustained_case, validate_temporal_case
from graynet.paths import REPOSITORY_ROOT
from graynet.runtime.commands import CommandError, run_process
from graynet.scenarios.definitions import CaseDefinition


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def snapshot_sources(repository: Path, case: CaseDefinition, network) -> dict[str, Any]:
    """Capture actual shared inputs; do not sweep up adjacent credentials/cache."""
    sources = [
        *repository.joinpath("graynet").rglob("*.py"),
        *repository.joinpath("docker").rglob("*.py"),
        *repository.joinpath("docker").rglob("Dockerfile"),
        *case.directory.rglob("Dockerfile"),
        case.directory / "scenario.yaml",
        case.incident_path,
        case.topology,
        network.path,
        *network.bound_files,
        repository / "pyproject.toml",
        repository / "uv.lock",
    ]
    manifest = case.directory / "scenario.yaml"
    if case.environment_path is not None:
        sources.append(case.environment_path)
    elif manifest.is_file():
        import yaml

        raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("environment"), str):
            environment = (case.directory / raw["environment"]).resolve()
            if not environment.is_relative_to(case.directory):
                raise ValueError("public environment source escaped case directory")
            sources.append(environment)
    records = {}
    for path in sources:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        # Repository cases use repository-relative keys. A separately supplied
        # case may be external, but only its explicitly referenced files enter.
        key = (
            str(path.relative_to(repository))
            if path.is_relative_to(repository)
            else "external-case/" + str(path.relative_to(case.directory))
        )
        records[key] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content": path.read_text(encoding="utf-8"),
        }
    return records


class ContainerlabBackend:
    """Run bounded fault primitives against a topology and environment profile."""

    def __init__(self, case: CaseDefinition):
        self.case = case
        self.network = load_network(case)
        from graynet.evaluation.integrity import audit_case, environment_fingerprint

        self.integrity = audit_case(case, self.network)
        if not self.integrity["passed"]:
            raise ValueError("case visibility/answer-contract audit failed")
        self.environment_sha256 = environment_fingerprint(case, self.network)
        self.LAB_NAME = self.network.lab_name
        self.NODES = self.network.nodes
        self.USER_SERVER_IP = self.network.policy.server_ip
        self.LOAD_SERVER_IP = self.network.background.get("server_ip")
        self.client = case.workload["client"]
        self.server = case.workload["server"]
        self.load_client = self.network.background.get("client")
        self.load_server = self.network.background.get("server")
        self.queue_node = self.network.queue.get("node")
        self.queue_interface = self.network.queue.get("interface")
        self.owned = False
        self.evidence: dict[str, Any] = {
            "phases": {},
            "integrity": self.integrity,
            "environment_sha256": self.environment_sha256,
        }
        self.workload_lock = threading.Lock()
        self._next_workload_test = 1
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_errors: list[str] = []
        self.duration = _positive_int(
            case.workload.get("duration_seconds"), "workload.duration_seconds"
        )
        self.cpu_limit = case.workload.get("cpu_limit_cores")
        if (
            self.cpu_limit is not None
            and not 0.01 <= _positive_number(self.cpu_limit, "cpu_limit_cores") <= 2
        ):
            raise ValueError("cpu_limit_cores must be between 0.01 and 2")
        self.is_control = case.fault.get("kind") == "none"
        self.fault_duration = (
            180
            if self.is_control
            else _positive_int(case.fault.get("duration_seconds"), "fault.duration_seconds")
        )
        self.fault = build_fault(self)
        self.rate = self.fault.rate
        if self.fault_duration <= self.duration * case.validation.min_samples:
            raise ValueError("fault duration must exceed the incident measurement window")
        if case.workload.get("kind") != "iperf3":
            raise ValueError("this backend only supports iperf3 workloads")
        if case.validation.metric != "throughput_mbps":
            raise ValueError("the iperf3 backend validates throughput_mbps, not another metric")

    def _container(self, node: str) -> str:
        if node not in self.NODES:
            raise ValueError(f"unknown node: {node}")
        return f"clab-{self.LAB_NAME}-{node}"

    def _docker(self, node: str, *args: str, timeout: int = 30, detached: bool = False) -> str:
        command = ["docker", "exec"]
        if detached:
            command.append("-d")
        command += [self._container(node), *args]
        return run_process(command, timeout=timeout).stdout

    def _bgp_summary(self, node: str) -> dict[str, Any]:
        raw = self._docker(node, "vtysh", "-c", "show bgp ipv4 unicast summary json")
        start = raw.find("{")
        if start == -1:
            raise CommandError(f"{node} did not return a BGP JSON summary: {raw[-500:]}")
        summary = json.loads(raw[start:])
        peers = summary.get("peers", {})
        if set(peers) != set(self.network.bgp_peers[node]) or any(
            peer.get("state") != "Established" for peer in peers.values()
        ):
            raise CommandError(f"{node} expected BGP sessions are not established: {peers}")
        return {
            "peers": {address: peer["state"] for address, peer in peers.items()},
            "prefixes_received": {address: peer.get("pfxRcd") for address, peer in peers.items()},
        }

    def _route(self, node: str, destination: str, expected: str) -> str:
        route = self._docker(node, "ip", "route", "get", destination).strip()
        if expected not in route:
            raise CommandError(f"{node} route to {destination} changed: {route}")
        return route

    def _qdisc(self) -> dict[str, int]:
        raw = self._docker(
            self.queue_node, "tc", "-s", "qdisc", "show", "dev", self.queue_interface
        )
        if "qdisc tbf" not in raw:
            raise CommandError(f"primary-link shaper is absent: {raw}")
        matches = {key: re.search(rf"{key} (\d+)", raw) for key in ("dropped", "overlimits")}
        if any(match is None for match in matches.values()):
            raise CommandError(f"cannot parse primary-link queue counters: {raw}")
        return {key: int(match.group(1)) for key, match in matches.items()}

    def _queue_snapshot(self) -> dict[str, Any]:
        raw = self._docker(
            self.queue_node, "tc", "-s", "qdisc", "show", "dev", self.queue_interface
        )
        backlog = re.search(r"backlog (\d+)b (\d+)p", raw)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "raw": raw.strip(),
            "backlog_bytes": int(backlog.group(1)) if backlog else None,
            "backlog_packets": int(backlog.group(2)) if backlog else None,
        }

    def reserve_agent_window(self, timeout_seconds: int) -> None:
        # Hold the sustained fault through measurements, agent startup, its
        # deadline, and the final integrity check. Never silently restart it.
        self.fault_duration = max(
            self.fault_duration,
            timeout_seconds + self.duration * self.case.validation.min_samples + 120,
        )

    def _service_sample(self, duration: int) -> dict[str, Any]:
        with self.workload_lock:
            readiness = self._wait_for_workload_ready()
            started = datetime.now(UTC).isoformat()
            started_monotonic = time.monotonic()
            queue = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._docker,
                    self.client,
                    "iperf3",
                    "-c",
                    self.USER_SERVER_IP,
                    "-t",
                    str(duration),
                    "-J",
                    timeout=duration + 20,
                )
                while not future.done():
                    queue.append(self._queue_snapshot())
                    time.sleep(0.25)
                raw = future.result()
            finished_monotonic = time.monotonic()
            data = json.loads(raw)
            if "error" in data:
                raise CommandError(f"iperf3 error: {data['error']}")
            from graynet.tools.network import cpu_statistics

            resources = cpu_statistics(
                run_process, self._container(self.server), sample_seconds=0.5
            )
            self._next_workload_test += 1
            return {
                "started_at": started,
                "finished_at": datetime.now(UTC).isoformat(),
                "throughput_mbps": data["end"]["sum_received"]["bits_per_second"] / 1e6,
                "started_monotonic": started_monotonic,
                "finished_monotonic": finished_monotonic,
                "iperf": data,
                "queue_samples": queue,
                "resources": resources,
                "server_readiness": readiness,
            }

    def _read_workload_log(self, node: str) -> str:
        if node not in self._service_servers():
            raise ValueError("not an owned workload server")
        pid = run_process(
            ["docker", "inspect", "--format", "{{.State.Pid}}", self._container(node)], timeout=5
        ).stdout.strip()
        if not pid.isdigit() or int(pid) <= 0:
            raise CommandError("workload server namespace unavailable")
        # Read outside the endpoint CPU quota. This fixed, private log is never
        # an agent tool. Bound the read even during long investigations.
        with Path(f"/proc/{pid}/root/tmp/graynet-workload.log").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 64000))
            return stream.read(64000).decode("utf-8", errors="replace")

    def _wait_for_workload_ready(
        self,
        *,
        server: str | None = None,
        expected: int | None = None,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        # iperf closes/reopens its listener between tests. A completed client
        # does not establish that server cleanup has finished, especially
        # under CPU contention. Poll readiness, not a retry of the workload.
        server = server or self.server
        expected = self._next_workload_test if expected is None else expected
        started = time.monotonic()
        record = {
            "expected_test": expected,
            "observed_test": None,
            "wait_seconds": 0.0,
        }
        if self.network.flows:
            record["server"] = server
        self.evidence.setdefault("workload_readiness", []).append(record)
        while True:
            try:
                raw = self._read_workload_log(server)
            except FileNotFoundError:
                raw = ""
            generations = re.findall(r"Server listening on 5201 \(test #([1-9][0-9]*)\)", raw)
            observed = int(generations[-1]) if generations else None
            record.update(observed_test=observed, wait_seconds=time.monotonic() - started)
            if observed == expected:
                return record
            if observed is not None and observed > expected:
                raise CommandError(
                    "unexpected workload test: another client or server state change"
                )
            if record["wait_seconds"] >= timeout_seconds:
                raise CommandError("workload server did not become ready for the next test")
            time.sleep(0.25)

    def _host_bytes(self, node: str, direction: str) -> int:
        if node not in (self.load_client, self.load_server) or direction not in ("tx", "rx"):
            raise ValueError("unsupported host counter")
        raw = self._docker(
            node, "ip", "-j", "-s", "link", "show", "dev", self.network.background["interface"]
        )
        try:
            return int(json.loads(raw)[0]["stats64"][direction]["bytes"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise CommandError(f"cannot read {node} interface counters: {raw[-500:]}") from exc

    def _checks(self) -> dict[str, Any]:
        peers = {node: self._bgp_summary(node) for node in self.network.bgp_peers}
        routes = {
            name: self._route(
                route["node"], route["destination"], f"via {route['via']} dev {route['interface']}"
            )
            for name, route in self.network.routes.items()
        }
        return {"bgp": peers, "routes": routes, **self._ping(self.client, self.USER_SERVER_IP)}

    def _ping(self, client: str, server_ip: str) -> dict[str, Any]:
        ping = self._docker(client, "ping", "-c", "3", "-W", "1", server_ip, timeout=8)
        rtt = re.search(r"= [0-9.]+/([0-9.]+)/", ping)
        return {
            "ping": ping.strip(),
            "rtt_avg_ms": float(rtt.group(1)) if rtt else None,
        }

    def _load_running(self) -> bool:
        if self.load_client is None:
            return False
        result = run_process(
            ["docker", "exec", self._container(self.load_client), "pgrep", "-x", "iperf3"],
            check=False,
        )
        if result.returncode not in (0, 1):
            raise CommandError(f"could not inspect cross traffic: {result.stderr.strip()}")
        return result.returncode == 0

    def _disable_offloads(self) -> None:
        """Netem must operate on wire-sized packets, not aggregated GSO skbs."""
        records = {}
        for node, interfaces in self.network.data_interfaces.items():
            pid = run_process(
                ["docker", "inspect", "--format", "{{.State.Pid}}", self._container(node)],
                timeout=5,
            ).stdout.strip()
            if not pid.isdigit() or int(pid) <= 0:
                raise CommandError("cannot find the owned node network namespace")
            for interface in interfaces:
                run_process(
                    [
                        "nsenter",
                        "--target",
                        pid,
                        "--net",
                        "ip",
                        "link",
                        "set",
                        "dev",
                        interface,
                        "mtu",
                        str(self.network.mtu),
                    ],
                    timeout=5,
                )
                link = json.loads(
                    run_process(
                        [
                            "nsenter",
                            "--target",
                            pid,
                            "--net",
                            "ip",
                            "-j",
                            "link",
                            "show",
                            "dev",
                            interface,
                        ],
                        timeout=5,
                    ).stdout
                )
                if len(link) != 1 or link[0].get("mtu") != self.network.mtu:
                    raise CommandError("data-plane MTU was not applied")
                prefix = ["nsenter", "--target", pid, "--net", "ethtool"]
                run_process(
                    [*prefix, "-K", interface, "tso", "off", "gso", "off", "gro", "off"], timeout=5
                )
                raw = run_process([*prefix, "-k", interface], timeout=5).stdout
                for feature in (
                    "tcp-segmentation-offload",
                    "generic-segmentation-offload",
                    "generic-receive-offload",
                ):
                    if not re.search(rf"^{feature}: off(?:\s|$)", raw, re.MULTILINE):
                        raise CommandError("data-plane packet offloads are still active")
                records[f"{node}:{interface}"] = raw.strip()
        self.evidence["offloads"] = records
        self.evidence["data_plane_mtu"] = self.network.mtu

    def deploy(self, case: CaseDefinition) -> None:
        if case != self.case:
            raise ValueError("backend was constructed for a different case")
        if platform.system() != "Linux" or os.geteuid() != 0:
            raise CommandError("live validation requires root on a Linux Docker host")
        names = run_process(
            ["docker", "ps", "-a", "--format", "{{.Names}}"], timeout=15
        ).stdout.splitlines()
        existing = set(names) & {self._container(node) for node in self.NODES}
        if existing:
            raise CommandError(f"refusing to reuse an existing lab: {', '.join(sorted(existing))}")
        self.evidence["runtime"] = {
            "containerlab": run_process(["containerlab", "version"], timeout=15).stdout.strip(),
            "docker": run_process(
                ["docker", "version", "--format", "{{.Server.Version}}"], timeout=15
            ).stdout.strip(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        }
        repository = REPOSITORY_ROOT
        revision = run_process(["git", "-C", str(repository), "rev-parse", "HEAD"], check=False)
        self.evidence["runtime"]["git_revision"] = revision.stdout.strip() or None
        status = run_process(["git", "-C", str(repository), "status", "--porcelain"], check=False)
        self.evidence["runtime"]["git_dirty"] = bool(status.stdout)
        self.evidence["source_snapshot"] = snapshot_sources(repository, case, self.network)
        self.evidence["images"] = {}
        for image in sorted(
            {node["image"] for node in self.network.topology["topology"]["nodes"].values()}
        ):
            info = json.loads(
                run_process(["docker", "image", "inspect", image], timeout=15).stdout
            )[0]
            self.evidence["images"][image] = {
                key: info.get(key) for key in ("Id", "RepoDigests", "Architecture")
            }
        self.owned = True
        run_process(["containerlab", "deploy", "--topo", str(case.topology)], timeout=240)
        self._disable_offloads()
        self._configure_endpoint_resources()
        self._wait_for_routing()
        for server in self._service_servers():
            self._docker(
                server,
                "iperf3",
                "-s",
                "-D",
                "--logfile",
                "/tmp/graynet-workload.log",
                "--forceflush",
            )
        time.sleep(1)
        self.evidence["post_deploy"] = self._checks()

    def _wait_for_routing(self) -> None:
        last_error: Exception | None = None
        for _ in range(30):
            try:
                for node in self.network.bgp_peers:
                    self._bgp_summary(node)
                return
            except (CommandError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1)
        raise CommandError(f"BGP did not converge: {last_error}")

    def _service_servers(self) -> tuple[str, ...]:
        return (self.server, self.load_server)

    def _configure_endpoint_resources(self) -> None:
        if resources := self._configure_endpoint(self.server):
            self.evidence["endpoint_resources"] = resources

    def _configure_endpoint(self, server: str) -> dict[str, Any] | None:
        command = ["docker", "update"]
        if self.cpu_limit is not None:
            command += ["--cpus", str(self.cpu_limit)]
        cpu = None
        if self.network.endpoint_cpu_affinity == "single_cpu":
            cpu = min(os.sched_getaffinity(0))
            command += ["--cpuset-cpus", str(cpu)]
        if len(command) == 2:
            return
        run_process([*command, self._container(server)], timeout=15)
        pid = run_process(
            ["docker", "inspect", "--format", "{{.State.Pid}}", self._container(server)],
            timeout=5,
        ).stdout.strip()
        if not pid.isdigit() or int(pid) <= 0:
            raise CommandError("endpoint resource namespace unavailable")
        from graynet.tools.network import _read_cgroup

        cpuset = _read_cgroup(pid, "cpuset.cpus.effective").strip()
        quota = _read_cgroup(pid, "cpu.max").strip()
        if cpu is not None and cpuset != str(cpu):
            raise CommandError("endpoint CPU affinity was not applied")
        fields = quota.split()
        if self.cpu_limit is not None and (
            len(fields) != 2
            or fields[0] == "max"
            or abs(int(fields[0]) / int(fields[1]) - self.cpu_limit) > 1e-6
        ):
            raise CommandError("endpoint CPU quota was not applied")
        return {
            "endpoint": server,
            "cpu_max": quota,
            "cpuset_cpus": cpuset,
            "affinity_policy": self.network.endpoint_cpu_affinity,
        }

    def measure(self, case: CaseDefinition, phase: Phase) -> list[float]:
        self.fault.verify(phase)
        fault_before = self.fault.snapshot()
        before = self._qdisc()
        tx_before = self._host_bytes(self.load_client, "tx")
        rx_before = self._host_bytes(self.load_server, "rx")
        started = time.monotonic()
        samples = []
        records = []
        partial = {"throughput_mbps": samples, "records": records, "fault_before": fault_before}
        self.evidence["phases"][phase] = partial
        for _ in range(case.validation.min_samples):
            try:
                record = self._service_sample(self.duration)
            except Exception as exc:
                partial["measurement_error"] = f"{type(exc).__name__}: {exc}"
                raise
            records.append(record)
            samples.append(record["throughput_mbps"])
        elapsed = time.monotonic() - started
        tx_after = self._host_bytes(self.load_client, "tx")
        rx_after = self._host_bytes(self.load_server, "rx")
        load_tx_mbps = (tx_after - tx_before) * 8 / elapsed / 1_000_000
        load_rx_mbps = (rx_after - rx_before) * 8 / elapsed / 1_000_000
        self.fault.verify(phase)
        expected_rate = (
            self.fault.mean_rate(started, started + elapsed)
            if isinstance(self.fault, ScheduledTraffic)
            else self.rate
        )
        if (
            phase == "incident"
            and self.fault.kind == "cross_traffic"
            and (load_tx_mbps < expected_rate * 0.5 or load_rx_mbps < expected_rate * 0.4)
        ):
            raise CommandError(
                f"cross traffic did not traverse the network: {load_tx_mbps:.2f} Mbps sent, {load_rx_mbps:.2f} Mbps received"
            )
        checks = self._checks()
        if phase == "incident" and self.fault.kind == "link_delay":
            baseline_rtt = self.evidence["phases"]["baseline"]["checks"]["rtt_avg_ms"]
            if (
                baseline_rtt is None
                or checks["rtt_avg_ms"] is None
                or checks["rtt_avg_ms"] - baseline_rtt < self.fault.value * 0.8
            ):
                raise CommandError("the delay case did not produce the expected RTT increase")
            if self.fault.snapshot()["dropped"]:
                raise CommandError("the delay case unintentionally dropped packets")
        after = self._qdisc()
        self.evidence["phases"][phase] = {
            "measurement_seconds": elapsed,
            "throughput_mbps": samples,
            "records": records,
            "checks": checks,
            "qdisc_before": before,
            "qdisc_after": after,
            "qdisc_delta": {key: after[key] - before[key] for key in before},
            "load_tx_mbps": load_tx_mbps,
            "load_rx_mbps": load_rx_mbps,
            "fault_before": fault_before,
            "fault_after": self.fault.snapshot(),
            "timeline": self.fault.timeline() if isinstance(self.fault, ScheduledTraffic) else None,
        }
        if (
            phase == "incident"
            and self.fault.kind == "link_loss"
            and self.evidence["phases"][phase]["fault_after"]["dropped"] <= fault_before["dropped"]
        ):
            raise CommandError("no loss was observed on the impaired interface")
        if (
            phase == "incident"
            and self.fault.kind == "endpoint_overload"
            and sum(
                (record["resources"]["allocation_utilization"] or 0) >= 0.75 for record in records
            )
            / len(records)
            < case.validation.min_affected_fraction
        ):
            raise CommandError("endpoint CPU allocation was not consistently saturated")
        if (
            phase == "incident"
            and self.fault.kind == "cross_traffic"
            and after["overlimits"] <= before["overlimits"]
        ):
            raise CommandError("primary-link queue did not register any shaper overlimits")
        if (
            phase == "incident"
            and self.fault.kind == "cross_traffic"
            and not any(
                (sample["backlog_bytes"] or 0) > 0
                for record in records
                for sample in record["queue_samples"]
            )
        ):
            raise CommandError("no queue backlog was observed during the incident workload")
        return samples

    def inject(self, case: CaseDefinition) -> None:
        self.fault.start()
        self.evidence["injection"] = {**case.fault, "duration_seconds": self.fault_duration}

    def start_agent_window(self, case: CaseDefinition) -> None:
        self._monitor_stop.clear()
        self._monitor_errors = []
        self.evidence["agent_window"] = {"records": [], "checks": []}
        if isinstance(self.fault, ScheduledTraffic) and self.fault.pattern == "gradual":
            # Preview admission and the scored investigation are separate,
            # explicitly recorded epochs. Otherwise the agent would only see
            # the plateau after pre-admission had consumed the entire ramp.
            self.fault.stop()
            healthy = self._service_sample(self.duration)
            import statistics

            baseline = statistics.median(self.evidence["phases"]["baseline"]["throughput_mbps"])
            if (
                abs(healthy["throughput_mbps"] - baseline) / baseline
                > case.validation.recovery_tolerance
            ):
                raise CommandError(
                    "the gradual case did not reset to healthy before the scored epoch"
                )
            self.evidence["agent_window"]["epoch_reset"] = healthy
            self.fault.start()

        def observe() -> None:
            while not self._monitor_stop.is_set():
                try:
                    self.fault.verify("incident")
                    record = self._service_sample(self.duration)
                    self.evidence["agent_window"]["records"].append(record)
                    self.evidence["agent_window"]["checks"].append(self._checks())
                except Exception as exc:
                    self._monitor_errors.append(f"{type(exc).__name__}: {exc}")
                    break

        self._monitor_thread = threading.Thread(target=observe, daemon=True)
        self._monitor_thread.start()

    def finish_agent_window(self, case: CaseDefinition, baseline: tuple[float, ...]):
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=self.duration + 45)
            if self._monitor_thread.is_alive():
                raise CommandError("agent-window monitor did not stop")
        if self._monitor_errors:
            raise CommandError("agent-window integrity failed: " + "; ".join(self._monitor_errors))
        self.fault.verify("incident")
        records = self.evidence["agent_window"]["records"]
        # Very short agent failures still receive a measured case-validity check.
        while len(records) < case.validation.min_samples:
            records.append(self._service_sample(self.duration))
        records.sort(key=lambda record: record["started_monotonic"])
        self.evidence["agent_window"]["final_checks"] = self._checks()
        report = self.validate(
            baseline, [record["throughput_mbps"] for record in records], phase="agent_window"
        )
        from dataclasses import asdict

        self.evidence["agent_window"]["validation"] = asdict(report)
        return report

    def probe_for_agent(self) -> dict[str, Any]:
        try:
            record = self._service_sample(self.duration)
        except Exception as exc:
            self.evidence.setdefault("agent_window", {}).setdefault("probe_errors", []).append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        if "agent_window" in self.evidence:
            record["origin"] = "agent_probe"
            self.evidence["agent_window"]["records"].append(record)
        return {
            "throughput_mbps": record["throughput_mbps"],
            "duration_seconds": self.duration,
            "tcp_mean_rtt_ms": self._tcp_mean_rtt(record),
            "measurement_started_at": record["started_at"],
            "measurement_finished_at": record["finished_at"],
        }

    @staticmethod
    def _tcp_mean_rtt(record) -> float | None:
        streams = record.get("iperf", {}).get("end", {}).get("streams", [])
        value = streams[0].get("sender", {}).get("mean_rtt") if streams else None
        return (
            value / 1000
            if type(value) in (int, float) and math.isfinite(value) and value >= 0
            else None
        )

    def healthy_reference(self) -> dict[str, Any]:
        """Operational history only, never injector selection or oracle status."""
        phase = self.evidence.get("phases", {}).get("baseline", {})
        # Public operational history has a fixed window. Private admission
        # sample counts differ for temporal cases and must not identify them.
        records = phase.get("records", [])[:5]
        if len(records) != 5 or "checks" not in phase:
            raise ValueError("normal workload reference is unavailable")
        queue_samples = [sample for record in records for sample in record.get("queue_samples", [])]
        queue = [sample.get("backlog_bytes") for sample in queue_samples]
        drop_counts = [
            int(match.group(1))
            for sample in queue_samples
            if (match := re.search(r"dropped (\d+)", sample.get("raw", "")))
        ]
        first, last = records[0].get("started_monotonic"), records[-1].get("finished_monotonic")
        span = (
            last - first
            if type(first) in (int, float) and type(last) in (int, float) and last >= first
            else None
        )
        return {
            "workload_duration_seconds": self.duration,
            "reference_flows": len(records),
            "measurement_seconds": span,
            "throughput_mbps": [record["throughput_mbps"] for record in records],
            "tcp_mean_rtt_ms": [self._tcp_mean_rtt(record) for record in records],
            "idle_ping_rtt_ms": phase["checks"].get("rtt_avg_ms"),
            "queue": {
                "node": self.queue_node,
                "interface": self.queue_interface,
                "max_backlog_bytes": max(
                    (value for value in queue if type(value) is int), default=None
                ),
                "drops_during_reference": drop_counts[-1] - drop_counts[0]
                if drop_counts and drop_counts[-1] >= drop_counts[0]
                else None,
            },
            "context": "Normal observations under the same standard workload. Loaded TCP RTT is not idle ping RTT; queueing and some drops can occur in normal runs.",
        }

    def validate(self, baseline, incident, recovery=None, *, phase="incident"):
        if self.case.validation.pattern != "sustained":
            records = (
                self.evidence["agent_window"]["records"]
                if phase == "agent_window"
                else self.evidence["phases"]["incident"]["records"]
            )
            timeline = (
                self.fault.timeline()
                if phase == "agent_window"
                else self.evidence["phases"]["incident"]["timeline"]
            )
            return validate_temporal_case(
                self.case.validation, baseline, records, timeline, recovery
            )
        return validate_sustained_case(self.case.validation, baseline, incident, recovery)

    def recover(self, case: CaseDefinition) -> None:
        if not self.owned:
            return
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=self.duration + 45)
            if self._monitor_thread.is_alive():
                raise CommandError("cannot recover while the observer is still active")
        self.fault.stop()

    def teardown(self, case: CaseDefinition) -> None:
        self._monitor_stop.set()
        if self.owned:
            stop_error = None
            try:
                try:
                    self.fault.stop()
                except Exception as exc:
                    stop_error = exc
                self._collect_workload_logs()
                run_process(
                    ["containerlab", "destroy", "--topo", str(case.topology), "--cleanup"],
                    timeout=120,
                )
            finally:
                self.owned = False
            if stop_error is not None:
                raise CommandError(f"fault cleanup failed before teardown: {stop_error}")

    def _collect_workload_logs(self) -> None:
        logs = {}
        for node in self._service_servers():
            try:
                result = run_process(
                    ["docker", "exec", self._container(node), "cat", "/tmp/graynet-workload.log"],
                    timeout=5,
                    check=False,
                )
                logs[node] = {
                    "stdout": result.stdout[-64000:],
                    "stderr": result.stderr[-2000:],
                    "exit_code": result.returncode,
                }
            except Exception as exc:
                logs[node] = {"error": f"{type(exc).__name__}: {exc}"}
        self.evidence["workload_server_logs"] = logs


def create_backend(case: CaseDefinition) -> ContainerlabBackend:
    """Select the workload implementation for the case's network profile."""
    if case.partial_path is not None:
        from graynet.environments.partial_path import PartialPathBackend

        return PartialPathBackend(case)
    return ContainerlabBackend(case)
