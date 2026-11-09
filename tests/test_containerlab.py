import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from graynet.environments.containerlab import ContainerlabBackend, snapshot_sources
from graynet.runtime.commands import CommandError
from graynet.scenarios.definitions import load_case

CASE = load_case(Path(__file__).parent.parent / "cases" / "primary_congestion")


def test_first_backend_accepts_only_its_case_layout() -> None:
    backend = ContainerlabBackend(CASE)
    assert backend.duration == 4
    with pytest.raises(ValueError, match="only supports"):
        ContainerlabBackend(replace(CASE, fault={**CASE.fault, "server": "user-server"}))
    with pytest.raises(ValueError, match="fault duration"):
        ContainerlabBackend(replace(CASE, fault={**CASE.fault, "duration_seconds": 20}))


def test_parses_queue_and_cross_traffic_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ContainerlabBackend(CASE)

    def fake_docker(node: str, *args: str, **kwargs: object) -> str:
        if node == "r1":
            return "qdisc tbf 8004: root rate 40Mbit\n Sent 100 bytes 1 pkt (dropped 7, overlimits 42 requeues 0)\n"
        return '[{"stats64":{"tx":{"bytes":123456},"rx":{"bytes":654321}}}]'

    monkeypatch.setattr(backend, "_docker", fake_docker)
    assert backend._qdisc() == {"dropped": 7, "overlimits": 42}
    assert backend._host_bytes("load-client", "tx") == 123456
    assert backend._host_bytes("load-server", "rx") == 654321


def test_rejects_missing_queue_or_bad_counter_output(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ContainerlabBackend(CASE)
    monkeypatch.setattr(backend, "_docker", lambda *args, **kwargs: "qdisc noqueue 0: root")
    with pytest.raises(CommandError, match="shaper is absent"):
        backend._qdisc()
    with pytest.raises(CommandError, match="interface counters"):
        backend._host_bytes("load-client", "tx")


def test_healthy_control_reuses_the_actual_frr_configuration():
    control = load_case(Path(__file__).parent.parent / "cases" / "healthy_control")
    backend = ContainerlabBackend(control)
    assert backend.is_control and backend.rate == 0
    assert control.environment == CASE.environment


def test_agent_deadline_extends_the_sustained_fault_window():
    backend = ContainerlabBackend(CASE)
    backend.reserve_agent_window(300)
    assert backend.fault_duration > 300 + CASE.validation.min_samples * backend.duration


def test_startup_checks_profile_routers_not_fixed_names(monkeypatch):
    backend = ContainerlabBackend(CASE)
    backend.network = replace(
        backend.network,
        bgp_peers={"edge-a": ("192.0.2.1",), "edge-b": ("192.0.2.2",), "edge-c": ("192.0.2.3",)},
    )
    seen = []
    monkeypatch.setattr(backend, "_bgp_summary", lambda node: seen.append(node))
    backend._wait_for_routing()
    assert seen == ["edge-a", "edge-b", "edge-c"]


@pytest.mark.parametrize("actual_cpuset", ["2", "2-4"])
def test_endpoint_affinity_is_constant_and_verified(monkeypatch, actual_cpuset):
    backend = ContainerlabBackend(CASE)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, "123\n" if command[1] == "inspect" else "", ""
        )

    monkeypatch.setattr("graynet.environments.containerlab.run_process", execute)
    monkeypatch.setattr(
        "graynet.environments.containerlab.os.sched_getaffinity", lambda pid: {2, 4}, raising=False
    )
    monkeypatch.setattr(
        "graynet.tools.network._read_cgroup",
        lambda pid, name: actual_cpuset if name == "cpuset.cpus.effective" else "10000 100000",
    )
    if actual_cpuset == "2":
        backend._configure_endpoint_resources()
        assert commands[0] == [
            "docker",
            "update",
            "--cpus",
            "0.1",
            "--cpuset-cpus",
            "2",
            backend._container(backend.server),
        ]
        assert backend.evidence["endpoint_resources"]["cpuset_cpus"] == "2"
    else:
        with pytest.raises(CommandError, match="affinity was not applied"):
            backend._configure_endpoint_resources()


@pytest.mark.parametrize(
    "incident_source", ["environments/shared/incident.md", "cases/sample/incident.md"]
)
def test_shared_topology_and_nested_runtime_sources_are_snapshotted_not_adjacent_secrets(
    tmp_path, incident_source
):
    paths = [
        "graynet/artifacts/atif/model.py",
        "docker/codex/Dockerfile",
        "environments/shared/topology.yml",
        "environments/shared/network.yaml",
        "cases/sample/scenario.yaml",
        incident_source,
        "cases/sample/operator-view.md",
        "cases/sample/auth.json",
        "cases/sample/__pycache__/cache.pyc",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "environment: operator-view.md\n" if name.endswith("scenario.yaml") else "fixture"
        )
    case = replace(
        CASE,
        directory=tmp_path / "cases/sample",
        topology=tmp_path / "environments/shared/topology.yml",
        incident_path=tmp_path / incident_source,
        environment_path=tmp_path / "cases/sample/operator-view.md",
    )
    sources = snapshot_sources(
        tmp_path,
        case,
        SimpleNamespace(path=tmp_path / "environments/shared/network.yaml", bound_files=()),
    )
    assert (
        "environments/shared/topology.yml" in sources
        and "graynet/artifacts/atif/model.py" in sources
    )
    assert "cases/sample/operator-view.md" in sources
    assert sources[incident_source]["content"] == "fixture"
    assert not any("auth.json" in key or "__pycache__" in key for key in sources)


def test_failed_agent_probe_retains_private_diagnostics(monkeypatch):
    backend = ContainerlabBackend(CASE)

    def fail(duration):
        raise CommandError("private measurement failure")

    monkeypatch.setattr(backend, "_service_sample", fail)
    with pytest.raises(CommandError):
        backend.probe_for_agent()
    assert (
        backend.evidence["agent_window"]["probe_errors"][0]["error"]
        == "CommandError: private measurement failure"
    )


def test_phase_measurement_failure_keeps_completed_samples(monkeypatch):
    backend = ContainerlabBackend(CASE)
    monkeypatch.setattr(backend.fault, "verify", lambda phase: None)
    monkeypatch.setattr(backend.fault, "snapshot", lambda: {})
    monkeypatch.setattr(backend, "_qdisc", lambda: {"dropped": 0, "overlimits": 0})
    monkeypatch.setattr(backend, "_host_bytes", lambda *args: 0)
    calls = []

    def sample(duration):
        calls.append(duration)
        if len(calls) > 1:
            raise CommandError("second measurement failed")
        return {"throughput_mbps": 38}

    monkeypatch.setattr(backend, "_service_sample", sample)
    with pytest.raises(CommandError):
        backend.measure(CASE, "baseline")
    phase = backend.evidence["phases"]["baseline"]
    assert phase["throughput_mbps"] == [38] and phase["records"] == [{"throughput_mbps": 38}]
    assert "second measurement failed" in phase["measurement_error"]


def test_normal_reference_distinguishes_idle_and_loaded_rtt_without_answer_key():
    from graynet.environments.containerlab import ContainerlabBackend

    record = {
        "throughput_mbps": 38,
        "iperf": {"end": {"streams": [{"sender": {"mean_rtt": 105000}}]}},
        "queue_samples": [{"backlog_bytes": 500000, "raw": "qdisc tbf private"}],
    }
    phase = {
        "records": [
            {**record, "started_monotonic": index * 5, "finished_monotonic": index * 5 + 4}
            for index in range(5)
        ],
        "checks": {"rtt_avg_ms": 10},
        "qdisc_delta": {"dropped": 50},
        "measurement_seconds": 25,
    }
    references = []
    for name in ("healthy_control", "link_loss", "endpoint_overload"):
        case = load_case(Path(__file__).parents[1] / "cases" / name)
        backend = ContainerlabBackend(case)
        backend.evidence["phases"]["baseline"] = phase
        reference = backend.healthy_reference()
        assert reference["tcp_mean_rtt_ms"] == [105] * 5 and reference["idle_ping_rtt_ms"] == 10
        assert reference["queue"]["max_backlog_bytes"] == 500000
        assert reference["reference_flows"] == 5 and reference["measurement_seconds"] == 24
        import json

        assert not any(
            secret in json.dumps(reference)
            for secret in (case.case_id, "qdisc", "ground_truth", "workers", "loss_percent")
        )
        references.append(reference)
    assert references[0] == references[1] == references[2]


def test_public_history_does_not_expose_private_admission_sample_count():
    records = [
        {
            "throughput_mbps": 38,
            "started_monotonic": index * 5,
            "finished_monotonic": index * 5 + 4,
            "queue_samples": [{"backlog_bytes": index * 100, "raw": f"dropped {index * 10}"}],
        }
        for index in range(12)
    ]
    references = []
    for count in (5, 12):
        backend = ContainerlabBackend(CASE)
        backend.evidence["phases"]["baseline"] = {
            "records": records[:count],
            "checks": {"rtt_avg_ms": 10},
            "qdisc_delta": {"dropped": count * 10},
            "measurement_seconds": count * 5,
        }
        references.append(backend.healthy_reference())
    assert references[0] == references[1]
    assert references[0]["reference_flows"] == 5
    assert references[0]["measurement_seconds"] == 24
    assert references[0]["queue"] == {
        "node": "r1",
        "interface": "eth3",
        "max_backlog_bytes": 400,
        "drops_during_reference": 40,
    }


def test_waits_for_next_server_generation_without_retrying_workload(monkeypatch):
    backend = ContainerlabBackend(CASE)
    backend._next_workload_test = 2
    logs = iter(["Server listening on 5201 (test #1)", "Server listening on 5201 (test #2)"])
    sleeps = []
    monkeypatch.setattr(backend, "_read_workload_log", lambda node: next(logs))
    monkeypatch.setattr("graynet.environments.containerlab.time.sleep", sleeps.append)
    ready = backend._wait_for_workload_ready()
    assert ready["expected_test"] == ready["observed_test"] == 2
    assert sleeps == [0.25] and backend._next_workload_test == 2
    assert backend.evidence["workload_readiness"] == [ready]


@pytest.mark.parametrize("log", ["", "Server listening on 5201 (test #1)"])
def test_server_readiness_timeout_is_not_a_silent_probe_retry(monkeypatch, log):
    backend = ContainerlabBackend(CASE)
    backend._next_workload_test = 2
    monkeypatch.setattr(backend, "_read_workload_log", lambda node: log)
    with pytest.raises(CommandError, match="did not become ready"):
        backend._wait_for_workload_ready(timeout_seconds=0)
    assert backend._next_workload_test == 2
    assert backend.evidence["workload_readiness"][0]["expected_test"] == 2


def test_unexpected_server_generation_is_rejected(monkeypatch):
    backend = ContainerlabBackend(CASE)
    monkeypatch.setattr(
        backend, "_read_workload_log", lambda node: "Server listening on 5201 (test #2)"
    )
    with pytest.raises(CommandError, match="another client or server state change"):
        backend._wait_for_workload_ready()


def test_workload_log_read_has_fixed_owned_path_and_bounded_size(monkeypatch, tmp_path):
    backend = ContainerlabBackend(CASE)
    log = tmp_path / "log"
    log.write_bytes(b"x" * 70000 + b"Server listening on 5201 (test #12)")
    real_open = Path.open
    observed = []

    def open_log(path, *args, **kwargs):
        observed.append(str(path))
        return real_open(log, *args, **kwargs)

    monkeypatch.setattr(
        "graynet.environments.containerlab.run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "123\n", ""),
    )
    monkeypatch.setattr(Path, "open", open_log)
    result = backend._read_workload_log(backend.server)
    assert len(result) == 64000 and result.endswith("(test #12)")
    assert observed == ["/proc/123/root/tmp/graynet-workload.log"]
    with pytest.raises(ValueError, match="owned workload server"):
        backend._read_workload_log("r1")


def test_workload_log_rejects_invalid_runtime_pid(monkeypatch):
    backend = ContainerlabBackend(CASE)
    monkeypatch.setattr(
        "graynet.environments.containerlab.run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "../private", ""),
    )
    with pytest.raises(CommandError, match="namespace unavailable"):
        backend._read_workload_log(backend.server)


def test_service_samples_advance_generation_only_after_success(monkeypatch):
    import json

    backend = ContainerlabBackend(CASE)
    seen = []
    monkeypatch.setattr(
        backend,
        "_wait_for_workload_ready",
        lambda: (
            seen.append(("ready", backend._next_workload_test))
            or {"expected_test": backend._next_workload_test}
        ),
    )
    monkeypatch.setattr(backend, "_queue_snapshot", lambda: {})
    monkeypatch.setattr("graynet.tools.network.cpu_statistics", lambda *args, **kwargs: {})

    def flow(*args, **kwargs):
        seen.append(("flow", backend._next_workload_test))
        return json.dumps({"end": {"sum_received": {"bits_per_second": 38000000}}})

    monkeypatch.setattr(backend, "_docker", flow)
    first, second = backend._service_sample(4), backend._service_sample(4)
    assert seen == [("ready", 1), ("flow", 1), ("ready", 2), ("flow", 2)]
    assert first["server_readiness"]["expected_test"] == 1
    assert second["server_readiness"]["expected_test"] == 2

    def fail(*args, **kwargs):
        raise CommandError("control reset")

    monkeypatch.setattr(backend, "_docker", fail)
    with pytest.raises(CommandError, match="control reset"):
        backend._service_sample(4)
    assert backend._next_workload_test == 3
