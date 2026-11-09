from dataclasses import replace
from pathlib import Path

import pytest

from graynet.environments.containerlab import ContainerlabBackend
from graynet.scenarios.definitions import load_case
from graynet.tools.network import queue_statistics

ROOT = Path(__file__).parents[1]
IMPAIRMENT_CASES = {"link_rate_limit": "capacity_limit_server_access"}


@pytest.mark.parametrize(
    "case_name",
    [
        "primary_congestion",
        "healthy_control",
        "link_loss",
        "link_delay",
        "endpoint_overload",
        "bursty_congestion",
        "gradual_congestion",
    ],
)
def test_all_cases_use_the_same_operational_view_and_tool_policy(case_name):
    case = load_case(ROOT / "cases" / case_name)
    healthy = load_case(ROOT / "cases/healthy_control")
    backend = ContainerlabBackend(case)
    assert case.incident == healthy.incident and case.incident_path == healthy.incident_path
    assert case.environment == healthy.environment
    assert backend.network.policy == ContainerlabBackend(healthy).network.policy
    assert case.workload == healthy.workload


@pytest.mark.parametrize("workers", [0, 65, True])
def test_endpoint_competing_work_is_bounded(workers):
    case = load_case(ROOT / "cases/endpoint_overload")
    with pytest.raises(ValueError, match="workers"):
        ContainerlabBackend(replace(case, fault={**case.fault, "workers": workers}))


def test_endpoint_overload_requires_matched_healthy_cpu_limit():
    case = load_case(ROOT / "cases/endpoint_overload")
    workload = {key: value for key, value in case.workload.items() if key != "cpu_limit_cores"}
    with pytest.raises(ValueError, match="CPU limit"):
        ContainerlabBackend(replace(case, workload=workload))


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("link_loss", "loss_percent", 101),
        ("link_delay", "delay_ms", -1),
        ("link_loss", "seed", True),
        ("link_jitter_server_access", "jitter_ms", 501),
        ("link_rate_limit", "rate_mbps", 0),
        ("mtu_blackhole_server_access", "mtu", 575),
        ("mtu_blackhole_server_access", "icmp_interface", "eth9"),
        ("policer_server_access", "rate_mbps", -1),
        ("bgp_session_down", "peer", "10.9.9.9"),
        ("bgp_session_down", "asn", "65001"),
    ],
)
def test_rejects_bad_impairment_parameters(kind, field, value):
    case = load_case(ROOT / "cases" / IMPAIRMENT_CASES.get(kind, kind))
    with pytest.raises(ValueError):
        ContainerlabBackend(replace(case, fault={**case.fault, field: value}))


def test_burst_loss_percentages_are_bounded():
    case = load_case(ROOT / "cases/link_loss")
    fault = {
        "kind": "link_burst_loss",
        "node": case.fault["node"],
        "interface": case.fault["interface"],
        "enter_percent": 1,
        "exit_percent": 30,
        "duration_seconds": case.fault["duration_seconds"],
    }
    ContainerlabBackend(replace(case, fault=fault))
    with pytest.raises(ValueError, match="exit_percent"):
        ContainerlabBackend(replace(case, fault={**fault, "exit_percent": 101}))


def test_impairment_cannot_replace_the_operational_shaper():
    case = load_case(ROOT / "cases/link_loss")
    with pytest.raises(ValueError, match="shaper"):
        ContainerlabBackend(replace(case, fault={**case.fault, "node": "r1", "interface": "eth3"}))


def test_loss_start_verify_restore_and_idempotent_cleanup(monkeypatch):
    backend = ContainerlabBackend(load_case(ROOT / "cases/link_loss"))
    commands = []
    active = False
    monkeypatch.setattr("graynet.environments.faults.time.sleep", lambda seconds: None)

    def docker(node, *args, **kwargs):
        nonlocal active
        commands.append((node, args))
        if args[:3] == ("tc", "qdisc", "add"):
            active = True
        if args[:3] == ("tc", "qdisc", "del"):
            active = False
        return (
            "qdisc netem 10: root loss 2%\n Sent 100 bytes 5 pkt (dropped 3, overlimits 0 requeues 0)"
            if active
            else "qdisc noqueue 0: root"
        )

    monkeypatch.setattr(backend, "_docker", docker)
    backend.fault.start()
    assert backend.fault.snapshot()["dropped"] == 3
    assert any("seed" in command[1] and "11" in command[1] for command in commands)
    backend.fault.stop()
    before = len(commands)
    backend.fault.stop()
    assert len(commands) == before


def test_queue_telemetry_never_reveals_simulator_parameters():
    stats = "\n Sent 123 bytes 7 pkt (dropped 2, overlimits 4 requeues 0)\n backlog 100b 1p\n"
    loss = queue_statistics("qdisc netem 10: root loss 2% seed 11" + stats)
    shape = queue_statistics("qdisc tbf 1: root rate 40Mbit" + stats)
    assert loss == shape
    for secret in ("netem", "tbf", "2%", "seed", "40Mbit", "10:"):
        assert secret not in loss
    with pytest.raises(ValueError):
        queue_statistics("qdisc netem root loss 2% malformed output")


def test_rejects_unknown_fields_and_unsupported_metrics():
    case = load_case(ROOT / "cases/link_loss")
    with pytest.raises(ValueError, match="unexpected"):
        ContainerlabBackend(replace(case, fault={**case.fault, "shell": "cat /answer"}))
    with pytest.raises(ValueError, match="throughput_mbps"):
        ContainerlabBackend(replace(case, validation=replace(case.validation, metric="latency")))


@pytest.mark.parametrize(
    "pattern",
    [
        {"kind": "bursty", "on_seconds": 0, "off_seconds": 10},
        {"kind": "gradual", "step_seconds": 10, "rates_mbps": [10, 5, 35]},
        {"kind": "gradual", "step_seconds": 10, "rates_mbps": [10, 20]},
        {"kind": "gradual", "step_seconds": 10, "rates_mbps": [10, 20, 30]},
    ],
)
def test_bad_schedules_are_rejected_before_starting_a_thread(pattern):
    case = load_case(ROOT / "cases/primary_congestion")
    with pytest.raises(ValueError):
        ContainerlabBackend(replace(case, fault={**case.fault, "pattern": pattern}))


def test_agent_probe_uses_the_same_workload_and_records_validation_evidence(monkeypatch):
    backend = ContainerlabBackend(load_case(ROOT / "cases/link_loss"))
    backend.evidence["agent_window"] = {"records": []}
    monkeypatch.setattr(
        backend,
        "_service_sample",
        lambda duration: {
            "throughput_mbps": 8,
            "started_monotonic": 1,
            "finished_monotonic": 5,
            "started_at": "2026-10-07T01:00:00Z",
            "finished_at": "2026-10-07T01:00:04Z",
        },
    )
    assert backend.probe_for_agent() == {
        "throughput_mbps": 8,
        "duration_seconds": 4,
        "tcp_mean_rtt_ms": None,
        "measurement_started_at": "2026-10-07T01:00:00Z",
        "measurement_finished_at": "2026-10-07T01:00:04Z",
    }
    assert backend.evidence["agent_window"]["records"][0]["origin"] == "agent_probe"


def test_new_case_profiles_build_their_faults():
    for path in sorted((ROOT / "cases").iterdir()):
        backend = ContainerlabBackend(load_case(path))
        assert backend.fault.kind == load_case(path).fault["kind"]


def test_composite_fault_rejects_routing_and_single_children():
    case = load_case(ROOT / "cases/multi_loss_on_two_links")
    with pytest.raises(ValueError, match="composite"):
        ContainerlabBackend(replace(case, fault={**case.fault, "faults": case.fault["faults"][:1]}))
    bgp = load_case(ROOT / "cases/bgp_session_down").fault
    with pytest.raises(ValueError, match="composite"):
        ContainerlabBackend(replace(case, fault={**case.fault, "faults": [bgp, bgp]}))


def test_bgp_fault_restores_the_session_and_verifies_the_route(monkeypatch):
    backend = ContainerlabBackend(load_case(ROOT / "cases/bgp_session_down"))
    commands = []
    alternate = False
    monkeypatch.setattr("graynet.environments.faults.time.sleep", lambda seconds: None)
    monkeypatch.setattr(backend, "_wait_for_routing", lambda: None)

    def docker(node, *args, **kwargs):
        nonlocal alternate
        commands.append(args)
        if args[0] == "vtysh":
            alternate = any("neighbor" in part and "no " not in part for part in args)
        return "via 10.12.1.2 dev eth4" if alternate else "via 10.12.0.2 dev eth3"

    monkeypatch.setattr(backend, "_docker", docker)
    backend.fault.start()
    assert backend.fault.active and any("neighbor 10.12.0.2 shutdown" in c for c in commands)
    backend.fault.stop()
    assert not backend.fault.active and any("no neighbor 10.12.0.2 shutdown" in c for c in commands)
