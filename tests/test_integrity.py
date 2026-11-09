from dataclasses import replace
from pathlib import Path

from graynet.agents.base import Diagnosis
from graynet.environments.network import load_network
from graynet.evaluation.integrity import (
    audit_case,
    audit_evidence,
    audit_observations,
    environment_fingerprint,
    runtime_environment_fingerprint,
)
from graynet.scenarios.definitions import load_case

ROOT = Path(__file__).parents[1]
CASE = load_case(ROOT / "cases/link_loss")
NETWORK = load_network(CASE)


def test_static_case_audit_rejects_answer_leaks_and_impossible_labels():
    assert audit_case(CASE, NETWORK)["passed"]
    for public in [
        CASE.incident + CASE.case_id,
        CASE.incident + CASE.title,
        CASE.incident + " ground_truth:",
    ]:
        assert not audit_case(replace(CASE, incident=public), NETWORK)["passed"]
    truth = replace(CASE.ground_truth, component="not-managed", component_aliases=())
    assert not audit_case(replace(CASE, ground_truth=truth), NETWORK)["passed"]


def test_matched_cases_share_fingerprint_but_operational_changes_do_not():
    healthy = load_case(ROOT / "cases/healthy_control")
    assert environment_fingerprint(CASE, NETWORK) == environment_fingerprint(
        healthy, load_network(healthy)
    )
    changed = replace(CASE, workload={**CASE.workload, "duration_seconds": 8})
    assert environment_fingerprint(changed, NETWORK) != environment_fingerprint(CASE, NETWORK)
    report_changed = replace(CASE, incident="A different reported symptom.")
    assert environment_fingerprint(report_changed, NETWORK) != environment_fingerprint(
        CASE, NETWORK
    )


def test_evidence_ids_must_reference_real_network_observations():
    trace = [
        {"observation_id": "obs-0001", "tool": "throughput_probe", "status": "completed"},
        {"observation_id": "obs-0002", "tool": "network_inventory", "status": "completed"},
        {"observation_id": "obs-0003", "tool": "run_command", "status": "error"},
        {
            "observation_id": "obs-0004",
            "tool": "run_command",
            "status": "completed",
            "result": {"exit_code": 1},
        },
    ]
    answer = Diagnosis(True, "r2:eth3", "packet_loss", "Loss is observed.", ("obs-0001",))
    assert audit_evidence(answer, trace)["passed"]
    for refs in [
        (),
        ("obs-0002",),
        ("obs-0003",),
        ("obs-0004",),
        ("obs-9999",),
        ("throughput was low",),
        ("obs-0001", "obs-0001"),
    ]:
        assert not audit_evidence(replace(answer, evidence=refs), trace)["passed"]


def test_visibility_audit_scans_returned_observations_not_command_names():
    trace = [
        {
            "observation_id": "obs-0001",
            "tool": "run_command",
            "status": "completed",
            "arguments": {"argv": ["tc", "-s", "qdisc", "show", "dev", "eth3"]},
            "result": {"stdout": '{"queues":[{"dropped":2}]}'},
        }
    ]
    assert audit_observations(trace)["passed"]
    trace.append(
        {
            "observation_id": "obs-0002",
            "tool": "run_command",
            "status": "completed",
            "result": {"stdout": '[{"ifname":"eth3","qdisc":"netem"}]'},
        }
    )
    report = audit_observations(trace)
    assert not report["passed"]
    assert report["findings"] == [{"observation_id": "obs-0002", "marker": "emulator_queue"}]


def test_config_fingerprint_does_not_collapse_equal_basenames(tmp_path):
    files = []
    for node in ("r1", "r2"):
        path = tmp_path / node / "frr.conf"
        path.parent.mkdir()
        path.write_text(node)
        files.append(path)
    network = replace(NETWORK, bound_files=tuple(files))
    original = environment_fingerprint(CASE, network)
    files[0].write_text("changed-first-router")
    assert environment_fingerprint(CASE, network) != original


def test_actual_image_and_endpoint_resource_changes_separate_runtime_identity():
    original = {
        "images": {"host": {"Id": "first"}},
        "endpoint_resources": {"cpuset_cpus": "0"},
        "runtime": {"kernel": "linux"},
    }
    first = runtime_environment_fingerprint(original)
    assert (
        runtime_environment_fingerprint({**original, "images": {"host": {"Id": "second"}}}) != first
    )
    assert (
        runtime_environment_fingerprint({**original, "endpoint_resources": {"cpuset_cpus": "1"}})
        != first
    )


def test_history_alone_does_not_count_as_current_diagnostic_evidence():
    trace = [
        {"observation_id": "obs-0001", "tool": "healthy_reference", "status": "completed"},
        {"observation_id": "obs-0002", "tool": "throughput_probe", "status": "completed"},
    ]
    answer = Diagnosis(False, None, None, "No change from normal.", ("obs-0001",))
    assert not audit_evidence(answer, trace)["passed"]
    assert audit_evidence(replace(answer, evidence=("obs-0001", "obs-0002")), trace)["passed"]
