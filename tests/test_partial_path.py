import threading
from collections import defaultdict
from dataclasses import replace
from subprocess import CompletedProcess

import pytest

from graynet.agents.flow_baseline import BASELINE_CALLS, FlowProbeBaseline
from graynet.environments.partial_path import PartialPathBackend
from graynet.evaluation.flow_diagnosis import score_flow_diagnosis
from graynet.evaluation.integrity import audit_observations
from graynet.evaluation.validation import validate_partial_paths
from graynet.execution.probe_experiment import plan_probe_experiment, run_probe_experiment
from graynet.paths import REPOSITORY_ROOT
from graynet.scenarios.definitions import load_case
from graynet.tools.network import NetworkTools


def scenario(condition):
    return load_case(REPOSITORY_ROOT / "cases" / f"partial_path_{condition}")


def test_admission_distinguishes_mild_slowdown_from_service_violation():
    case = scenario("alpha_mild")
    baseline = {"alpha": [40] * 5, "beta": [39] * 5}
    incident = {"alpha": [36, 37, 36, 35, 36], "beta": [39] * 5}
    report = validate_partial_paths(
        case.validation, case.partial_path, baseline, incident, baseline
    )
    assert all(row.result.valid for row in report.values())
    assert report["alpha"].result.relative_change == pytest.approx(0.1)
    assert report["alpha"].service_min_mbps == pytest.approx(30)
    assert report["alpha"].service_violation is False
    assert report["alpha"].service_violation_fraction == 0


def test_admission_rejects_spillover_and_failed_recovery():
    case = scenario("beta_near")
    baseline = {"alpha": [40] * 5, "beta": [40] * 5}
    incident = {"alpha": [35] * 5, "beta": [28] * 5}
    recovery = {"alpha": [40] * 5, "beta": [35] * 5}
    report = validate_partial_paths(
        case.validation, case.partial_path, baseline, incident, recovery
    )
    assert "unaffected_flow_not_stable" in report["alpha"].result.reasons
    assert report["beta"].service_violation is True
    assert "recovery_not_healthy" in report["beta"].result.reasons


def test_admission_requires_repeatable_service_state_near_limit():
    case = scenario("beta_near")
    baseline = {"alpha": [40] * 5, "beta": [40] * 5}
    incident = {"alpha": [40] * 5, "beta": [28, 28, 28, 31, 31]}
    report = validate_partial_paths(case.validation, case.partial_path, baseline, incident)
    assert report["beta"].service_violation is True
    assert report["beta"].service_violation_fraction == pytest.approx(0.6)
    assert "service_state_not_repeatable" in report["beta"].result.reasons


def stub_tools(backend, targets):
    calls = defaultdict(int)

    def probe(name):
        calls[name] += 1
        return {"flow": name, "throughput_mbps": 19 if name == "beta" else 40}

    def executor(argv, **kwargs):
        dropped = sum(calls.values()) * 10 if argv[-1] == "eth4" else 0
        return CompletedProcess(
            argv,
            0,
            f"qdisc netem 8001: root limit 10000 loss 0.4% seed 11\n"
            f" Sent 1000 bytes 20 pkt (dropped {dropped}, overlimits 0 requeues 0)\n",
            "",
        )

    return NetworkTools(
        backend.case.environment,
        lab_name=backend.LAB_NAME,
        workload_lock=threading.Lock(),
        executor=executor,
        policy=backend.network.policy,
        probe=probe,
        reference=lambda: {
            "flows": {
                name: {
                    "throughput_mbps": [40] * 5,
                    "service_min_mbps": 30,
                    "preferred_egress": ":".join(flow.sender),
                }
                for name, flow in backend.flows.items()
            }
        },
        probe_targets=targets,
    )


def test_broker_enforces_target_and_keeps_other_tools_identical():
    backend = PartialPathBackend(scenario("beta_clear"))
    fixed = stub_tools(backend, ("alpha",))
    choice = stub_tools(backend, ("alpha", "beta"))
    with pytest.raises(ValueError, match="not an available"):
        fixed.call("throughput_probe", {"flow": "beta"})
    assert choice.call("throughput_probe", {"flow": "beta"})["flow"] == "beta"
    fixed_tools = fixed.definitions()
    choice_tools = choice.definitions()
    for left, right in zip(fixed_tools, choice_tools, strict=True):
        if left["name"] == "throughput_probe":
            assert left["inputSchema"]["properties"]["flow"]["enum"] == ["alpha"]
            assert right["inputSchema"]["properties"]["flow"]["enum"] == ["alpha", "beta"]
        else:
            assert left == right


def test_scripted_comparison_and_evidence_support():
    backend = PartialPathBackend(scenario("beta_clear"))
    fixed = FlowProbeBaseline(
        stub_tools(backend, ("alpha",)), max_calls=BASELINE_CALLS, timeout_seconds=60
    )
    choice = FlowProbeBaseline(
        stub_tools(backend, ("alpha", "beta")), max_calls=BASELINE_CALLS, timeout_seconds=60
    )
    fixed_answer = fixed.diagnose(backend.case.incident)
    choice_answer = choice.diagnose(backend.case.incident)
    assert len(fixed.trace) == len(choice.trace) == BASELINE_CALLS
    assert audit_observations(fixed.trace)["passed"] and audit_observations(choice.trace)["passed"]
    fixed_score = score_flow_diagnosis(backend.case, backend.flows, fixed_answer, fixed.trace)
    choice_score = score_flow_diagnosis(backend.case, backend.flows, choice_answer, choice.trace)
    assert fixed_score.detection == fixed_score.affected_flow == fixed_score.location == "uncertain"
    assert (
        choice_score.detection == choice_score.affected_flow == choice_score.location == "correct"
    )
    assert all(choice_score.evidence_support.values())
    # A correct-looking location without cited counters has no evidence support.
    reduced = replace(
        choice_answer,
        evidence=tuple(
            row["observation_id"] for row in choice.trace if row["tool"] != "run_command"
        ),
    )
    unsupported = score_flow_diagnosis(backend.case, backend.flows, reduced, choice.trace)
    assert unsupported.location == "correct"
    assert unsupported.evidence_support == {
        "detection": True,
        "affected_flow": True,
        "location": False,
    }


def test_plan_balances_which_path_fixed_probe_observes():
    plan = plan_probe_experiment(REPOSITORY_ROOT / "experiments/partial_path.yaml")
    assert len(plan["pairs"]) == 14
    assert [pair["fixed_probe_intersects_fault"] for pair in plan["pairs"]].count(True) == 6
    assert [pair["fixed_probe_intersects_fault"] for pair in plan["pairs"]].count(False) == 6
    by_case = defaultdict(list)
    for pair in plan["pairs"]:
        by_case[pair["case_id"]].append(pair)
    for pairs in by_case.values():
        first, second = sorted(pairs, key=lambda pair: pair["repetition"])
        assert first["arm_order"] == second["arm_order"][::-1]
        if first["fault_seed"] is not None:
            assert [first["fault_seed"], second["fault_seed"]] == [11, 12]


def test_modified_plan_cannot_execute(tmp_path):
    plan = plan_probe_experiment(REPOSITORY_ROOT / "experiments/partial_path.yaml")
    plan["targets"]["fixed"] = ["beta"]
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="plan hash"):
        run_probe_experiment(plan, destination)
    assert not destination.exists()
