import json
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from graynet.agents.codex import CodexAgentError, diagnosis_schema, parse_diagnosis
from graynet.environments.network import DiagnosticPolicy, load_network
from graynet.scenarios.definitions import load_case
from graynet.tools.broker import ToolBroker
from graynet.tools.network import NetworkTools, validate_command

CASE = load_case(Path(__file__).parents[1] / "cases/primary_congestion")


def test_policy_comes_from_topology_not_ground_truth():
    network = load_network(CASE)
    changed = replace(
        CASE, ground_truth=replace(CASE.ground_truth, component="secret", mechanism="delay")
    )
    assert load_network(changed).policy == network.policy
    assert network.policy.interfaces["r2"] == ("eth1", "eth2", "eth3", "eth4")
    assert "load-client" not in network.policy.components
    assert network.lab_name == "graynet-primary-congestion"


def test_backend_uses_renamed_lab_without_code_changes(tmp_path):
    topology = yaml.safe_load(CASE.topology.read_text())
    topology["name"] = "renamed-lab"
    for node in ("r1", "r2"):
        topology["topology"]["nodes"][node]["binds"] = [
            str(CASE.topology.parent / bind.split(":")[0]) + ":" + ":".join(bind.split(":")[1:])
            for bind in topology["topology"]["nodes"][node]["binds"]
        ]
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(topology))
    from graynet.environments.containerlab import ContainerlabBackend

    backend = ContainerlabBackend(replace(CASE, directory=tmp_path, topology=path))
    assert backend._container("user-server") == "clab-renamed-lab-user-server"


def test_custom_policy_controls_commands_tools_and_answer_schema():
    policy = DiagnosticPolicy(
        {"edge": ("eth7",), "client": ("eth1",)},
        frozenset({"edge"}),
        frozenset({"192.0.2.10"}),
        "client",
        "192.0.2.10",
    )
    validate_command("edge", ["tc", "-s", "qdisc", "show", "dev", "eth7"], policy)
    for node, args in [
        ("r1", ["ip", "-j", "route", "show"]),
        ("edge", ["ip", "route", "get", "10.2.0.2"]),
        ("edge", ["tc", "-s", "qdisc", "show", "dev", "eth0"]),
    ]:
        with pytest.raises(ValueError):
            validate_command(node, args, policy)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, '{"end":{"sum_received":{"bits_per_second":12000000}}}', ""
        )

    tools = NetworkTools(
        "context", lab_name="new", workload_lock=threading.Lock(), executor=execute, policy=policy
    )
    assert tools.call("throughput_probe", {})["throughput_mbps"] == 12
    assert commands[0][2] == "clab-new-client" and "192.0.2.10" in commands[0]
    definitions = ToolBroker(tools, max_calls=3, timeout_seconds=10).dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert definitions["result"]["tools"][1]["inputSchema"]["properties"]["node"]["enum"] == [
        "edge",
        "client",
    ]
    schema = diagnosis_schema(policy.components)
    answer = {
        "detected": True,
        "component": "edge:eth7",
        "mechanism": "packet_loss",
        "explanation": "Loss observed",
        "evidence": ["drops increased"],
    }
    assert parse_diagnosis(json.dumps(answer), schema).component == "edge:eth7"
    with pytest.raises(CodexAgentError):
        parse_diagnosis(json.dumps({**answer, "component": "r1:eth3"}), schema)


@pytest.mark.parametrize("mutation", ["duplicate", "management", "unknown"])
def test_rejects_invalid_topology_endpoints(tmp_path, mutation):
    topology = yaml.safe_load(CASE.topology.read_text())
    if mutation == "duplicate":
        topology["topology"]["links"].append(topology["topology"]["links"][0])
    else:
        topology["topology"]["links"][0]["endpoints"][0] = (
            "user-client:eth0" if mutation == "management" else "unknown:eth1"
        )
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(topology))
    with pytest.raises(ValueError, match="link endpoint"):
        load_network(replace(CASE, topology=path))
