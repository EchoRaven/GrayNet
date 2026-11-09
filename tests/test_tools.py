import json
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from graynet.tools.broker import ToolBroker
from graynet.tools.network import NetworkTools, validate_command


@pytest.mark.parametrize(
    "node,argv",
    [
        ("r1", ["vtysh", "-c", "show bgp ipv4 unicast summary json"]),
        ("r2", ["vtysh", "-c", "show interface json"]),
        ("r1", ["tc", "-s", "qdisc", "show", "dev", "eth4"]),
        ("r2", ["tc", "-s", "qdisc", "show", "dev", "eth1"]),
        ("user-client", ["ip", "-j", "-s", "link", "show", "dev", "eth1"]),
        ("r1", ["ip", "-j", "-s", "link", "show"]),
        ("r2", ["ip", "route", "get", "10.1.0.2"]),
        ("user-client", ["ping", "-c", "3", "-W", "1", "10.2.0.2"]),
        ("user-server", ["ss", "-tin"]),
    ],
)
def test_generic_read_only_commands(node, argv):
    validate_command(node, argv)


@pytest.mark.parametrize(
    "node,argv",
    [
        ("load-client", ["ps"]),
        ("r1", ["cat", "/etc/frr/frr.conf"]),
        ("r1", ["sh", "-c", "ip route show"]),
        ("r1", ["ip", "route", "flush", "table", "main"]),
        ("r1", ["vtysh", "-c", "show ip route json; configure terminal"]),
        ("r1", ["vtysh", "-c", "show running-config"]),
        ("r1", ["tc", "-s", "qdisc", "show", "dev", "eth0"]),
        ("r2", ["ip", "route", "get", "169.254.169.254"]),
        ("r1", ["ip", "route", "get", "10.2.0.2", ";", "cat", "/answer"]),
        ("user-client", ["iperf3", "-c", "10.2.0.2", "-t", "999"]),
        ("user-client", ["ping", "-c", "99999", "-W", "1", "10.2.0.2"]),
        ("r1", ["ip", "-j", "-s", "link", "show", "dev", "eth3;cat /answer"]),
    ],
)
def test_rejects_shortcuts_mutations_and_unbounded_probes(node, argv):
    with pytest.raises(ValueError):
        validate_command(node, argv)


def make_tools():
    def execute(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, '{"end":{"sum_received":{"bits_per_second":40000000}}}', ""
        )

    return NetworkTools(
        "Public service context", lab_name="test", workload_lock=threading.Lock(), executor=execute
    )


def call_message(name="network_inventory", arguments=None):
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


def test_mcp_protocol_budget_and_errors():
    broker = ToolBroker(make_tools(), max_calls=2, timeout_seconds=10)
    init = broker.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert broker.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert not broker.dispatch(call_message())["result"]["isError"]
    assert not broker.dispatch(call_message("throughput_probe"))["result"]["isError"]
    assert broker.dispatch(call_message())["result"]["isError"]
    assert len(broker.trace) == 3
    assert broker.trace[1]["result"]["throughput_mbps"] == 40


def test_denied_calls_are_traced_and_do_not_execute():
    tools = make_tools()
    tools.executor = lambda *args, **kwargs: pytest.fail("denied call reached Docker")
    broker = ToolBroker(tools, max_calls=5, timeout_seconds=10)
    result = broker.dispatch(call_message("run_command", {"node": "load-client", "argv": ["ps"]}))
    assert result["result"]["isError"]
    assert broker.trace[0]["status"] == "error"
    broker.deadline = time.monotonic() - 1
    assert broker.dispatch(call_message())["result"]["isError"]
    assert "time budget" in broker.trace[1]["error"]


@pytest.mark.parametrize("node", ["r1", "load-client", "../../answer"])
def test_resource_probe_is_not_a_private_file_or_controller_endpoint(node):
    tools = make_tools()
    tools.executor = lambda *args, **kwargs: pytest.fail("denied resource call reached Docker")
    with pytest.raises(ValueError):
        tools.call("resource_probe", {"node": node})


def test_cpu_probe_reports_usage_without_processes_or_injector_names(monkeypatch):
    from graynet.tools.network import cpu_statistics

    values = iter(
        [
            "usage_usec 100\nnr_throttled 0\nthrottled_usec 0\n",
            "usage_usec 50100\nnr_throttled 5\nthrottled_usec 20000\n",
            "10000 100000\n",
        ]
    )
    monkeypatch.setattr("graynet.tools.network.time.sleep", lambda seconds: None)
    times = iter([10.0, 10.5])
    monkeypatch.setattr("graynet.tools.network.time.monotonic", lambda: next(times))
    monkeypatch.setattr("graynet.tools.network._read_cgroup", lambda pid, name: next(values))
    output = cpu_statistics(
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "1234", ""), "owned-endpoint"
    )
    assert output["allocation_utilization"] == 1
    assert output["cpu_usage_cores"] == 0.1 and output["throttled_periods"] == 5
    assert "sha256sum" not in json.dumps(output) and "/sys/" not in json.dumps(output)


def test_every_successful_network_observation_has_a_traceable_id():
    broker = ToolBroker(make_tools(), max_calls=2, timeout_seconds=10)
    response = broker.dispatch(call_message("throughput_probe"))
    output = json.loads(response["result"]["content"][0]["text"])
    assert output["observation_id"] == broker.trace[0]["observation_id"] == "obs-0001"
    assert "observed_at" in output


def test_normal_reference_is_read_only_bounded_and_citable_history():
    tools = make_tools()
    tools.reference = lambda: {"tcp_mean_rtt_ms": [105], "idle_ping_rtt_ms": 10}
    broker = ToolBroker(tools, max_calls=1, timeout_seconds=10)
    result = broker.dispatch(call_message("healthy_reference"))
    output = json.loads(result["result"]["content"][0]["text"])
    assert output["tcp_mean_rtt_ms"] == [105] and output["observation_id"] == "obs-0001"
    assert broker.dispatch(call_message("healthy_reference"))["result"]["isError"]
    with pytest.raises(ValueError):
        tools.call("healthy_reference", {"path": "/private/answer"})


@pytest.mark.parametrize(
    "argv",
    [
        ["ip", "-j", "addr", "show"],
        ["ip", "-j", "-s", "link", "show"],
        ["ip", "-j", "-s", "link", "show", "dev", "eth3"],
    ],
)
def test_all_interface_views_hide_simulator_and_management_artifacts(argv):
    data = [
        {"ifname": "eth0", "address": "private-management", "qdisc": "noqueue"},
        {
            "ifname": "eth3",
            "mtu": 1500,
            "qdisc": "netem",
            "altnames": ["clab-private-case"],
            "link_netnsid": 99,
            "stats64": {"tx": {"bytes": 1000, "dropped": 2}},
        },
    ]
    tools = make_tools()
    tools.executor = lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, json.dumps(data), ""
    )
    output = tools.call("run_command", {"node": "r1", "argv": argv})
    public = json.dumps(output)
    for secret in ("netem", "qdisc", "clab-", "eth0", "private-management", "netnsid"):
        assert secret not in public
    assert json.loads(output["stdout"])[0]["stats64"]["tx"]["dropped"] == 2


def test_frr_interface_view_preserves_operational_state_not_lab_metadata():
    raw = {
        "clab-123": {"mtu": 9500},
        "eth0": {"ipAddresses": [{"address": "172.20.20.4/24"}]},
        "eth3": {
            "mtu": 1500,
            "operationalStatus": "up",
            "interfaceType": "VETH",
            "parentIfindex": 42,
            "qdisc": "netem",
            "ipAddresses": [{"address": "10.12.0.1/30"}],
        },
    }
    tools = make_tools()
    tools.executor = lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, json.dumps(raw), ""
    )
    result = tools.call(
        "run_command", {"node": "r1", "argv": ["vtysh", "-c", "show interface json"]}
    )
    assert json.loads(result["stdout"]) == {
        "eth3": {
            "mtu": 1500,
            "operationalStatus": "up",
            "ipAddresses": [{"address": "10.12.0.1/30"}],
        }
    }


@pytest.mark.parametrize("frr", [False, True])
def test_route_views_exclude_management_paths(frr):
    if frr:
        raw = {
            "172.20.20.0/24": [{"nexthops": [{"interfaceName": "eth0"}]}],
            "0.0.0.0/0": [{"nexthops": [{"interfaceName": "eth0"}]}],
            "10.2.0.0/24": [
                {"protocol": "bgp", "nexthops": [{"interfaceName": "eth3", "ip": "10.12.0.2"}]}
            ],
        }
        argv = ["vtysh", "-c", "show ip route json"]
    else:
        raw = [
            {"dst": "default", "dev": "eth0", "gateway": "172.20.20.1"},
            {"dst": "172.20.20.0/24", "dev": "eth0"},
            {"dst": "10.2.0.0/24", "dev": "eth3", "gateway": "10.12.0.2"},
        ]
        argv = ["ip", "-j", "route", "show"]
    tools = make_tools()
    tools.executor = lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, json.dumps(raw), ""
    )
    result = tools.call("run_command", {"node": "r1", "argv": argv})
    assert "eth0" not in result["stdout"] and "172.20.20" not in result["stdout"]
    assert "eth3" in result["stdout"] and "10.12.0.2" in result["stdout"]


def test_socket_view_preserves_tcp_evidence_only_for_public_endpoints():
    raw = (
        "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
        "ESTAB 0 0 172.20.20.4:1234 172.20.20.1:22\n\t secret-management\n"
        "ESTAB 0 1000 10.1.0.2:4321 10.2.0.2:5201\n\t rtt:212.5 cwnd:10 retrans:0/3\n"
    )
    tools = make_tools()
    tools.executor = lambda command, **kwargs: subprocess.CompletedProcess(command, 0, raw, "")
    result = tools.call("run_command", {"node": "user-client", "argv": ["ss", "-tin"]})
    assert "rtt:212.5" in result["stdout"] and "retrans:0/3" in result["stdout"]
    assert "management" not in result["stdout"] and "172.20.20" not in result["stdout"]


def test_runtime_failure_does_not_expose_privileged_errors():
    tools = make_tools()
    tools.executor = lambda command, **kwargs: subprocess.CompletedProcess(
        command, 1, "clab-secret-case", "/private/evaluator/path"
    )
    result = tools.call("run_command", {"node": "r1", "argv": ["ip", "-j", "addr", "show"]})
    assert result == {"stdout": "", "stderr": "network command failed", "exit_code": 1}


def test_broker_over_real_unix_socket():
    broker = ToolBroker(make_tools(), max_calls=5, timeout_seconds=10)
    # macOS AF_UNIX paths are limited to 104 bytes; pytest's default is longer.
    with tempfile.TemporaryDirectory(prefix="graynet-test-", dir="/tmp") as directory:
        path = Path(directory) / "tools.sock"
        broker.start(path)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(path))
                connection.sendall(json.dumps(call_message()).encode() + b"\n")
                reply = json.loads(connection.makefile("r").readline())
            assert reply["id"] == 3
            assert "Public service context" in reply["result"]["content"][0]["text"]
        finally:
            broker.close()
