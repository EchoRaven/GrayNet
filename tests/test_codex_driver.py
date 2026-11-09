import json
import subprocess
import threading
import tomllib
from pathlib import Path

import pytest

from graynet.agents.codex import (
    CodexAgentError,
    CodexDriver,
    codex_config,
    parse_diagnosis,
    parse_events,
)
from graynet.runtime.codex import SessionInterrupted, verify_isolation
from graynet.runtime.egress_proxy import allowed_destination
from graynet.tools.broker import ToolBroker
from graynet.tools.network import NetworkTools

ANSWER = {
    "detected": True,
    "component": "r1:eth3",
    "mechanism": "congestion",
    "explanation": "The forwarding path is congested.",
    "evidence": ["obs-0001"],
}


def isolation_info(network="private"):
    return {
        "Config": {"User": "1000:1000"},
        "HostConfig": {
            "ReadonlyRootfs": True,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
        },
        "NetworkSettings": {"Networks": {network: {}}},
        "Mounts": [
            {"Destination": path, "RW": False}
            for path in ("/workspace", "/run/graynet", "/run/codex-auth.json")
        ],
    }


def test_isolation_verification_fails_closed():
    assert verify_isolation(isolation_info(), "private")["privileged"] is False
    info = isolation_info()
    info["Mounts"].append({"Destination": "/var/run/docker.sock", "RW": False})
    with pytest.raises(CodexAgentError, match="isolation policy"):
        verify_isolation(info, "private")
    info = isolation_info()
    info["HostConfig"]["Privileged"] = True
    with pytest.raises(CodexAgentError):
        verify_isolation(info, "private")


def test_structured_diagnosis_and_healthy_answer():
    assert parse_diagnosis(json.dumps(ANSWER)).component == "r1:eth3"
    healthy = {**ANSWER, "detected": False, "component": None, "mechanism": None}
    assert not parse_diagnosis(json.dumps(healthy)).detected


@pytest.mark.parametrize(
    "change",
    [
        {"detected": "yes"},
        {"component": None},
        {"mechanism": ""},
        {"explanation": ""},
        {"evidence": []},
        {"extra": 1},
    ],
)
def test_malformed_submission_is_agent_error(change):
    with pytest.raises(CodexAgentError):
        parse_diagnosis(json.dumps({**ANSWER, **change}))


def test_usage_includes_cache_without_double_counting():
    events, usage = parse_events(
        "not-json\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 70, "output_tokens": 12},
            }
        )
    )
    assert len(events) == 1
    assert usage == {
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "output_tokens": 12,
        "reasoning_output_tokens": None,
    }


def test_incomplete_multi_turn_usage_is_unknown_not_a_complete_sum():
    raw = "\n".join(
        json.dumps({"type": "turn.completed", "usage": value})
        for value in [{"input_tokens": 10, "output_tokens": 2}, {"output_tokens": 3}]
    )
    _, usage = parse_events(raw)
    assert usage["input_tokens"] is None and usage["output_tokens"] == 5


def test_config_disables_host_access_history_and_external_tools():
    config = tomllib.loads(codex_config("test-model", "low"))
    assert config["model"] == "test-model"
    assert config["features"]["shell_tool"] is False
    assert config["features"]["multi_agent"] is False
    assert config["features"]["plugins"] is False
    assert config["web_search"] == "disabled"
    assert config["mcp_servers"]["network"]["required"] is True
    assert list(config["mcp_servers"]) == ["network"]


def test_provider_proxy_rejects_other_hosts_ports_and_suffixes():
    allowed = {"chatgpt.com", "auth.openai.com"}
    assert allowed_destination("chatgpt.com", 443, allowed)
    assert not allowed_destination("chatgpt.com.evil.test", 443, allowed)
    assert not allowed_destination("chatgpt.com", 80, allowed)
    assert not allowed_destination("169.254.169.254", 443, allowed)
    assert not allowed_destination("github.com", 443, allowed)


@pytest.mark.parametrize("timeout", [False, True, "interrupt"])
def test_driver_runs_or_times_out_and_cleans_all_resources(monkeypatch, tmp_path: Path, timeout):
    commands = []
    brokers = []
    auth = tmp_path / "user-auth.json"
    auth.write_text('{"tokens":{"access_token":"private-test-token"}}')
    tools = NetworkTools(
        "Only public network context",
        lab_name="test",
        workload_lock=threading.Lock(),
        executor=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, '[{"dst":"10.2.0.0/24","dev":"eth3","gateway":"10.12.0.2"}]', ""
        ),
    )
    driver = CodexDriver(
        tools,
        auth_file=auth,
        artifacts=tmp_path / "artifacts",
        timeout_seconds=1,
        runtime_context=lambda: {"environment_runtime_sha256": "actual-live-environment"},
    )
    driver.metadata["image_id"] = "sha256:test"
    driver.provider_hosts = ("chatgpt.com", "auth.openai.com")

    def command(args, **kwargs):
        commands.append(args)
        if args[:2] == ["docker", "inspect"]:
            network = args[2].replace("graynet-agent-", "graynet-agent-net-")
            return subprocess.CompletedProcess(args, 0, json.dumps([isolation_info(network)]), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("graynet.runtime.codex.run_process", command)
    monkeypatch.setattr(ToolBroker, "start", lambda self, path: brokers.append(self))

    class Process:
        returncode = None
        calls = 0

        def __init__(self, command, **kwargs):
            commands.append(command)
            joined = " ".join(command)
            assert "private-test-token" not in joined
            assert "docker.sock" not in joined
            assert "--read-only" in command and "--cap-drop" in command
            assert "--network" in command and "graynet-agent-net-" in joined
            for mount in (
                command[index + 1] for index, part in enumerate(command) if part == "--mount"
            ):
                assert mount.endswith(",readonly")
            assert not any("/GrayNet" in part for part in command)

        def communicate(self, instruction=None, timeout=None):
            self.calls += 1
            if self.calls == 1 and globals_timeout == "interrupt":
                raise KeyboardInterrupt
            if self.calls == 1 and globals_timeout:
                raise subprocess.TimeoutExpired("docker", 1)
            self.returncode = 0
            brokers[0].dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "run_command",
                        "arguments": {"node": "r1", "argv": ["ip", "-j", "route", "show"]},
                    },
                }
            )
            return "\n".join(
                [
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": json.dumps(ANSWER)},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 50,
                                "output_tokens": 20,
                            },
                        }
                    ),
                ]
            ), ""

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    globals_timeout = timeout
    monkeypatch.setattr("graynet.runtime.codex.subprocess.Popen", Process)
    if timeout == "interrupt":
        with pytest.raises(SessionInterrupted):
            driver.diagnose("Investigate poor throughput")
        assert driver.metadata["status"] == "interrupted"
    elif timeout:
        with pytest.raises(CodexAgentError, match="deadline"):
            driver.diagnose("Investigate poor throughput")
        assert driver.metadata["status"] == "timeout"
    else:
        assert driver.diagnose("Investigate poor throughput").detected
        assert driver.metadata["status"] == "completed"
    assert any(command[:3] == ["docker", "network", "rm"] for command in commands)
    assert sum(command[:3] == ["docker", "rm", "-f"] for command in commands) >= 2
    assert driver.metadata["usage"]["input_tokens"] == 100
    assert driver.metadata["environment_runtime_sha256"] == "actual-live-environment"
    assert "actual-live-environment" not in (driver.artifacts / "input/prompt.txt").read_text()
    assert (driver.artifacts / "input/incident.md").read_text() == "Investigate poor throughput"
    assert (
        "Incident report:\nInvestigate poor throughput"
        in (driver.artifacts / "input/prompt.txt").read_text()
    )
    assert auth.read_text() == '{"tokens":{"access_token":"private-test-token"}}'
    assert not list(driver.artifacts.rglob("auth.json"))
    assert "private-test-token" not in "".join(
        path.read_text() for path in driver.artifacts.rglob("*") if path.is_file()
    )
    assert (driver.artifacts / "input/environment.md").read_text() == tools.inventory
    assert (driver.artifacts / "logs/events.jsonl").is_file()
    trajectory = json.loads((driver.artifacts / "trajectory.json").read_text())
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["extra"]["graynet"]["status"] == (
        "interrupted" if timeout == "interrupt" else "timeout" if timeout else "completed"
    )
    assert any(step.get("tool_calls") for step in trajectory["steps"])
    assert "Agent trajectory" in (driver.artifacts / "trajectory.md").read_text()
