import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from graynet.evaluation.codex_model import CodexJudgmentModel
from graynet.evaluation.models import CompletionError
from graynet.runtime.catalog import preflight_model
from graynet.runtime.codex import (
    CodexRuntimeError,
    IsolatedCodexSession,
    SessionError,
    SessionInterrupted,
)


def fake_runtime(
    monkeypatch,
    tmp_path,
    *,
    timeout=False,
    interrupted=False,
    cleanup_failure=False,
    forbidden_tool=False,
):
    commands = []
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens":{"access_token":"private-fixture-secret"}}')

    def execute(args, **kwargs):
        commands.append(args)
        if args[:2] == ["docker", "inspect"]:
            network = args[2].replace("graynet-judge-", "graynet-judge-net-")
            info = {
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
            return subprocess.CompletedProcess(args, 0, json.dumps([info]), "")
        if cleanup_failure and args[:3] == ["docker", "rm", "-f"]:
            raise TimeoutError("container removal stalled")
        return subprocess.CompletedProcess(args, 0, "", "")

    class Process:
        returncode = None

        def __init__(self, args, **kwargs):
            commands.append(args)
            assert "private-fixture-secret" not in " ".join(args)
            mount = next(
                part
                for part in args
                if part.startswith("type=bind,src=") and "dst=/workspace" in part
            )
            workspace = Path(mount.split("src=", 1)[1].split(",", 1)[0])
            config = tomllib.loads((workspace / "config.toml").read_text())
            assert "mcp_servers" not in config
            assert config["features"]["shell_tool"] is False
            self.communicated = False

        def communicate(self, prompt=None, timeout=None):
            if not self.communicated and interrupted:
                self.communicated = True
                raise KeyboardInterrupt
            if not self.communicated and should_timeout:
                self.communicated = True
                raise subprocess.TimeoutExpired("docker", 1)
            self.communicated = True
            self.returncode = 0
            events = [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": '{"judgment":true}'},
                },
                {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
            ]
            if forbidden_tool:
                events.append({"type": "item.completed", "item": {"type": "command_execution"}})
            return "\n".join(json.dumps(event) for event in events), "stderr fixture"

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    should_timeout = timeout
    monkeypatch.setattr("graynet.runtime.codex.run_process", execute)
    monkeypatch.setattr("graynet.runtime.codex.subprocess.Popen", Process)
    return auth, commands


@pytest.mark.parametrize("timeout,cleanup_failure", [(False, False), (True, False), (False, True)])
def test_tool_free_session_retains_partial_output_and_attempts_all_cleanup(
    monkeypatch, tmp_path, timeout, cleanup_failure
):
    auth, commands = fake_runtime(
        monkeypatch, tmp_path, timeout=timeout, cleanup_failure=cleanup_failure
    )
    session = IsolatedCodexSession(
        image="sha256:pinned",
        model="test",
        effort="low",
        auth_file=auth,
        provider_hosts=("chatgpt.com",),
        timeout_seconds=1,
        role="judge",
    )
    if timeout or cleanup_failure:
        with pytest.raises(SessionError) as error:
            session.run("Evaluate the evidence.", {"type": "object"})
        output = error.value.output
        assert output.metadata["status"] == ("error" if cleanup_failure else "timeout")
    else:
        output = session.run("Evaluate the evidence.", {"type": "object"})
        assert output.metadata["status"] == "completed"
    assert "agent_message" in output.stdout and output.stderr == "stderr fixture"
    assert any(command[:3] == ["docker", "network", "rm"] for command in commands)
    assert sum(command[:3] == ["docker", "rm", "-f"] for command in commands) == 2
    assert bool(output.metadata["cleanup_errors"]) == cleanup_failure
    assert auth.read_text() == '{"tokens":{"access_token":"private-fixture-secret"}}'


def test_judge_adapter_rejects_unexpected_tool_execution(monkeypatch, tmp_path):
    auth, _ = fake_runtime(monkeypatch, tmp_path, forbidden_tool=True)
    model = CodexJudgmentModel(model="test", effort="low", auth_file=auth)
    model._identity = {"driver": "codex", "model": "test", "image_id": "sha256:pinned"}
    model.provider_hosts = ("chatgpt.com",)
    with pytest.raises(CompletionError, match="forbidden tool") as error:
        model.complete("Evaluate.", {"type": "object"})
    assert error.value.completion.metadata["usage"]["input_tokens"] == 10


def test_model_identity_requires_preflight(tmp_path):
    model = CodexJudgmentModel(model="test", effort="low", auth_file=tmp_path / "missing")
    with pytest.raises(ValueError, match="preflight"):
        _ = model.identity


def test_user_interruption_retains_model_output_and_cleans_resources(monkeypatch, tmp_path):
    auth, commands = fake_runtime(monkeypatch, tmp_path, interrupted=True)
    session = IsolatedCodexSession(
        image="sha256:pinned",
        model="test",
        effort="low",
        auth_file=auth,
        provider_hosts=("chatgpt.com",),
        timeout_seconds=5,
        role="judge",
    )
    with pytest.raises(SessionInterrupted) as error:
        session.run("Evaluate.", {"type": "object"})
    assert error.value.output.metadata["status"] == "interrupted"
    assert "agent_message" in error.value.output.stdout
    assert any(command[:3] == ["docker", "network", "rm"] for command in commands)
    assert sum(command[:3] == ["docker", "rm", "-f"] for command in commands) == 2


@pytest.mark.parametrize(
    "model,effort",
    [(None, None), ("available", "low"), ("unknown", "low"), ("available", "unsupported")],
)
def test_shared_preflight_resolves_defaults_and_rejects_unsupported_models(
    monkeypatch, tmp_path, model, effort
):
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens":{"access_token":"private-preflight-token"}}')
    commands = []

    def execute(args, **kwargs):
        commands.append(args)
        assert "private-preflight-token" not in " ".join(args)
        if args[:3] == ["docker", "image", "inspect"]:
            raw = [{"Id": "sha256:actual-image"}]
        elif args[-1] == "--list-models":
            assert "sha256:actual-image" in args and "mutable-tag" not in args
            raw = {
                "data": [
                    {
                        "model": "available",
                        "isDefault": True,
                        "defaultReasoningEffort": "low",
                        "supportedReasoningEfforts": [{"reasoningEffort": "low"}],
                    }
                ]
            }
        else:
            return subprocess.CompletedProcess(args, 0, "codex-cli fixture", "")
        return subprocess.CompletedProcess(args, 0, json.dumps(raw), "")

    monkeypatch.setattr("graynet.runtime.catalog.run_process", execute)
    if model == "unknown" or effort == "unsupported":
        with pytest.raises(CodexRuntimeError, match="catalog|reasoning effort"):
            preflight_model("mutable-tag", auth, model, effort)
    else:
        runtime = preflight_model("mutable-tag", auth, model, effort)
        assert runtime.image_id == "sha256:actual-image" and runtime.model == "available"
        assert runtime.effort == "low" and runtime.provider_hosts == (
            "chatgpt.com",
            "auth.openai.com",
        )
    assert any(command[:3] == ["docker", "rm", "-f"] for command in commands)


def test_judge_preflight_uses_the_same_model_catalog_as_agents(monkeypatch, tmp_path):
    def reject(image, auth, model, effort):
        raise CodexRuntimeError("requested model is not in the account catalog")

    monkeypatch.setattr("graynet.evaluation.codex_model.preflight_model", reject)
    judge = CodexJudgmentModel(model="unknown", effort="low", auth_file=tmp_path / "auth.json")
    with pytest.raises(CodexRuntimeError, match="catalog"):
        judge.preflight()
    with pytest.raises(ValueError, match="preflight"):
        _ = judge.identity
