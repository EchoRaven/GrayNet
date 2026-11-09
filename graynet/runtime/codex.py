"""Shared Codex process policy; contains no case, network, or scoring logic."""

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from graynet.runtime.commands import run_process


class CodexRuntimeError(RuntimeError):
    """The CLI runtime failed before it produced a usable completion."""


@dataclass(frozen=True)
class SessionOutput:
    stdout: str
    stderr: str
    proxy_log: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionError(CodexRuntimeError):
    def __init__(self, message: str, output: SessionOutput):
        super().__init__(message)
        self.output = output


class SessionInterrupted(KeyboardInterrupt):
    """Preserve partial output while keeping user interruption distinct from failure."""

    def __init__(self, output: SessionOutput):
        super().__init__("model session interrupted")
        self.output = output


class SessionBroker(Protocol):
    deadline: float

    def start(self, path: Path) -> None: ...

    def close(self) -> None: ...


def parse_events(stdout: str) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    usage_events = [
        event["usage"]
        for event in events
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict)
    ]
    usage = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        values = [
            value[key] for value in usage_events if type(value.get(key)) is int and value[key] >= 0
        ]
        usage[key] = sum(values) if usage_events and len(values) == len(usage_events) else None
    return events, usage


def codex_config(model: str | None, effort: str | None, *, network_tools: bool = True) -> str:
    # A fresh config prevents user hooks, MCP servers, plugins, memories, and
    # project history from entering an evaluated episode.
    lines = [
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        'web_search = "disabled"',
        'cli_auth_credentials_store = "file"',
        'history.persistence = "none"',
        "[features]",
        "shell_tool = false",
        "unified_exec = false",
        "multi_agent = false",
        "apps = false",
        "plugins = false",
        "memories = false",
        "hooks = false",
        "shell_snapshot = false",
        "skill_mcp_dependency_install = false",
        "[mcp_servers.network]",
        'command = "python3"',
        'args = ["-I", "/opt/agent/mcp_stdio.py", "/run/graynet/tools.sock"]',
        "required = true",
        "startup_timeout_sec = 10",
        "tool_timeout_sec = 25",
        'default_tools_approval_mode = "auto"',
    ]
    if not network_tools:
        lines = lines[: lines.index("[mcp_servers.network]")]
    overrides = []
    if model:
        overrides.append(f"model = {json.dumps(model)}")
    if effort:
        overrides.append(f"model_reasoning_effort = {json.dumps(effort)}")
    return "\n".join(overrides + lines) + "\n"


def verify_isolation(info: dict[str, Any], network: str) -> dict[str, Any]:
    config, host = info["Config"], info["HostConfig"]
    networks = list(info["NetworkSettings"]["Networks"])
    mounts = {mount["Destination"]: mount["RW"] for mount in info["Mounts"]}
    readonly = {"/workspace", "/run/graynet", "/run/codex-auth.json"}
    writable = {"/tmp", "/scratch", "/home/agent/.codex"}
    valid = (
        config["User"] == "1000:1000"
        and host["ReadonlyRootfs"]
        and not host["Privileged"]
        and "ALL" in (host["CapDrop"] or [])
        and networks == [network]
        and any(option.startswith("no-new-privileges") for option in host["SecurityOpt"] or [])
        and readonly.issubset(mounts)
        and all(not mounts[path] for path in readonly)
        and set(mounts).issubset(readonly | writable)
    )
    if not valid:
        raise CodexRuntimeError("agent container does not satisfy the required isolation policy")
    return {
        "user": config["User"],
        "read_only_root": host["ReadonlyRootfs"],
        "privileged": host["Privileged"],
        "cap_drop": host["CapDrop"],
        "security_options": host["SecurityOpt"],
        "networks": networks,
        "mounts": [{"destination": key, "writable": value} for key, value in mounts.items()],
    }


def isolated_codex_command(
    name: str,
    network: str,
    public: Path,
    broker: Path,
    auth: Path,
    image: str,
    *,
    use_proxy: bool = True,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        network,
        "--user",
        "1000:1000",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "2g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,mode=1777,size=256m",
        "--tmpfs",
        "/scratch:rw,nosuid,nodev,uid=1000,gid=1000,size=64m",
        "--tmpfs",
        "/home/agent/.codex:rw,nosuid,nodev,uid=1000,gid=1000,size=128m",
        "--mount",
        f"type=bind,src={public},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={broker},dst=/run/graynet,readonly",
        "--mount",
        f"type=bind,src={auth},dst=/run/codex-auth.json,readonly",
        *(
            [
                "-e",
                "HTTPS_PROXY=http://provider-proxy:8080",
                "-e",
                "HTTP_PROXY=http://provider-proxy:8080",
                "-e",
                "NO_PROXY=localhost,127.0.0.1,::1",
            ]
            if use_proxy
            else []
        ),
        image,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--output-schema",
        "/workspace/diagnosis.schema.json",
        "--color",
        "never",
        "-",
    ]


class IsolatedCodexSession:
    """One bounded model turn, with optional evaluator-owned tools.

    Agents and judges share resource policy and cleanup. The session never reads
    scenario files, decides an answer, or interprets a judge rubric.
    """

    def __init__(
        self,
        *,
        image: str,
        model: str | None,
        effort: str | None,
        auth_file: Path,
        provider_hosts: tuple[str, ...],
        timeout_seconds: int,
        role: str = "agent",
        broker: SessionBroker | None = None,
    ):
        if (
            role not in ("agent", "judge")
            or type(timeout_seconds) is not int
            or timeout_seconds < 1
        ):
            raise ValueError("session needs an agent or judge role and a positive deadline")
        self.image, self.model, self.effort = image, model, effort
        self.auth_file, self.provider_hosts = auth_file, provider_hosts
        self.timeout_seconds, self.role, self.broker = timeout_seconds, role, broker

    def run(
        self, prompt: str, schema: dict[str, Any], *, files: dict[str, str] | None = None
    ) -> SessionOutput:
        suffix = uuid.uuid4().hex[:12]
        name, network, proxy = (
            f"graynet-{self.role}-{suffix}",
            f"graynet-{self.role}-net-{suffix}",
            f"graynet-{self.role}-provider-{suffix}",
        )
        started = time.monotonic()
        stdout = stderr = proxy_log = ""
        error = None
        interrupted = False
        process = None
        created_network = created_proxy = False
        cleanup_errors: list[str] = []
        metadata: dict[str, Any] = {
            "status": "starting",
            "container_name": name,
            "provider_hosts": self.provider_hosts,
        }

        def cleanup(label, action):
            try:
                result = action()
                if result is not None and result.returncode != 0:
                    # Removing an already auto-removed container is idempotent.
                    if (
                        "No such container" not in result.stderr
                        and "not found" not in result.stderr
                    ):
                        raise RuntimeError(f"command exited with status {result.returncode}")
            except Exception as exc:
                cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        with tempfile.TemporaryDirectory(prefix=f"graynet-{self.role}-") as temporary:
            root = Path(temporary)
            public, endpoint = root / "public", root / "broker"
            public.mkdir(mode=0o755)
            endpoint.mkdir(mode=0o755)
            (public / "config.toml").write_text(
                codex_config(self.model, self.effort, network_tools=self.broker is not None),
                encoding="utf-8",
            )
            (public / "diagnosis.schema.json").write_text(json.dumps(schema), encoding="utf-8")
            for filename, contents in (files or {}).items():
                if Path(filename).name != filename or filename in (
                    "config.toml",
                    "diagnosis.schema.json",
                ):
                    raise ValueError("session inputs must be separate workspace filenames")
                (public / filename).write_text(contents, encoding="utf-8")
            auth = root / "auth.json"
            shutil.copyfile(self.auth_file, auth)
            auth.chmod(0o644)  # Private parent, read-only mount; never copied into results.
            try:
                if self.broker is not None:
                    self.broker.start(endpoint / "tools.sock")
                created_network = True
                run_process(["docker", "network", "create", "--internal", network], timeout=15)
                created_proxy = True
                run_process(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        proxy,
                        "--network",
                        "bridge",
                        "--read-only",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        "64",
                        "--memory",
                        "128m",
                        "--cpus",
                        "0.5",
                        "-e",
                        f"GRAYNET_PROVIDER_HOSTS={','.join(self.provider_hosts)}",
                        "--entrypoint",
                        "python3",
                        self.image,
                        "-I",
                        "/opt/agent/egress_proxy.py",
                    ],
                    timeout=20,
                )
                run_process(
                    ["docker", "network", "connect", "--alias", "provider-proxy", network, proxy],
                    timeout=15,
                )
                process = subprocess.Popen(
                    isolated_codex_command(name, network, public, endpoint, auth, self.image),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                deadline = time.monotonic() + self.timeout_seconds
                if self.broker is not None:
                    self.broker.deadline = deadline
                for _ in range(30):
                    inspection = run_process(["docker", "inspect", name], timeout=5, check=False)
                    if inspection.returncode == 0 and inspection.stdout.strip():
                        metadata["isolation"] = verify_isolation(
                            json.loads(inspection.stdout)[0], network
                        )
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("model startup deadline exceeded")
                    time.sleep(0.1)
                else:
                    raise CodexRuntimeError("could not verify the model container's isolation")
                metadata["status"] = "running"
                stdout, stderr = process.communicate(
                    prompt, timeout=max(0.01, deadline - time.monotonic())
                )
                metadata["exit_code"] = process.returncode
                if process.returncode != 0:
                    raise CodexRuntimeError(f"Codex exited with status {process.returncode}")
                metadata["status"] = "completed"
            except (Exception, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    interrupted = True
                    metadata["status"] = "interrupted"
                elif isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)):
                    metadata["status"] = "timeout"
                    error = f"Codex exceeded its {self.timeout_seconds}-second deadline"
                else:
                    metadata["status"] = "error"
                    error = f"{type(exc).__name__}: {exc}"
            finally:
                # Each action is independent. A failed removal must not skip the
                # remaining cleanup or discard the partial investigation.
                cleanup(
                    "model container",
                    lambda: run_process(["docker", "rm", "-f", name], timeout=15, check=False),
                )
                if process is not None:
                    if process.poll() is None:
                        cleanup("model process", lambda: process.kill())
                    if not stdout:
                        try:
                            stdout, stderr = process.communicate(timeout=15)
                        except Exception as exc:
                            cleanup_errors.append(f"partial output: {type(exc).__name__}: {exc}")
                if self.broker is not None:
                    cleanup("tool broker", self.broker.close)
                if created_proxy:
                    try:
                        logs = run_process(["docker", "logs", proxy], timeout=10, check=False)
                        proxy_log = logs.stdout + logs.stderr
                    except Exception as exc:
                        metadata["proxy_log_error"] = f"{type(exc).__name__}: {exc}"
                    cleanup(
                        "provider proxy",
                        lambda: run_process(["docker", "rm", "-f", proxy], timeout=15, check=False),
                    )
                if created_network:
                    cleanup(
                        "private network",
                        lambda: run_process(
                            ["docker", "network", "rm", network], timeout=15, check=False
                        ),
                    )
        metadata.update(duration_seconds=time.monotonic() - started, cleanup_errors=cleanup_errors)
        output = SessionOutput(stdout, stderr, proxy_log, metadata)
        if interrupted:
            raise SessionInterrupted(output)
        if cleanup_errors:
            error = (error + "; " if error else "") + "model resource cleanup failed"
            metadata["status"] = "error"
        if error is not None:
            raise SessionError(error, output)
        return output
