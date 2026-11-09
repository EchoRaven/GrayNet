"""Resolve credentials, image identity, and account model support before a run."""

import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graynet.runtime.codex import CodexRuntimeError, codex_config, isolated_codex_command
from graynet.runtime.commands import run_process


@dataclass(frozen=True)
class ModelRuntime:
    image_id: str
    cli_version: str
    model: str
    effort: str
    provider_hosts: tuple[str, ...]
    catalog: list[dict[str, Any]]


def preflight_model(
    image: str, auth_file: Path, model: str | None, effort: str | None
) -> ModelRuntime:
    if auth_file.is_symlink() or not auth_file.is_file():
        raise CodexRuntimeError("model runtime needs a regular authentication file")
    auth = json.loads(auth_file.read_text(encoding="utf-8"))
    if not isinstance(auth, dict) or not (auth.get("tokens") or auth.get("OPENAI_API_KEY")):
        raise CodexRuntimeError("model credentials are not supported")
    provider_hosts = (
        ("api.openai.com",) if auth.get("OPENAI_API_KEY") else ("chatgpt.com", "auth.openai.com")
    )
    info = json.loads(run_process(["docker", "image", "inspect", image], timeout=15).stdout)[0]
    image_id = info["Id"]
    version = run_process(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "codex",
            image_id,
            "--version",
        ],
        timeout=20,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="graynet-preflight-") as temporary:
        root = Path(temporary)
        public, empty = root / "public", root / "broker"
        public.mkdir(mode=0o755)
        empty.mkdir(mode=0o755)
        (public / "config.toml").write_text(codex_config(None, None, network_tools=False))
        auth_copy = root / "auth.json"
        shutil.copyfile(auth_file, auth_copy)
        auth_copy.chmod(0o644)
        name = "graynet-preflight-" + uuid.uuid4().hex[:12]
        args = isolated_codex_command(
            name, "bridge", public, empty, auth_copy, image_id, use_proxy=False
        )
        # Model discovery has no model turn, lab, tools, or private case mounts.
        try:
            catalog = json.loads(
                run_process(
                    [*args[: args.index(image_id)], image_id, "--list-models"], timeout=40
                ).stdout
            )
        finally:
            removed = run_process(["docker", "rm", "-f", name], timeout=15, check=False)
            if removed.returncode and "No such container" not in removed.stderr:
                raise CodexRuntimeError("model discovery container cleanup failed")
    if not isinstance(catalog, dict):
        raise CodexRuntimeError("model catalog has an invalid format")
    models = catalog.get("data", [])
    if not isinstance(models, list) or any(not isinstance(entry, dict) for entry in models):
        raise CodexRuntimeError("model catalog has an invalid format")
    selected = (
        next((entry for entry in models if entry.get("model") == model), None)
        if model
        else (next((entry for entry in models if entry.get("isDefault")), None))
    )
    if selected is None:
        available = ", ".join(str(entry.get("model")) for entry in models)
        raise CodexRuntimeError(
            f"requested model is not in the account catalog; available: {available}"
        )
    selected_effort = effort if effort is not None else selected.get("defaultReasoningEffort")
    supported = {
        entry.get("reasoningEffort") for entry in selected.get("supportedReasoningEfforts", [])
    }
    if (
        not isinstance(selected_effort, str)
        or not selected_effort
        or (supported and selected_effort not in supported)
    ):
        raise CodexRuntimeError("requested reasoning effort is not supported by this model")
    return ModelRuntime(
        image_id, version, selected["model"], selected_effort, provider_hosts, models
    )
