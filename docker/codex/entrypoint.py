"""Initialize only this disposable container's Codex home, then execute Codex."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_PAGE_SIZE = 100
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5


def discover_models() -> dict[str, Any]:
    """Use the tool-free discovery config without inventing an MCP server table."""
    process = subprocess.Popen(
        ["codex", "app-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # Inherit stderr so a startup failure remains diagnosable by the host.
        # The discovery config has no MCP servers; no model turn is started.
        text=True,
    )

    def request(identifier: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        process.stdin.write(
            json.dumps({"id": identifier, "method": method, "params": params}) + "\n"
        )
        process.stdin.flush()
        for line in process.stdout:
            reply = json.loads(line)
            if reply.get("id") == identifier:
                if "error" in reply:
                    raise RuntimeError("Codex model discovery failed")
                return reply["result"]
        raise RuntimeError("Codex stopped during model discovery")

    try:
        request(1, "initialize", {"clientInfo": {"name": "graynet", "version": "0.1.0"}})
        process.stdin.write('{"method":"initialized","params":{}}\n')
        process.stdin.flush()
        result = request(2, "model/list", {"limit": MODEL_PAGE_SIZE, "includeHidden": False})
        identifier = 3
        while result.get("nextCursor"):
            page = request(
                identifier,
                "model/list",
                {
                    "limit": MODEL_PAGE_SIZE,
                    "includeHidden": False,
                    "cursor": result["nextCursor"],
                },
            )
            result["data"].extend(page["data"])
            result["nextCursor"] = page.get("nextCursor")
            identifier += 1
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)


def main() -> None:
    """Prepare isolated credentials and config, then discover models or execute the CLI."""
    agent_home = Path(os.environ["CODEX_HOME"])
    agent_home.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("/run/codex-auth.json", agent_home / "auth.json")
    (agent_home / "auth.json").chmod(0o600)
    shutil.copyfile("/workspace/config.toml", agent_home / "config.toml")
    if sys.argv[1:] == ["--list-models"]:
        print(json.dumps(discover_models()), flush=True)
        return
    os.execvp("codex", ["codex", *sys.argv[1:]])


if __name__ == "__main__":
    main()
