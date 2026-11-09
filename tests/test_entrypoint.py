import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "graynet_container_entrypoint", Path(__file__).parents[1] / "docker/codex/entrypoint.py"
)
entrypoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entrypoint)


@pytest.mark.parametrize("fails", [False, True])
def test_model_discovery_uses_tool_free_config_and_reaps_the_process(monkeypatch, fails):
    class Process:
        stdin = io.StringIO()
        stdout = io.StringIO(
            ""
            if fails
            else "\n".join(
                json.dumps(reply)
                for reply in [
                    {"id": 1, "result": {}},
                    {"id": 2, "result": {"data": [{"model": "first"}], "nextCursor": "page2"}},
                    {"id": 3, "result": {"data": [{"model": "second"}], "nextCursor": None}},
                ]
            )
        )
        terminated = waited = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.waited = True

    process = Process()

    def launch(args, **kwargs):
        assert args == ["codex", "app-server"]
        assert "stderr" not in kwargs  # Startup errors reach the host's captured stderr.
        return process

    monkeypatch.setattr(entrypoint.subprocess, "Popen", launch)
    if fails:
        with pytest.raises(RuntimeError, match="stopped during model discovery"):
            entrypoint.discover_models()
    else:
        result = entrypoint.discover_models()
        assert [row["model"] for row in result["data"]] == ["first", "second"]
        requests = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
        assert requests[-1]["params"]["cursor"] == "page2"
    assert process.terminated and process.waited
