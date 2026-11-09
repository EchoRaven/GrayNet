import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from graynet.artifacts.records import RunArtifacts
from graynet.cli import main
from graynet.execution.runner import CaseRun
from graynet.scenarios.definitions import load_case

CASE = replace(
    load_case(Path(__file__).parent / "fixtures/sustained_congestion"), environment="Public context"
)


def patch_backend(monkeypatch):
    class Backend:
        LAB_NAME = "test"
        evidence = {"phases": {"baseline": [10]}}
        workload_lock = threading.Lock()
        from types import SimpleNamespace

        from graynet.tools.network import DEFAULT_POLICY

        network = SimpleNamespace(policy=DEFAULT_POLICY)
        environment_sha256 = "test-env"

        def __init__(self, case):
            pass

        def reserve_agent_window(self, timeout):
            pass

    monkeypatch.setattr("graynet.scenarios.definitions.load_case", lambda path: CASE)
    monkeypatch.setattr("graynet.cli.load_case", lambda path: CASE)
    monkeypatch.setattr("graynet.environments.containerlab.ContainerlabBackend", Backend)


def test_cli_validation_writes_one_directory(monkeypatch, tmp_path, capsys):
    patch_backend(monkeypatch)
    result = CaseRun(
        CASE.case_id, "valid", (10,), (2,), (10,), None, None, (), "not_run", None, None, None
    )
    monkeypatch.setattr("graynet.execution.runner.validate_case", lambda case, backend: result)
    directory = tmp_path / "validation"
    monkeypatch.setattr(
        sys, "argv", ["graynet", "validate-case", "case", "--output", str(directory)]
    )
    assert main() == 0
    assert (directory / "result.json").is_file()
    assert (directory / "evaluator/evidence.json").is_file()
    assert not (directory / "agent").exists()
    assert "run directory" in capsys.readouterr().out


def test_preflight_failure_persists_attempt_without_auth_path(monkeypatch, tmp_path):
    patch_backend(monkeypatch)

    def unavailable(self):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("graynet.agents.codex.CodexDriver.preflight", unavailable)
    directory = tmp_path / "failure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graynet",
            "run-case",
            "case",
            "--output",
            str(directory),
            "--auth-file",
            "/private/credentials/auth.json",
            "--evaluation",
            "classification",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["stage"] == "preflight"
    assert not (directory / "result.json").exists()
    assert "/private/credentials" not in "".join(
        path.read_text() for path in directory.rglob("*") if path.is_file()
    )


def test_cli_audit_writes_a_distinct_report(monkeypatch, tmp_path):
    report = {"passed": True, "checks": {"no_socket": True}}
    monkeypatch.setattr("graynet.runtime.isolation.audit_agent_isolation", lambda: report)
    directory = tmp_path / "audit"
    monkeypatch.setattr(
        sys, "argv", ["graynet", "audit-agent-isolation", "--output", str(directory)]
    )
    assert main() == 0
    assert json.loads((directory / "audit.json").read_text()) == report
    assert not (directory / "result.json").exists()


def test_cli_summary_accepts_run_directory(monkeypatch, tmp_path, capsys):
    artifacts = RunArtifacts.create("test", "validate-case", root=tmp_path)
    artifacts.finish(
        {
            "run": {"case_id": "test", "status": "valid", "agent_status": "not_run"},
            "agent": {},
            "ground_truth": {"anomaly": True},
        }
    )
    monkeypatch.setattr(sys, "argv", ["graynet", "summarize", str(artifacts.directory)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["excluded"]["no_agent"] == 1


def test_cli_checks_valid_atif_and_rejects_the_previous_custom_schema(monkeypatch, tmp_path):
    from graynet.artifacts.records import write_json
    from graynet.artifacts.trajectory import normalize_trajectory

    path = tmp_path / "trajectory.json"
    write_json(path, normalize_trajectory("", prompt="Investigate", trace=[], metadata={}))
    monkeypatch.setattr(sys, "argv", ["graynet", "check-trajectory", str(path)])
    assert main() == 0
    write_json(path, {"schema_version": 1, "steps": [{"kind": "prompt"}]})
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


def test_semantic_evaluation_requires_an_explicit_judge_before_any_execution(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["graynet", "run-case", "case"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_interrupted_validation_retains_partial_network_evidence(monkeypatch, tmp_path):
    patch_backend(monkeypatch)

    def interrupt(case, backend):
        raise KeyboardInterrupt

    monkeypatch.setattr("graynet.execution.runner.validate_case", interrupt)
    directory = tmp_path / "interrupted"
    monkeypatch.setattr(
        sys, "argv", ["graynet", "validate-case", "case", "--output", str(directory)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 130
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["status"] == "interrupted"
    assert json.loads((directory / "evaluator/evidence.json").read_text())["phases"][
        "baseline"
    ] == [10]
    assert not (directory / "result.json").exists()
