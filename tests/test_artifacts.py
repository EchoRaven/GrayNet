import hashlib
import json

import pytest

from graynet.artifacts.migrations import convert_run_trajectories, import_legacy_run
from graynet.artifacts.records import (
    RunArtifacts,
    load_results,
    write_json,
)


def record():
    return {
        "run": {
            "case_id": "test-case",
            "status": "valid",
            "agent_status": "completed",
            "score": {"detection_correct": True},
        },
        "ground_truth": {"anomaly": True},
        "agent": {"driver": "codex", "status": "completed"},
        "evidence": {"phases": {"baseline": [10], "incident": [2]}},
        "diagnostic_trace": [
            {
                "tool": "throughput_probe",
                "arguments": {},
                "status": "completed",
                "result": {"throughput_mbps": 2},
            }
        ],
    }


def test_run_layout_separates_evidence_trace_and_summary(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.stage("evaluation")
    artifacts.finish(record(), trace=record()["diagnostic_trace"])
    result = json.loads((artifacts.directory / "result.json").read_text())
    assert set(result) == {"schema_version", "run", "ground_truth", "agent"}
    assert json.loads((artifacts.evaluator / "evidence.json").read_text()) == record()["evidence"]
    assert json.loads((artifacts.agent / "tools.json").read_text()) == record()["diagnostic_trace"]
    manifest = json.loads((artifacts.directory / "run.json").read_text())
    assert manifest["status"] == "finished" and manifest["finished_at"]
    for name, info in manifest["artifacts"].items():
        raw = (artifacts.directory / name).read_bytes()
        assert info == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_unique_run_ids_and_no_overwrites(tmp_path):
    first = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    second = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    assert first.directory != second.directory
    with pytest.raises(FileExistsError):
        RunArtifacts.create("test-case", "run-case", output=first.directory)
    with pytest.raises(ValueError, match="directory"):
        RunArtifacts.create("test-case", "run-case", output=tmp_path / "old.json")


@pytest.mark.parametrize(
    "command,root",
    [
        ("run-case", "results/runs"),
        ("validate-case", "diagnostics/validation"),
        ("smoke-case", "diagnostics/smoke"),
        ("audit-agent-isolation", "diagnostics/audits"),
        ("audit-network-tools", "diagnostics/audits"),
        ("stress-workloads", "diagnostics/stress"),
        ("development-check", "diagnostics/other"),
    ],
)
def test_default_roots_separate_agent_runs_from_development_artifacts(
    monkeypatch, tmp_path, command, root
):
    monkeypatch.chdir(tmp_path)
    artifacts = RunArtifacts.create("test-case", command)
    assert artifacts.directory.parent == tmp_path / root / "test-case"


def test_explicit_output_overrides_default_without_changing_it(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "chosen-run"
    artifacts = RunArtifacts.create("test-case", "run-case", output=output)
    assert artifacts.directory == output
    assert not (tmp_path / "results").exists()


def test_aggregation_rejects_changed_recorded_artifacts(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.finish(record())
    assert len(load_results([artifacts.directory])) == 1
    write_json(artifacts.evaluator / "evidence.json", {"tampered": True})
    with pytest.raises(ValueError, match="manifest"):
        load_results([artifacts.directory])


@pytest.mark.parametrize(
    "error,status",
    [(RuntimeError("model unavailable"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_failed_attempt_keeps_manifest_without_fabricated_result(tmp_path, error, status):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.stage("preflight")
    artifacts.fail(error)
    manifest = json.loads((artifacts.directory / "run.json").read_text())
    assert manifest["status"] == status and manifest["stage"] == "preflight"
    assert not (artifacts.directory / "result.json").exists()
    with pytest.raises(ValueError, match="no completed"):
        load_results([tmp_path])


def test_import_is_lossless_idempotent_and_does_not_copy_credentials(tmp_path):
    source = tmp_path / "old-run.json"
    write_json(source, record())
    old_agent = tmp_path / "old-run-agent"
    old_agent.mkdir()
    events = '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n'
    (old_agent / "events.jsonl").write_text(events)
    (old_agent / "prompt.txt").write_text("Check network")
    (old_agent / "auth.json").write_text("must not copy")
    before = {path: path.read_bytes() for path in (source, *old_agent.iterdir())}
    destination = import_legacy_run(source, root=tmp_path / "imported")
    assert destination == import_legacy_run(source, root=tmp_path / "imported")
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert (destination / "original/result.json").read_bytes() == before[source]
    assert (destination / "agent/logs/events.jsonl").read_text() == events
    assert not list(destination.rglob("auth.json"))
    assert (destination / "agent/trajectory.md").is_file()
    copied = json.loads((destination / "result.json").read_text())
    assert copied["run"] == record()["run"]
    assert copied["agent"] == record()["agent"]
    assert "private" not in str(copied)
    manifest = json.loads((destination / "run.json").read_text())
    assert manifest["origin"]["kind"] == "legacy_import"
    assert manifest["origin"]["original_run_time"] is None
    assert len(load_results([source, destination, tmp_path / "imported"])) == 1


def test_import_rejects_symlinked_artifacts(tmp_path):
    source = tmp_path / "old.json"
    write_json(source, record())
    agent = tmp_path / "old-agent"
    agent.mkdir()
    (agent / "prompt.txt").symlink_to(source)
    with pytest.raises(ValueError, match="symlinked"):
        import_legacy_run(source, root=tmp_path / "imported")
    manifest = next((tmp_path / "imported").rglob("run.json"))
    assert json.loads(manifest.read_text())["status"] == "failed"


def test_results_discovery_ignores_audits_originals_and_overlapping_paths(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.finish(record())
    original = artifacts.directory / "original/result.json"
    write_json(original, record())
    audit = RunArtifacts.create("agent-isolation", "audit", root=tmp_path / "audits")
    audit.finish_audit({"passed": True, "checks": {"no_socket": True}})
    assert (
        len(load_results([tmp_path, artifacts.directory, artifacts.directory / "result.json"])) == 1
    )
    assert not (audit.directory / "result.json").exists()
    assert json.loads((audit.directory / "audit.json").read_text())["passed"]


def test_import_audit_without_treating_it_as_an_evaluated_episode(tmp_path):
    source = tmp_path / "old-audit.json"
    write_json(source, {"passed": True, "checks": {"no_socket": True}})
    destination = import_legacy_run(source, root=tmp_path / "imported")
    assert (destination / "audit.json").is_file()
    assert not (destination / "result.json").exists()


def test_early_validation_import_keeps_missing_reference_unknown(tmp_path):
    source = tmp_path / "early-pilot.json"
    early = {
        "run": {"case_id": "pilot", "status": "valid", "agent_status": "not_run"},
        "evidence": {"phases": {}},
    }
    write_json(source, early)
    destination = import_legacy_run(source, root=tmp_path / "imported")
    result = json.loads((destination / "result.json").read_text())
    assert result["ground_truth"] is None and result["agent"] == {}
    assert result["run"] == early["run"]
    assert len(load_results([source, destination])) == 1


def test_early_agent_import_retains_trace_without_inventing_driver_or_reference(tmp_path):
    source = tmp_path / "old-smoke.json"
    early = {
        "run": {
            "case_id": "pilot",
            "status": "valid",
            "agent_status": "completed",
            "score": {"detection_correct": True},
        },
        "diagnostic_trace": [{"tool": "probe", "result": {"throughput": 2}}],
    }
    write_json(source, early)
    destination = import_legacy_run(source, root=tmp_path / "imported")
    result = json.loads((destination / "result.json").read_text())
    assert result["ground_truth"] is None and result["agent"] == {}
    assert result["run"]["score"] == early["run"]["score"]
    trajectory = json.loads((destination / "agent/trajectory.json").read_text())
    assert trajectory["agent"]["name"] == "unknown"
    assert json.loads(trajectory["steps"][0]["observation"]["results"][0]["content"]) == {
        "throughput": 2
    }
    assert (destination / "agent/tools.json").is_file()


def test_results_loader_rejects_unrelated_json(tmp_path):
    source = tmp_path / "evidence.json"
    write_json(source, {"phases": {}})
    with pytest.raises(ValueError, match="not an evaluation"):
        load_results([source])


def test_conversion_preserves_scores_originals_and_custom_views_and_updates_hashes(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    custom = {"schema_version": 1, "steps": [{"kind": "prompt", "text": "Investigate"}]}
    write_json(artifacts.agent / "trajectory.json", custom)
    (artifacts.agent / "trajectory.md").write_text("Old readable view")
    (artifacts.agent / "input").mkdir()
    (artifacts.agent / "input/prompt.txt").write_text("Investigate")
    artifacts.finish(record(), trace=record()["diagnostic_trace"])
    before_result = (artifacts.directory / "result.json").read_bytes()
    before_manifest = json.loads((artifacts.directory / "run.json").read_text())
    paths = convert_run_trajectories([tmp_path, artifacts.directory])
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["schema_version"] == "ATIF-v1.7"
    assert json.loads((artifacts.agent / "legacy/pre-atif/trajectory.json").read_text()) == custom
    assert (artifacts.agent / "legacy/pre-atif/trajectory.md").read_text() == "Old readable view"
    assert (artifacts.directory / "result.json").read_bytes() == before_result
    manifest = json.loads((artifacts.directory / "run.json").read_text())
    assert manifest["created_at"] == before_manifest["created_at"]
    assert manifest["finished_at"] == before_manifest["finished_at"]
    assert len(manifest["postprocessing"]) == 1
    for relative, entry in manifest["artifacts"].items():
        assert (
            entry["sha256"]
            == hashlib.sha256((artifacts.directory / relative).read_bytes()).hexdigest()
        )
    assert convert_run_trajectories([artifacts.directory]) == paths
    assert json.loads((artifacts.directory / "run.json").read_text()) == manifest


def test_conversion_refuses_to_bless_modified_evidence(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.finish(record(), trace=record()["diagnostic_trace"])
    write_json(artifacts.evaluator / "evidence.json", {"modified": True})
    with pytest.raises(ValueError, match="no longer matches"):
        convert_run_trajectories([artifacts.directory])


def test_refresh_preserves_previous_atif_and_does_not_touch_the_result(tmp_path):
    artifacts = RunArtifacts.create("test-case", "run-case", root=tmp_path)
    artifacts.finish(record(), trace=record()["diagnostic_trace"])
    [path] = convert_run_trajectories([artifacts.directory])
    previous = path.read_bytes()
    result_bytes = (artifacts.directory / "result.json").read_bytes()
    assert convert_run_trajectories([artifacts.directory], refresh=True) == [path]
    archived = (
        artifacts.agent
        / "legacy"
        / ("atif-" + hashlib.sha256(previous).hexdigest()[:12])
        / "trajectory.json"
    )
    assert archived.read_bytes() == previous
    assert (artifacts.directory / "result.json").read_bytes() == result_bytes
