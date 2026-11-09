"""Explicit offline imports and trajectory migrations; never part of episode execution."""

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from graynet.artifacts.records import (
    RunArtifacts,
    artifact_index,
    path_component,
    utc_now,
    validate_result_record,
    write_json,
)


def convert_run_trajectories(paths: Iterable[Path], *, refresh: bool = False) -> list[Path]:
    """Replace derived views only; archive old custom views and reindex the manifest."""
    from graynet.artifacts.atif import Trajectory
    from graynet.artifacts.trajectory import save_trajectory

    converted = []
    seen = set()
    for supplied in paths:
        if not supplied.is_dir():
            raise ValueError(f"expected a run directory or results tree: {supplied}")
        manifests = (
            [supplied / "run.json"]
            if (supplied / "run.json").is_file()
            else sorted(supplied.rglob("run.json"))
        )
        for manifest_path in manifests:
            directory = manifest_path.parent.resolve()
            if directory in seen:
                continue
            seen.add(directory)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "running":
                raise ValueError(
                    f"refusing to postprocess an active or unfinished run: {directory}"
                )
            agent = directory / "agent"
            if not agent.is_dir():
                continue
            trajectory_file = agent / "trajectory.json"
            existing = None
            if trajectory_file.is_file():
                existing = json.loads(trajectory_file.read_text(encoding="utf-8"))
                if existing.get("schema_version") == "ATIF-v1.7":
                    Trajectory.model_validate(existing)
                    if not refresh:
                        converted.append(trajectory_file)
                        continue
            # Check integrity before touching a recorded experiment. Re-indexing
            # must not silently bless unrelated modifications to its evidence.
            for relative, expected in manifest.get("artifacts", {}).items():
                path = directory / relative
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
                ):
                    raise ValueError(f"artifact no longer matches its manifest: {path}")
            preserved = []
            backup_category = "pre-atif"
            if existing and existing.get("schema_version") == "ATIF-v1.7":
                backup_category = (
                    "atif-" + hashlib.sha256(trajectory_file.read_bytes()).hexdigest()[:12]
                )
            for name in ("trajectory.json", "trajectory.md"):
                previous = agent / name
                if previous.is_file():
                    backup = agent / "legacy" / backup_category / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if backup.exists():
                        if backup.read_bytes() != previous.read_bytes():
                            raise FileExistsError(f"different backup already exists: {backup}")
                    else:
                        shutil.copyfile(previous, backup)
                    preserved.append(backup.relative_to(directory).as_posix())
            metadata_file = agent / "metadata.json"
            result_file = directory / "result.json"
            record = (
                json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
            )
            metadata = (
                json.loads(metadata_file.read_text(encoding="utf-8"))
                if metadata_file.exists()
                else record.get("agent", {})
            )
            if "status" not in metadata and record.get("run"):
                metadata = {**metadata, "status": record["run"]["agent_status"]}
            trace_file = agent / "tools.json"
            trace = (
                json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.exists() else []
            )
            save_trajectory(agent, trace=trace, metadata=metadata)
            manifest.setdefault("postprocessing", []).append(
                {
                    "kind": "atif_conversion",
                    "schema_version": "ATIF-v1.7",
                    "converted_at": utc_now(),
                    "preserved": preserved,
                    "refresh": refresh,
                }
            )
            manifest["artifacts"] = artifact_index(directory)
            write_json(manifest_path, manifest)
            converted.append(trajectory_file)
    if not converted:
        raise ValueError("no run directories with agent recordings found")
    return converted


# Explicit allowlist: do not copy credentials, sessions, or arbitrary adjacent files.
LEGACY_AGENT_FILES = {
    "prompt.txt": "input/prompt.txt",
    "config.toml": "input/config.toml",
    "diagnosis.schema.json": "input/diagnosis.schema.json",
    "events.jsonl": "logs/events.jsonl",
    "stderr.txt": "logs/stderr.txt",
    "egress.jsonl": "logs/egress.jsonl",
    "tools.json": "tools.json",
    "diagnosis.json": "diagnosis.json",
    "metadata.json": "metadata.json",
}


def import_legacy_run(source: Path, *, root: Path = Path("diagnostics/imported")) -> Path:
    """Copy a historical attempt, retaining original bytes, scores and provenance."""
    from graynet.artifacts.trajectory import save_trajectory

    if source.is_symlink() or not source.is_file():
        raise ValueError("legacy source must be a regular JSON run file")
    raw = source.read_bytes()
    record = json.loads(raw)
    is_audit = (
        isinstance(record, dict)
        and type(record.get("passed")) is bool
        and isinstance(record.get("checks"), dict)
    )
    if not is_audit:
        record = validate_result_record(record, source)
    if "schema_version" in record:
        raise ValueError("this run already uses the structured format")
    digest = hashlib.sha256(raw).hexdigest()
    case_id = "agent-isolation" if is_audit else record["run"]["case_id"]
    destination = root / path_component(case_id) / f"{path_component(source.stem)}-{digest[:12]}"
    origin = {
        "kind": "legacy_import",
        "source": str(source.resolve()),
        "source_sha256": digest,
        "original_run_time": None,
        "note": "Imported copy, not a new execution; original timestamps and scores are unchanged.",
    }
    if destination.exists():
        manifest = json.loads((destination / "run.json").read_text(encoding="utf-8"))
        if manifest.get("origin") == origin and manifest.get("status") == "finished":
            return destination.resolve()
        raise FileExistsError(f"incomplete or different import already exists: {destination}")
    artifacts = RunArtifacts(destination, case_id=case_id, command="import", origin=origin)
    try:
        artifacts.stage("import")
        originals = artifacts.directory / "original"
        originals.mkdir()
        shutil.copyfile(source, originals / "result.json")
        if is_audit:
            artifacts.finish_audit(record)
            return artifacts.directory
        legacy_agent = source.with_name(source.stem + "-agent")
        if legacy_agent.is_symlink():
            raise ValueError("legacy agent directory must not be a symlink")
        for name, target in LEGACY_AGENT_FILES.items():
            path = legacy_agent / name
            if path.is_symlink():
                raise ValueError(f"refusing a symlinked agent artifact: {path}")
            if path.is_file():
                copied = artifacts.agent / target
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, copied)
        # Observations already embedded in old records survive even if -agent is absent.
        trace = record.get("diagnostic_trace", [])
        if (artifacts.agent / "tools.json").is_file():
            trace = json.loads((artifacts.agent / "tools.json").read_text(encoding="utf-8"))
        if record["agent"] or trace:
            save_trajectory(
                artifacts.agent,
                trace=trace,
                metadata={"status": record["run"]["agent_status"], **record["agent"]},
            )
        artifacts.finish(record, trace=trace if record["agent"] or trace else None)
    except BaseException as exc:
        artifacts.fail(exc)
        raise
    return artifacts.directory
