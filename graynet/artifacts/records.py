"""Versioned, evaluator-owned records and artifact integrity."""

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def default_output_root(command: str) -> Path:
    """Keep scored agent runs separate from development diagnostics."""
    if command == "run-case":
        return Path("results/runs")
    diagnostics = {
        "validate-case": "validation",
        "smoke-case": "smoke",
        "audit-agent-isolation": "audits",
        "audit-network-tools": "audits",
        "stress-workloads": "stress",
    }
    return Path("diagnostics") / diagnostics.get(command, "other")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON document atomically, including on interrupted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def path_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", value)
    if value in ("", ".", ".."):
        raise ValueError("invalid run path component")
    return value


class RunArtifacts:
    """One exclusive directory per attempt; never mounted into the agent."""

    def __init__(
        self,
        directory: Path,
        *,
        case_id: str,
        command: str,
        requested: dict[str, Any] | None = None,
        origin: dict[str, Any] | None = None,
    ):
        if directory.suffix == ".json":
            raise ValueError("--output is now a run directory, not a JSON file")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.agent = self.directory / "agent"
        self.evaluator = self.directory / "evaluator"
        self.evaluator.mkdir()
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.directory.name,
            "case_id": case_id,
            "command": command,
            "status": "running",
            "stage": "setup",
            "created_at": utc_now(),
            "finished_at": None,
            "requested": requested or {},
            "origin": origin or {"kind": "execution"},
            "artifacts": {},
        }
        self._save()

    @classmethod
    def create(
        cls,
        case_id: str,
        command: str,
        *,
        output: Path | None = None,
        root: Path | None = None,
        requested: dict[str, Any] | None = None,
    ) -> "RunArtifacts":
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{command}-{uuid.uuid4().hex[:8]}"
        return cls(
            output
            if output is not None
            else (root if root is not None else default_output_root(command))
            / path_component(case_id)
            / run_id,
            case_id=case_id,
            command=command,
            requested=requested,
        )

    def _save(self) -> None:
        write_json(self.directory / "run.json", self.manifest)

    def stage(self, name: str) -> None:
        self.manifest["stage"] = name
        self._save()

    def finish(self, record: dict[str, Any], *, trace: list[dict[str, Any]] | None = None) -> None:
        # result.json remains directly readable by the existing aggregate scorer.
        result = {
            "schema_version": SCHEMA_VERSION,
            **{key: record[key] for key in ("run", "ground_truth", "agent")},
        }
        write_json(self.directory / "result.json", result)
        write_json(self.evaluator / "evidence.json", record.get("evidence", {}))
        if trace is not None and not (self.agent / "tools.json").exists():
            write_json(self.agent / "tools.json", trace)
        if record["agent"] and not (self.agent / "metadata.json").exists():
            write_json(self.agent / "metadata.json", record["agent"])
        self.manifest.update(
            status="finished",
            stage="complete",
            finished_at=utc_now(),
            case_status=record["run"]["status"],
            agent_status=record["run"]["agent_status"],
            evaluation_status=(record["run"].get("evaluation") or {}).get("status"),
        )
        self._index()

    def fail(self, error: BaseException) -> None:
        self.manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        self._index()

    def finish_audit(self, report: dict[str, Any]) -> None:
        write_json(self.directory / "audit.json", report)
        self.manifest.update(
            status="finished", stage="complete", finished_at=utc_now(), passed=report["passed"]
        )
        self._index()

    def _index(self) -> None:
        self.manifest["artifacts"] = artifact_index(self.directory)
        self._save()


def artifact_index(directory: Path) -> dict[str, Any]:
    return {
        path.relative_to(directory).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "run.json"
    }


def validate_result_record(value: Any, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("run"), dict):
        raise ValueError(f"not an evaluation result: {source}")
    if not all(key in value["run"] for key in ("case_id", "status", "agent_status")):
        raise ValueError(f"incomplete evaluation record: {source}")
    if "schema_version" in value and value["schema_version"] not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported result schema version: {source}")
    run = value["run"]
    if (
        not isinstance(run["case_id"], str)
        or not run["case_id"]
        or run["status"] not in ("valid", "invalid", "error")
        or run["agent_status"] not in ("not_run", "completed", "error")
    ):
        raise ValueError(f"invalid case identity or outcome: {source}")
    if not all(key in value for key in ("ground_truth", "agent")):
        # Early pilot records predate these fields. Unknown reference
        # stays null; do not backfill it from today's case definition.
        return {**value, "ground_truth": value.get("ground_truth"), "agent": value.get("agent", {})}
    return value


def load_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Accept result files, one run directory, or a results tree; deduplicate overlaps."""
    records = []
    identities = set()
    for supplied in paths:
        if supplied.is_dir():
            if (supplied / "run.json").is_file():
                candidates = (
                    [supplied / "result.json"] if (supplied / "result.json").is_file() else []
                )
            else:
                candidates = sorted(
                    path
                    for path in supplied.rglob("result.json")
                    if (path.parent / "run.json").is_file()
                )
        else:
            if not supplied.is_file():
                raise FileNotFoundError(supplied)
            candidates = [supplied]
        for path in candidates:
            record = validate_result_record(json.loads(path.read_text(encoding="utf-8")), path)
            manifest_path = path.parent / "run.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["status"] != "finished":
                    continue
                for relative, expected in manifest.get("artifacts", {}).items():
                    artifact = path.parent / relative
                    if (
                        artifact.is_symlink()
                        or not artifact.resolve().is_relative_to(path.parent.resolve())
                        or not artifact.is_file()
                        or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected["sha256"]
                    ):
                        raise ValueError(f"artifact no longer matches its manifest: {artifact}")
                identity = manifest.get("origin", {}).get("source_sha256") or str(path.resolve())
            else:
                identity = hashlib.sha256(path.read_bytes()).hexdigest()
            if identity not in identities:
                records.append(record)
                identities.add(identity)
    if not records:
        raise ValueError(
            "no completed evaluation records found (preflight failures and unfinished runs have no result)"
        )
    return records
