"""Serial, resumable experiments with frozen inputs and explicit repetitions."""

import hashlib
import json
import random
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from graynet.agents.registry import available_drivers
from graynet.artifacts.records import load_results, write_json
from graynet.environments.containerlab import create_backend
from graynet.evaluation.reporting import summarize
from graynet.paths import REPOSITORY_ROOT
from graynet.scenarios.definitions import load_case

FAULT_SEED_LIMIT = 2**32


def runtime_fingerprint() -> str:
    """Hash the source and dependency inputs used to execute a batch."""
    root = REPOSITORY_ROOT
    paths = sorted(
        [
            *root.joinpath("graynet").rglob("*.py"),
            *root.joinpath("docker").rglob("*.py"),
            *root.joinpath("docker").rglob("Dockerfile"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    )
    return hashlib.sha256(
        json.dumps(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def case_fingerprint(directory: Path) -> str:
    """Hash a scenario manifest and its bound environment inputs."""
    case = load_case(directory)
    backend = create_backend(case)
    value = {
        "manifest": (directory / "scenario.yaml").read_text(),
        "environment": backend.environment_sha256,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _slug(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
        raise ValueError("experiment and agent IDs must be lowercase slugs")
    return value


def plan_digest(plan: dict[str, Any]) -> str:
    """Hash frozen trial inputs without including execution progress."""
    identity = {
        key: plan[key]
        for key in (
            "schema_version",
            "name",
            "order_seed",
            "fault_seed",
            "repetitions",
            "runtime_sha256",
            "case_sha256",
        )
    }
    identity["trials"] = [
        {
            key: trial[key]
            for key in ("id", "case_directory", "case_id", "agent", "repetition", "fault_seed")
        }
        for trial in plan["trials"]
    ]
    if plan["schema_version"] == 2:
        identity["evaluation"] = plan["evaluation"]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated cases, agent settings, and repetition settings for a batch."""

    name: str
    cases: tuple[Path, ...]
    agents: tuple[dict[str, Any], ...]
    repetitions: int
    order_seed: int
    fault_seed: int
    evaluation: dict[str, Any] | None = None


def _validate_evaluation_settings(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    if evaluation == {"mode": "classification"}:
        return
    if (
        set(evaluation) != {"mode", "model", "reasoning_effort", "timeout_seconds"}
        or evaluation["mode"] != "semantic"
    ):
        raise ValueError("invalid evaluation settings")
    if any(
        not isinstance(evaluation[field], str) or not evaluation[field].strip()
        for field in ("model", "reasoning_effort")
    ):
        raise ValueError("judge model and effort must be explicit")
    if type(evaluation["timeout_seconds"]) is not int or evaluation["timeout_seconds"] < 1:
        raise ValueError("judge timeout must be positive")


def _validate_agent_settings(agent: Any) -> None:
    fields = {
        "id",
        "driver",
        "model",
        "reasoning_effort",
        "timeout_seconds",
        "max_tool_calls",
    }
    if (
        not isinstance(agent, dict)
        or set(agent) != fields
        or agent["driver"] not in available_drivers()
    ):
        raise ValueError(
            "each agent needs a registered driver, explicit model, effort, and budgets"
        )
    _slug(agent["id"])
    for field in ("model", "reasoning_effort"):
        if not isinstance(agent[field], str) or not agent[field].strip():
            raise ValueError(f"agent {field} must be explicit")
    for field in ("timeout_seconds", "max_tool_calls"):
        if type(agent[field]) is not int or agent[field] < 1:
            raise ValueError(f"agent {field} must be positive")


def load_experiment(path: Path) -> ExperimentSpec:
    """Load a batch configuration and reject invalid settings before execution."""
    path = path.resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("experiment must be a mapping")
    fields = {
        "schema_version",
        "name",
        "cases",
        "agents",
        "repetitions",
        "order_seed",
        "fault_seed",
    }
    if raw.get("schema_version") == 2:
        fields.add("evaluation")
    if (
        set(raw) != fields
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] not in (1, 2)
    ):
        raise ValueError("invalid experiment fields or schema_version")
    evaluation = raw.get("evaluation")
    if raw["schema_version"] == 2:
        _validate_evaluation_settings(evaluation)
    for field in ("repetitions", "order_seed", "fault_seed"):
        if type(raw[field]) is not int or raw[field] < (1 if field == "repetitions" else 0):
            raise ValueError(f"invalid {field}")
    if raw["fault_seed"] + raw["repetitions"] - 1 >= FAULT_SEED_LIMIT:
        raise ValueError("fault seeds must fit unsigned 32-bit values")
    if (
        not isinstance(raw["cases"], list)
        or not raw["cases"]
        or any(not isinstance(item, str) for item in raw["cases"])
    ):
        raise ValueError("cases must be a nonempty path list")
    cases = tuple((path.parent / item).resolve() for item in raw["cases"])
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate experiment cases")
    definitions = [load_case(directory) for directory in cases]
    if any(case.partial_path is not None for case in definitions):
        raise ValueError("partial path cases use plan-probe-experiment")
    ids = [case.case_id for case in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError("experiment cases must have unique IDs")
    if (
        evaluation is not None
        and evaluation["mode"] == "semantic"
        and any(not case.ground_truth.diagnosis for case in definitions)
    ):
        raise ValueError("semantic experiments require a reference diagnosis for each case")
    agents = raw["agents"]
    if not isinstance(agents, list) or not agents:
        raise ValueError("agents must be a nonempty list")
    for agent in agents:
        _validate_agent_settings(agent)
    if len({agent["id"] for agent in agents}) != len(agents):
        raise ValueError("duplicate agent IDs")
    return ExperimentSpec(
        _slug(raw["name"]),
        cases,
        tuple(agents),
        raw["repetitions"],
        raw["order_seed"],
        raw["fault_seed"],
        evaluation,
    )


def plan_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    """Build and shuffle a frozen trial matrix without executing it."""
    hashes = {str(directory): case_fingerprint(directory) for directory in spec.cases}
    trials = []
    for repetition in range(spec.repetitions):
        for directory in spec.cases:
            case = load_case(directory)
            for agent in spec.agents:
                trials.append(
                    {
                        "case_directory": str(directory),
                        "case_id": case.case_id,
                        "agent": agent,
                        "repetition": repetition + 1,
                        "fault_seed": spec.fault_seed + repetition
                        if case.fault["kind"] == "link_loss"
                        else None,
                        "status": "pending",
                    }
                )
    random.Random(spec.order_seed).shuffle(trials)
    for index, trial in enumerate(trials, 1):
        trial["id"] = (
            f"trial-{index:04d}-{trial['case_id']}-{trial['agent']['id']}-r{trial['repetition']:03d}"
        )
    plan = {
        "schema_version": 2 if spec.evaluation is not None else 1,
        "name": spec.name,
        "status": "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "order_seed": spec.order_seed,
        "fault_seed": spec.fault_seed,
        "repetitions": spec.repetitions,
        "runtime_sha256": runtime_fingerprint(),
        "case_sha256": hashes,
        "trials": trials,
    }
    if spec.evaluation is not None:
        plan["evaluation"] = spec.evaluation
    plan["plan_sha256"] = plan_digest(plan)
    return plan


def _trial_command(
    trial: dict[str, Any],
    evaluation: dict[str, Any],
    run_directory: Path,
    auth_file: Path,
) -> list[str]:
    agent = trial["agent"]
    command = [
        sys.executable,
        "-m",
        "graynet",
        "run-case",
        trial["case_directory"],
        "--agent",
        agent["driver"],
        "--model",
        agent["model"],
        "--reasoning-effort",
        agent["reasoning_effort"],
        "--timeout",
        str(agent["timeout_seconds"]),
        "--max-tool-calls",
        str(agent["max_tool_calls"]),
        "--output",
        str(run_directory),
        "--auth-file",
        str(auth_file),
        "--evaluation",
        evaluation["mode"],
    ]
    if evaluation["mode"] == "semantic":
        command.extend(
            [
                "--judge-model",
                evaluation["model"],
                "--judge-reasoning-effort",
                evaluation["reasoning_effort"],
                "--judge-timeout",
                str(evaluation["timeout_seconds"]),
            ]
        )
    if trial["fault_seed"] is not None:
        command.extend(["--fault-seed", str(trial["fault_seed"])])
    return command


def run_experiment(
    directory: Path,
    plan: dict[str, Any],
    *,
    auth_file: Path,
    execute: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """Never replace failed trials or silently retry partially recorded episodes."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "experiment.json", plan)
    return resume_experiment(directory, auth_file=auth_file, execute=execute, progress=progress)


def resume_experiment(
    directory: Path,
    *,
    auth_file: Path,
    execute: Callable[..., Any] = subprocess.run,
    progress: Callable[[str], Any] = print,
) -> dict[str, Any]:
    """Execute pending trials only, preserving completed and interrupted attempts."""
    directory = directory.resolve()
    path = directory / "experiment.json"
    plan = json.loads(path.read_text())
    try:
        if plan.get("schema_version") not in (1, 2) or plan.get("plan_sha256") != plan_digest(plan):
            raise ValueError("experiment plan identity changed")
        for trial in plan["trials"]:
            _slug(trial["id"])
            if trial["status"] not in ("pending", "running", "interrupted", "finished"):
                raise ValueError("invalid trial state")
            expected = f"runs/{trial['id']}"
            if trial.get("run_directory", expected) != expected:
                raise ValueError("trial run directory escaped its frozen identity")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed experiment plan") from exc
    if plan["runtime_sha256"] != runtime_fingerprint() or any(
        case_fingerprint(Path(source)) != digest for source, digest in plan["case_sha256"].items()
    ):
        raise ValueError(
            "runtime or scenario inputs changed; start a new experiment rather than mixing revisions"
        )
    if any(trial["status"] in ("running", "interrupted") for trial in plan["trials"]):
        raise ValueError("an unfinished trial needs inspection; refusing to silently rerun it")
    plan["status"] = "running"
    write_json(path, plan)
    current = None
    try:
        for trial in plan["trials"]:
            if trial["status"] != "pending":
                continue
            if (
                runtime_fingerprint() != plan["runtime_sha256"]
                or case_fingerprint(Path(trial["case_directory"]))
                != plan["case_sha256"][trial["case_directory"]]
            ):
                raise ValueError("experiment inputs changed while running")
            current = trial
            run_dir = directory / "runs" / trial["id"]
            if run_dir.exists():
                raise ValueError(
                    "pending trial already has artifacts; inspect instead of overwriting"
                )
            trial.update(status="running", run_directory=str(run_dir.relative_to(directory)))
            write_json(path, plan)
            progress(f"{trial['id']}: starting")
            evaluation = plan.get("evaluation", {"mode": "classification"})
            command = _trial_command(trial, evaluation, run_dir, auth_file)
            completed = execute(command, text=True, capture_output=True, check=False)
            logs = directory / "logs"
            logs.mkdir(exist_ok=True)
            (logs / f"{trial['id']}.stdout.txt").write_text(completed.stdout)
            (logs / f"{trial['id']}.stderr.txt").write_text(completed.stderr)
            manifest_path = run_dir / "run.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            trial.update(
                status="finished",
                exit_code=completed.returncode,
                case_status=manifest.get("case_status", "preflight_error"),
                agent_status=manifest.get("agent_status"),
                run_status=manifest.get("status", "missing"),
                evaluation_status=manifest.get("evaluation_status"),
            )
            write_json(path, plan)
            progress(f"{trial['id']}: {trial['case_status']}, agent {trial['agent_status']}")
            current = None
            result_path = run_dir / "result.json"
            if result_path.is_file():
                record = json.loads(result_path.read_text())
                recorded_evaluation = record["run"].get("evaluation") or {}
                if evaluation["mode"] == "semantic":
                    actual = recorded_evaluation.get("identity", {}).get("model", {})
                    if recorded_evaluation.get("mode") != "semantic" or any(
                        actual.get(field) != evaluation[field]
                        for field in ("model", "reasoning_effort", "timeout_seconds")
                    ):
                        raise ValueError("recorded judge differs from the frozen evaluation plan")
                if record["run"].get("cleanup_errors") or record.get("agent", {}).get(
                    "cleanup_errors"
                ):
                    raise ValueError("trial cleanup failed; pending trials were not run")
                judge_completion = run_dir / "evaluator/judge/completion.json"
                if judge_completion.is_file() and json.loads(judge_completion.read_text()).get(
                    "metadata", {}
                ).get("cleanup_errors"):
                    raise ValueError("judge cleanup failed; pending trials were not run")
    except BaseException as exc:
        if current is not None:
            current["status"] = "interrupted"
        plan.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        write_json(path, plan)
        raise
    selected = [
        directory / trial["run_directory"]
        for trial in plan["trials"]
        if trial.get("run_status") == "finished"
    ]
    summary = summarize(load_results(selected)) if selected else summarize([])
    summary["experiment"] = {
        "planned_trials": len(plan["trials"]),
        "finished_trials": sum(trial["status"] == "finished" for trial in plan["trials"]),
        "preflight_errors": sum(
            trial.get("case_status") == "preflight_error" for trial in plan["trials"]
        ),
        "failed_case_trials": sum(
            trial.get("case_status") in ("error", "invalid") for trial in plan["trials"]
        ),
        "evaluation_errors": sum(
            trial.get("evaluation_status") == "error" for trial in plan["trials"]
        ),
    }
    write_json(directory / "summary.json", summary)
    plan.update(status="finished", finished_at=datetime.now(UTC).isoformat())
    plan.pop("error", None)
    write_json(path, plan)
    return summary
