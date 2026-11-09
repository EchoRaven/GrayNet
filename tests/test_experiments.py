import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from graynet.artifacts.records import RunArtifacts, write_json
from graynet.execution.experiments import (
    load_experiment,
    plan_experiment,
    resume_experiment,
    run_experiment,
)
from graynet.scenarios.definitions import load_case

ROOT = Path(__file__).parents[1]
SPEC = load_experiment(ROOT / "experiments/pilot.yaml")


def executor(commands, *, agent_error=False):
    def execute(command, **kwargs):
        commands.append(command)
        case = load_case(command[4])
        output = Path(command[command.index("--output") + 1])
        artifacts = RunArtifacts.create(case.case_id, "run-case", output=output)
        answer = {"detected": case.ground_truth.anomaly}
        score = (
            None
            if agent_error
            else {
                "detection_correct": True,
                "localization_correct": True,
                "mechanism_correct": True,
            }
        )
        semantic = command[command.index("--evaluation") + 1] == "semantic"
        evaluation = (
            {
                "mode": "semantic",
                "status": "not_run" if agent_error else "completed",
                "identity_sha256": "test-judge",
                "identity": {
                    "model": {
                        "model": command[command.index("--judge-model") + 1],
                        "reasoning_effort": command[command.index("--judge-reasoning-effort") + 1],
                        "timeout_seconds": int(command[command.index("--judge-timeout") + 1]),
                    }
                },
                "judgment": None if agent_error else {"passed": True, "dimensions": {}},
            }
            if semantic
            else {"mode": "classification", "status": "not_run"}
        )
        artifacts.finish(
            {
                "run": {
                    "case_id": case.case_id,
                    "status": "valid",
                    "agent_status": "error" if agent_error else "completed",
                    "agent_answer": answer,
                    "score": score,
                    "evaluation": evaluation,
                },
                "ground_truth": {"anomaly": case.ground_truth.anomaly},
                "agent": {"driver": "codex", "model": "test"},
            }
        )
        return subprocess.CompletedProcess(command, 1 if agent_error else 0, "finished", "")

    return execute


def test_matrix_order_repetitions_and_seed_are_predeclared():
    spec = replace(
        SPEC,
        cases=(ROOT / "cases/link_loss", ROOT / "cases/healthy_control"),
        repetitions=2,
        agents=(*SPEC.agents, {**SPEC.agents[0], "id": "second", "reasoning_effort": "medium"}),
    )
    first, second = plan_experiment(spec), plan_experiment(spec)
    assert first["trials"] == second["trials"] and len(first["trials"]) == 8
    assert {
        trial["fault_seed"] for trial in first["trials"] if trial["case_id"] == "link-loss-01"
    } == {11, 12}
    assert all(
        trial["fault_seed"] is None
        for trial in first["trials"]
        if trial["case_id"] == "healthy-control-01"
    )
    assert first["plan_sha256"] == second["plan_sha256"]


def test_semantic_experiments_freeze_one_judge_for_all_agents(tmp_path):
    raw = yaml.safe_load((ROOT / "experiments/pilot.yaml").read_text())
    raw.update(
        schema_version=2,
        cases=[str(ROOT / "cases/healthy_control")],
        evaluation={
            "mode": "semantic",
            "model": "judge-test",
            "reasoning_effort": "low",
            "timeout_seconds": 30,
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    plan = plan_experiment(load_experiment(path))
    assert plan["schema_version"] == 2 and plan["evaluation"] == raw["evaluation"]
    commands = []
    summary = run_experiment(
        tmp_path / "batch", plan, auth_file=Path("ignored"), execute=executor(commands)
    )
    assert commands[0][commands[0].index("--judge-model") + 1] == "judge-test"
    assert summary["experiment"]["evaluation_errors"] == 0


def test_serial_runner_retains_valid_agent_errors_and_auth_is_not_in_plan(tmp_path):
    plan = plan_experiment(replace(SPEC, cases=(ROOT / "cases/link_loss",), repetitions=2))
    commands = []
    directory = tmp_path / "experiment"
    summary = run_experiment(
        directory,
        plan,
        auth_file=Path("/private/auth.json"),
        execute=executor(commands, agent_error=True),
    )
    assert len(commands) == 2 and summary["agents"][0]["agent_errors"] == 2
    assert summary["experiment"]["failed_case_trials"] == 0
    assert "/private/auth.json" not in (directory / "experiment.json").read_text()
    assert {command[command.index("--fault-seed") + 1] for command in commands} == {"11", "12"}
    resume_experiment(
        directory,
        auth_file=Path("ignored"),
        execute=lambda *args, **kwargs: pytest.fail("completed trial reran"),
    )
    with pytest.raises(FileExistsError):
        run_experiment(directory, plan, auth_file=Path("ignored"))


@pytest.mark.parametrize("mutation", ["case_hash", "runtime", "path", "seed"])
def test_resume_refuses_changed_identity_or_escaped_paths(tmp_path, mutation):
    directory = tmp_path / "batch"
    directory.mkdir()
    plan = plan_experiment(replace(SPEC, cases=(ROOT / "cases/healthy_control",)))
    if mutation == "case_hash":
        plan["case_sha256"][str(ROOT / "cases/healthy_control")] = "changed"
    elif mutation == "runtime":
        plan["runtime_sha256"] = "changed"
    elif mutation == "path":
        plan["trials"][0]["run_directory"] = "../../outside"
    else:
        plan["trials"][0]["fault_seed"] = 999
    write_json(directory / "experiment.json", plan)
    with pytest.raises(ValueError):
        resume_experiment(
            directory,
            auth_file=Path("ignored"),
            execute=lambda *args, **kw: pytest.fail("invalid plan executed"),
        )


def test_interrupted_trials_are_retained_and_not_silently_retried(tmp_path):
    directory = tmp_path / "batch"
    plan = plan_experiment(replace(SPEC, cases=(ROOT / "cases/healthy_control",)))
    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            directory,
            plan,
            auth_file=Path("ignored"),
            execute=lambda *args, **kw: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    saved = json.loads((directory / "experiment.json").read_text())
    assert saved["status"] == "interrupted" and saved["trials"][0]["status"] == "interrupted"
    with pytest.raises(ValueError, match="unfinished"):
        resume_experiment(directory, auth_file=Path("ignored"))


def test_preflight_failures_remain_separate_from_scored_cases(tmp_path):
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, "", "preflight failed")

    summary = run_experiment(
        tmp_path / "batch",
        plan_experiment(replace(SPEC, cases=(ROOT / "cases/healthy_control",))),
        auth_file=Path("ignored"),
        execute=fail,
    )
    assert summary["agents"] == [] and summary["experiment"]["preflight_errors"] == 1


def test_runtime_change_mid_batch_stops_before_next_trial(monkeypatch, tmp_path):
    plan = plan_experiment(replace(SPEC, cases=(ROOT / "cases/link_loss",), repetitions=2))
    commands = []
    execute = executor(commands)
    monkeypatch.setattr(
        "graynet.execution.experiments.runtime_fingerprint",
        lambda: "changed" if commands else plan["runtime_sha256"],
    )
    with pytest.raises(ValueError, match="inputs changed while running"):
        run_experiment(tmp_path / "batch", plan, auth_file=Path("ignored"), execute=execute)
    saved = json.loads((tmp_path / "batch/experiment.json").read_text())
    assert len(commands) == 1 and saved["status"] == "interrupted"
    assert [trial["status"] for trial in saved["trials"]] == ["finished", "pending"]


@pytest.mark.parametrize(
    "field,value",
    [("repetitions", True), ("order_seed", -1), ("fault_seed", 2**32), ("agents", [])],
)
def test_bad_experiment_configs_fail_before_execution(tmp_path, field, value):
    raw = yaml.safe_load((ROOT / "experiments/pilot.yaml").read_text())
    raw["cases"] = [str(ROOT / "cases/healthy_control")]
    raw[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment(path)


@pytest.mark.parametrize(
    "evaluation,message",
    [
        (None, "evaluation must be a mapping"),
        ({"mode": "other"}, "invalid evaluation settings"),
        ({"mode": "semantic"}, "invalid evaluation settings"),
        (
            {"mode": "semantic", "model": "", "reasoning_effort": "low", "timeout_seconds": 30},
            "judge model and effort must be explicit",
        ),
        (
            {"mode": "semantic", "model": "judge", "reasoning_effort": "low", "timeout_seconds": 0},
            "judge timeout must be positive",
        ),
    ],
)
def test_evaluation_validation_preserves_error_messages(tmp_path, evaluation, message):
    raw = yaml.safe_load((ROOT / "experiments/pilot.yaml").read_text())
    raw["evaluation"] = evaluation
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=f"^{message}$"):
        load_experiment(path)


def test_classification_batch_preserves_command_arguments(tmp_path):
    spec = replace(SPEC, cases=(ROOT / "cases/healthy_control",), evaluation=None)
    commands = []
    directory = tmp_path / "batch"
    plan = plan_experiment(spec)
    run_experiment(directory, plan, auth_file=Path("auth.json"), execute=executor(commands))
    agent = spec.agents[0]
    assert commands == [
        [
            sys.executable,
            "-m",
            "graynet",
            "run-case",
            str(spec.cases[0]),
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
            str(directory / "runs" / plan["trials"][0]["id"]),
            "--auth-file",
            "auth.json",
            "--evaluation",
            "classification",
        ]
    ]
