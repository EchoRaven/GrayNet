"""Read-only verification of a finished experiment and its native ATIF files.

Usage: python scripts/check_experiment_artifacts.py results/experiments/pilot
Incorrect diagnoses and explicitly excluded cases are retained outcomes, not
artifact corruption. This checks frozen identities, hashes, exports, recorded
visibility checks and the stored aggregate; it does not judge explanations.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from graynet.agents.base import Diagnosis, validate_submission
from graynet.artifacts.atif.validator import TrajectoryValidator
from graynet.artifacts.records import load_results
from graynet.evaluation.integrity import audit_evidence, audit_observations
from graynet.evaluation.judge import identity_digest, parse_judgment
from graynet.evaluation.reporting import summarize
from graynet.execution.experiments import plan_digest


def verify_evaluation(run_dir: Path, record: dict) -> bool:
    """Recheck recorded evaluator artifacts, without asking a model to rescore."""
    evaluation = record["run"].get("evaluation")
    if evaluation is None:  # Historical label-only records.
        return False
    if identity_digest(evaluation["identity"]) != evaluation["identity_sha256"]:
        raise ValueError("evaluation identity digest does not match its recorded settings")
    if evaluation["identity"].get("mode") != evaluation["mode"]:
        raise ValueError("evaluation mode differs from its recorded identity")
    if evaluation["status"] == "not_run":
        if evaluation.get("judgment") is not None or evaluation.get("error") is not None:
            raise ValueError("an unexecuted evaluator cannot contain a verdict or error")
        return False
    if evaluation["mode"] != "semantic" or evaluation["status"] not in ("completed", "error"):
        raise ValueError("unknown evaluation mode or status")
    judge_dir = run_dir / "evaluator/judge"
    saved = json.loads((judge_dir / "evaluation.json").read_text())
    identity = json.loads((judge_dir / "identity.json").read_text())
    if saved != evaluation or identity != evaluation["identity"]:
        raise ValueError("result disagrees with the recorded judge attempt")
    completion = json.loads((judge_dir / "completion.json").read_text())
    if (
        completion["events"] != (judge_dir / "events.jsonl").read_text()
        or completion["stderr"] != (judge_dir / "stderr.txt").read_text()
    ):
        raise ValueError("judge completion differs from its raw event or error log")
    if (
        "egress_log" in completion
        and completion["egress_log"] != (judge_dir / "egress.jsonl").read_text()
    ):
        raise ValueError("judge completion differs from its provider-egress log")
    if evaluation["status"] == "error":
        if not evaluation.get("error") or evaluation.get("judgment") is not None:
            raise ValueError("an evaluator error cannot contain a scored verdict")
        return False
    if evaluation.get("error") is not None:
        raise ValueError("a completed judgment cannot also be an evaluator error")
    dossier = json.loads((judge_dir / "input.json").read_text())
    answer_value = record["run"]["agent_answer"]
    answer = Diagnosis(**{**answer_value, "evidence": tuple(answer_value["evidence"])})
    validate_submission(answer)
    trace = json.loads((run_dir / "agent/tools.json").read_text())
    if not audit_evidence(answer, trace)["passed"]:
        raise ValueError("recorded answer has invalid evidence provenance")
    references = set(answer.evidence)
    observations = [
        {
            key: entry[key]
            for key in (
                "observation_id",
                "tool",
                "arguments",
                "result",
                "timestamp",
                "duration_seconds",
            )
            if key in entry
        }
        for entry in trace
        if entry.get("observation_id") in references
    ]
    if len(observations) != len(references):
        raise ValueError("recorded evidence IDs are not unique")
    expected_input = {
        "reference": record["ground_truth"],
        "environment": (run_dir / "agent/input/environment.md").read_text(),
        "answer": answer_value,
        "observations": observations,
    }
    contract_version = identity.get("input_contract_version", 1)
    if type(contract_version) is not int or contract_version not in (1, 2):
        raise ValueError("unsupported recorded judge input contract")
    if contract_version == 2:
        expected_input["incident"] = (run_dir / "agent/input/incident.md").read_text()
    if dossier != expected_input:
        raise ValueError("judge input differs from the submitted answer or cited observations")
    parsed = parse_judgment(
        completion["text"], observations, rubric_version=identity.get("rubric_version")
    )
    # Convert tuples to their JSON representation before comparing stored data.
    if json.loads(json.dumps(asdict(parsed))) != evaluation["judgment"]:
        raise ValueError("stored verdict differs from the recorded raw model response")
    return True


def verify_planned_evaluation(plan: dict, record: dict) -> None:
    configured = plan.get("evaluation", {"mode": "classification"})
    actual = record["run"].get("evaluation") or {"mode": "classification"}
    if actual["mode"] != configured["mode"]:
        raise ValueError("recorded evaluation mode differs from the frozen experiment")
    if configured["mode"] == "semantic":
        settings = actual.get("identity", {}).get("model", {})
        if any(
            settings.get(field) != configured[field]
            for field in ("model", "reasoning_effort", "timeout_seconds")
        ):
            raise ValueError("recorded judge settings differ from the frozen experiment")


def verify_summary(stored: dict, computed: dict, records: list[dict]) -> None:
    """Validate old label-only aggregates without rewriting their format or evidence."""
    if stored.get("schema_version") == 2:
        if any(stored.get(key) != value for key, value in computed.items()):
            raise ValueError("stored summary differs from artifact-verified records")
        return
    if "schema_version" in stored or any(record["run"].get("evaluation") for record in records):
        raise ValueError("legacy summaries can only describe historical label-only records")

    def matches(expected, actual):
        if isinstance(expected, dict) and isinstance(actual, dict):
            return all(
                key in actual and matches(value, actual[key]) for key, value in expected.items()
            )
        if isinstance(expected, list) and isinstance(actual, list):
            return len(expected) == len(actual) and all(
                matches(left, right) for left, right in zip(expected, actual, strict=True)
            )
        return expected == actual

    # New descriptions and extra fields do not alter any stored metric. Compare
    # every recorded metric, cohort key, and exclusion; ignore prose only.
    for key in ("excluded", "agents"):
        if key not in stored or not matches(stored[key], computed[key]):
            raise ValueError("stored historical summary differs from artifact-verified records")


def check(directory: Path) -> dict:
    directory = directory.resolve()
    plan = json.loads((directory / "experiment.json").read_text())
    if plan["status"] != "finished" or plan["plan_sha256"] != plan_digest(plan):
        raise ValueError("experiment is unfinished or its frozen identity changed")
    records = []
    trajectories = 0
    visibility_checked = 0
    judgments_checked = 0
    for trial in plan["trials"]:
        if trial["status"] != "finished":
            raise ValueError("planned trial was not finalized")
        relative = "runs/" + trial["id"]
        if trial.get("run_directory") != relative:
            raise ValueError("trial directory differs from frozen identity")
        run_dir = directory / relative
        if not run_dir.resolve().is_relative_to(directory):
            raise ValueError("trial path escaped experiment directory")
        if trial.get("case_status") == "preflight_error":
            manifest = json.loads((run_dir / "run.json").read_text())
            if manifest.get("status") != "failed" or (run_dir / "result.json").exists():
                raise ValueError("preflight error has inconsistent artifacts")
            for name, expected in manifest.get("artifacts", {}).items():
                path = run_dir / name
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(run_dir)
                    or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
                ):
                    raise ValueError("preflight artifact does not match manifest")
            continue
        loaded = load_results([run_dir])
        if not loaded:
            if trial.get("case_status") != "preflight_error":
                raise ValueError("finished case is missing its evaluation result")
            continue
        record = loaded[0]
        verify_planned_evaluation(plan, record)
        judgments_checked += verify_evaluation(run_dir, record)
        if record["run"]["case_id"] != trial["case_id"]:
            raise ValueError("run belongs to another planned case")
        metadata = record.get("agent", {})
        if metadata.get("runtime_sha256") != plan["runtime_sha256"]:
            raise ValueError("run runtime differs from frozen experiment")
        if any(
            metadata.get(key) != trial["agent"][key]
            for key in ("driver", "model", "reasoning_effort", "timeout_seconds", "max_tool_calls")
        ):
            raise ValueError("run agent differs from frozen matrix")
        if trial["fault_seed"] is not None:
            evidence = json.loads((run_dir / "evaluator/evidence.json").read_text())
            if evidence.get("injection", {}).get("seed") != trial["fault_seed"]:
                raise ValueError("recorded loss seed differs from frozen matrix")
        trajectory = run_dir / "agent/trajectory.json"
        if trajectory.exists():
            validator = TrajectoryValidator()
            if not validator.validate(trajectory):
                raise ValueError("invalid ATIF: " + "; ".join(validator.get_errors()))
            trajectories += 1
        elif record["run"]["agent_status"] != "not_run":
            raise ValueError("attempted agent is missing its native trajectory")
        tool_trace = run_dir / "agent/tools.json"
        if tool_trace.exists():
            observed = audit_observations(json.loads(tool_trace.read_text()))
            if observed != metadata.get("visibility_audit"):
                raise ValueError(
                    "recorded visibility audit disagrees with actual returned observations"
                )
            if not observed["passed"] and record["run"]["status"] == "valid":
                raise ValueError("exposed private observations were admitted as a valid episode")
            visibility_checked += 1
        records.append(record)
    computed = summarize(records)
    stored = json.loads((directory / "summary.json").read_text())
    # This optional cohort field was added after the first hardened pilot.
    # Missing historical values stay unknown; do not rewrite old summaries.
    for agent in stored.get("agents", []):
        agent.setdefault("environment_runtime_sha256", None)
    verify_summary(stored, computed, records)
    return {
        "planned_trials": len(plan["trials"]),
        "evaluation_records": len(records),
        "valid_atif": trajectories,
        "checked_visibility_audits": visibility_checked,
        "checked_semantic_judgments": judgments_checked,
        "preflight_errors": sum(
            trial.get("case_status") == "preflight_error" for trial in plan["trials"]
        ),
        "excluded": computed["excluded"],
        "scope": "Artifact and recorded-verdict consistency; not independent validation of model judgment quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
