"""Host-side case execution and inspection of evaluator-owned run artifacts."""

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from graynet.agents.registry import DriverConfig, DriverContext, available_drivers, create_driver
from graynet.artifacts.migrations import convert_run_trajectories, import_legacy_run
from graynet.artifacts.records import (
    RunArtifacts,
    load_results,
    write_json,
)
from graynet.scenarios.definitions import CaseFormatError, load_case


def main() -> int:
    parser = argparse.ArgumentParser(prog="graynet")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check-case", help="validate a case manifest and its referenced files"
    )
    check.add_argument("directory")
    case_audit = commands.add_parser(
        "audit-case",
        help="check scenario visibility and answer-contract integrity without deploying",
    )
    case_audit.add_argument("directory")
    plan = commands.add_parser(
        "plan-experiment",
        help="validate and print a reproducible case/agent/repetition plan without executing",
    )
    plan.add_argument("config", type=Path)
    probe_plan = commands.add_parser(
        "plan-probe-experiment", help="check the matched fixed-probe and choice-probe plan"
    )
    probe_plan.add_argument("config", type=Path)
    probe_run = commands.add_parser(
        "run-probe-experiment", help="run the paired flow-probe baseline without a model"
    )
    probe_run.add_argument("config", type=Path)
    probe_run.add_argument("--output", type=Path, required=True)
    probe_run.add_argument("--case-id", action="append", help="include this case; can repeat")
    probe_run.add_argument(
        "--repetition", type=int, help="run one repetition of each selected case"
    )
    batch = commands.add_parser(
        "run-experiment", help="run a frozen serial case/agent/repetition matrix"
    )
    batch.add_argument("config", type=Path)
    batch.add_argument("--output", type=Path, required=True)
    resume = commands.add_parser(
        "resume-experiment", help="continue pending trials without replacing existing attempts"
    )
    resume.add_argument("directory", type=Path)
    for batch_command in (batch, resume):
        batch_command.add_argument(
            "--auth-file",
            type=Path,
            default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json",
        )
    summary = commands.add_parser(
        "summarize",
        help="aggregate result files or run directories without dropping valid agent failures",
    )
    summary.add_argument("runs", type=Path, nargs="+")
    imports = commands.add_parser(
        "import-results",
        help="copy legacy JSON runs into structured directories, leaving originals unchanged",
    )
    imports.add_argument("runs", type=Path, nargs="+")
    imports.add_argument("--root", type=Path, default=Path("diagnostics/imported"))
    convert = commands.add_parser(
        "convert-trajectories",
        help="regenerate ATIF for recorded runs, preserving old derived views",
    )
    convert.add_argument("runs", type=Path, nargs="+")
    convert.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate existing ATIF too, preserving the previous version",
    )
    trajectory_check = commands.add_parser(
        "check-trajectory", help="validate ATIF with the vendored Harbor validator"
    )
    trajectory_check.add_argument("file", type=Path)
    audit = commands.add_parser(
        "audit-agent-isolation",
        help="test the agent sandbox and shortcut rejection without calling a model",
    )
    audit.add_argument("--output", type=Path)
    for command, help_text in (
        ("validate-case", "validate a live Containerlab scenario on Linux"),
        ("run-case", "evaluate an agent on a live network case"),
    ):
        live = commands.add_parser(command, help=help_text)
        live.add_argument("directory")
        live.add_argument(
            "--fault-seed",
            type=int,
            help="override the packet-loss generator seed; only for link_loss cases",
        )
        live.add_argument(
            "--output",
            "--run-dir",
            dest="output",
            type=Path,
            help="new directory for this run (not a JSON filename)",
        )
        if command == "run-case":
            live.add_argument("--agent", choices=available_drivers(), default="codex")
            live.add_argument("--model", help="model ID; omitted uses the driver default")
            live.add_argument(
                "--reasoning-effort", help="reasoning effort; omitted uses the model default"
            )
            live.add_argument("--timeout", type=int, default=300, help="agent deadline in seconds")
            live.add_argument("--max-tool-calls", type=int, default=64)
            live.add_argument(
                "--evaluation",
                choices=["semantic", "classification"],
                default="semantic",
                help="semantic diagnosis judgment (default), or label-only classification metrics",
            )
            live.add_argument(
                "--judge-model",
                help="explicit evaluator model ID, required for semantic evaluation",
            )
            live.add_argument("--judge-reasoning-effort", default="low")
            live.add_argument("--judge-timeout", type=int, default=120)
            live.add_argument(
                "--auth-file",
                type=Path,
                default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json",
            )
    args = parser.parse_args()
    if args.command == "run-case" and args.evaluation == "semantic" and not args.judge_model:
        parser.error(
            "semantic evaluation needs --judge-model; use --evaluation classification for label metrics only"
        )
    if args.command in ("plan-probe-experiment", "run-probe-experiment"):
        from graynet.execution.probe_experiment import plan_probe_experiment, run_probe_experiment

        try:
            probe_report = plan_probe_experiment(args.config)
            if args.command == "run-probe-experiment":
                probe_report = run_probe_experiment(
                    probe_report,
                    args.output,
                    case_ids=set(args.case_id) if args.case_id else None,
                    repetition=args.repetition,
                    progress=lambda text: print(text, flush=True),
                )
        except (ValueError, OSError, RuntimeError, KeyboardInterrupt) as exc:
            parser.exit(
                130 if isinstance(exc, KeyboardInterrupt) else 1,
                f"probe experiment failed: {exc}\n",
            )
        print(json.dumps(probe_report, indent=2))
        return (
            0
            if args.command == "plan-probe-experiment"
            or probe_report["complete_pairs"] == len(probe_report["pairs"])
            else 1
        )
    if args.command in ("plan-experiment", "run-experiment", "resume-experiment"):
        from graynet.execution.experiments import (
            load_experiment,
            plan_experiment,
            resume_experiment,
            run_experiment,
        )

        try:
            if args.command == "resume-experiment":
                report = resume_experiment(
                    args.directory,
                    auth_file=args.auth_file,
                    progress=lambda text: print(text, flush=True),
                )
            else:
                report = plan_experiment(load_experiment(args.config))
                if args.command == "run-experiment":
                    report = run_experiment(
                        args.output,
                        report,
                        auth_file=args.auth_file,
                        progress=lambda text: print(text, flush=True),
                    )
        except (ValueError, OSError, KeyboardInterrupt) as exc:
            parser.exit(
                130 if isinstance(exc, KeyboardInterrupt) else 1, f"experiment failed: {exc}\n"
            )
        print(json.dumps(report, indent=2))
        if args.command == "plan-experiment":
            return 0
        counts = report["experiment"]
        return (
            1
            if counts["preflight_errors"]
            or counts["failed_case_trials"]
            or counts["evaluation_errors"]
            else 0
        )
    if args.command == "check-trajectory":
        from graynet.artifacts.atif.validator import TrajectoryValidator

        validator = TrajectoryValidator()
        if not validator.validate(args.file):
            parser.exit(1, "invalid ATIF: " + "; ".join(validator.get_errors()) + "\n")
        print(f"valid ATIF: {args.file}")
        return 0
    if args.command == "convert-trajectories":
        try:
            for trajectory in convert_run_trajectories(args.runs, refresh=args.refresh):
                print(trajectory)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"trajectory conversion failed: {exc}\n")
        return 0
    if args.command == "import-results":
        try:
            for source in args.runs:
                print(import_legacy_run(source, root=args.root))
        except (ValueError, OSError) as exc:
            parser.exit(1, f"import failed: {exc}\n")
        return 0
    if args.command == "audit-agent-isolation":
        from graynet.runtime.isolation import audit_agent_isolation

        try:
            artifacts = RunArtifacts.create("agent-isolation", args.command, output=args.output)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"cannot create audit directory: {exc}\n")
        print(f"audit directory: {artifacts.directory}", flush=True)
        try:
            artifacts.stage("audit")
            report = audit_agent_isolation()
            artifacts.finish_audit(report)
        except (Exception, KeyboardInterrupt) as exc:
            artifacts.fail(exc)
            parser.exit(130 if isinstance(exc, KeyboardInterrupt) else 1, f"audit failed: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if args.command == "summarize":
        from graynet.evaluation.reporting import summarize

        try:
            records = load_results(args.runs)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"cannot load results: {exc}\n")
        print(json.dumps(summarize(records), indent=2))
        return 0

    try:
        case = load_case(args.directory)
    except (CaseFormatError, OSError) as exc:
        parser.exit(1, f"invalid case: {exc}\n")
    if args.command == "run-case" and case.partial_path is not None:
        parser.error("partial path cases use run-probe-experiment for the flow answer contract")
    if getattr(args, "fault_seed", None) is not None:
        if case.fault["kind"] != "link_loss" or not 0 <= args.fault_seed < 2**32:
            parser.exit(1, "--fault-seed requires link_loss and an unsigned 32-bit seed\n")
        case = replace(case, fault={**case.fault, "seed": args.fault_seed})
    if args.command in ("check-case", "audit-case"):
        if case.network:
            from graynet.environments.containerlab import create_backend

            try:
                backend = create_backend(case)
            except (ValueError, OSError) as exc:
                parser.exit(1, f"invalid network profile: {exc}\n")
        elif args.command == "audit-case":
            parser.exit(1, "audit-case requires a live environment profile\n")
        if args.command == "audit-case":
            print(json.dumps(backend.integrity, indent=2))
            return 0
        print(f"case definition valid: {case.case_id}")
        return 0

    from graynet.environments.containerlab import create_backend
    from graynet.execution.runner import run_case, validate_case

    requested = {
        key: getattr(args, key)
        for key in (
            "agent",
            "model",
            "reasoning_effort",
            "timeout",
            "max_tool_calls",
            "fault_seed",
            "evaluation",
            "judge_model",
            "judge_reasoning_effort",
            "judge_timeout",
        )
        if hasattr(args, key)
    }
    try:
        artifacts = RunArtifacts.create(
            case.case_id, args.command, output=args.output, requested=requested
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"cannot create run directory: {exc}\n")
    print(f"run directory: {artifacts.directory}", flush=True)
    trace = []
    agent_metadata = {}
    driver = None
    backend = None
    try:
        backend = create_backend(case)
        if args.command == "run-case":
            from graynet.evaluation.integrity import runtime_environment_fingerprint
            from graynet.tools.network import NetworkTools

            artifacts.stage("preflight")
            if not case.environment:
                raise ValueError("an agent case needs a public operational environment description")
            tools = NetworkTools(
                case.environment,
                lab_name=backend.LAB_NAME,
                workload_lock=backend.workload_lock,
                policy=backend.network.policy,
                probe=getattr(backend, "probe_for_agent", None),
                reference=getattr(backend, "healthy_reference", None),
            )
            driver = create_driver(
                args.agent,
                DriverContext(
                    tools,
                    args.auth_file,
                    artifacts.agent,
                    runtime_context=lambda: {
                        "environment_runtime_sha256": runtime_environment_fingerprint(
                            backend.evidence
                        )
                    },
                ),
                DriverConfig(args.model, args.reasoning_effort, args.timeout, args.max_tool_calls),
            )
            from graynet.execution.experiments import runtime_fingerprint

            driver.metadata["runtime_sha256"] = runtime_fingerprint()
            driver.metadata["environment_sha256"] = backend.environment_sha256
            driver.preflight()
            evaluator = None
            if args.evaluation == "semantic":
                from graynet.evaluation.codex_model import CodexJudgmentModel
                from graynet.evaluation.judge import SemanticDiagnosisEvaluator

                if not case.ground_truth.diagnosis:
                    raise ValueError("semantic evaluation requires ground_truth.diagnosis")
                judge_model = CodexJudgmentModel(
                    model=args.judge_model,
                    effort=args.judge_reasoning_effort,
                    auth_file=args.auth_file,
                    timeout_seconds=args.judge_timeout,
                )
                judge_model.preflight()
                evaluator = SemanticDiagnosisEvaluator(judge_model, artifacts.evaluator / "judge")
            backend.reserve_agent_window(args.timeout)
            artifacts.stage("evaluation")
            result = run_case(case, backend, driver, evaluator=evaluator)
            trace = driver.trace
            agent_metadata = driver.metadata
        else:
            artifacts.stage("evaluation")
            result = validate_case(case, backend)
        artifacts.finish(
            {
                "run": asdict(result),
                "ground_truth": asdict(case.ground_truth),
                "agent": agent_metadata,
                "evidence": backend.evidence,
            },
            trace=trace if agent_metadata else None,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if driver is not None:
            write_json(artifacts.agent / "metadata.json", driver.metadata)
        if backend is not None:
            write_json(artifacts.evaluator / "evidence.json", backend.evidence)
        artifacts.fail(exc)
        parser.exit(
            130 if isinstance(exc, KeyboardInterrupt) else 1,
            f"run failed during {artifacts.manifest['stage']}: {exc}\nartifacts: {artifacts.directory}\n",
        )
    print(f"case {result.status}: {case.case_id}")
    if result.validation is not None:
        medians = (
            result.validation.baseline_median,
            result.validation.incident_median,
            result.validation.recovery_median,
        )
        if all(value is not None for value in medians):
            print(
                f"baseline={medians[0]:.2f} incident={medians[1]:.2f} recovery={medians[2]:.2f} Mbps"
            )
        if result.validation.reasons:
            print("validation reasons: " + ", ".join(result.validation.reasons))
    if result.error:
        print(f"backend error: {result.error}")
    if args.command == "run-case":
        print(
            f"{agent_metadata.get('driver')} agent: {result.agent_status}, classification={result.classification}"
        )
        if result.agent_error:
            print(f"agent error: {result.agent_error}")
        if result.evaluation is not None and result.evaluation.mode == "semantic":
            print(
                f"semantic evaluation: {result.evaluation.status}, judgment={result.evaluation.judgment}"
            )
            if result.evaluation.error:
                print(f"evaluation error: {result.evaluation.error}")
    for error in result.cleanup_errors:
        print(f"cleanup error: {error}")
    print(f"result: {artifacts.directory / 'result.json'}")
    if (artifacts.agent / "trajectory.md").is_file():
        print(f"trajectory: {artifacts.agent / 'trajectory.md'}")
    return (
        0
        if result.status == "valid"
        and (
            args.command == "validate-case"
            or args.command == "run-case"
            and result.agent_status == "completed"
            and (result.evaluation is None or result.evaluation.status != "error")
        )
        else 1
    )
