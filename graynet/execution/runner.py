"""Case lifecycle. The network backend is intentionally a replaceable boundary."""

from dataclasses import dataclass, replace
from typing import Literal

from graynet.agents.base import (
    Agent,
    Diagnosis,
    FlowDiagnosis,
    validate_flow_submission,
    validate_submission,
)
from graynet.environments.base import NetworkBackend
from graynet.evaluation.classification import ClassificationScore, score_classification
from graynet.evaluation.integrity import audit_evidence, audit_observations
from graynet.evaluation.judge import (
    EvaluationResult,
    SemanticDiagnosisEvaluator,
    classification_evaluation,
)
from graynet.evaluation.validation import ValidationReport, validate_sustained_case
from graynet.scenarios.definitions import CaseDefinition


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    status: Literal["valid", "invalid", "error"]
    baseline: tuple[float, ...]
    incident: tuple[float, ...]
    recovery: tuple[float, ...]
    validation: ValidationReport | None
    error: str | None
    cleanup_errors: tuple[str, ...]
    agent_status: Literal["not_run", "completed", "error"]
    agent_answer: Diagnosis | FlowDiagnosis | None
    agent_error: str | None
    classification: ClassificationScore | None
    evaluation: EvaluationResult | None = None


def run_case(
    case: CaseDefinition,
    backend: NetworkBackend,
    agent: Agent | None = None,
    *,
    evaluator: SemanticDiagnosisEvaluator | None = None,
) -> CaseRun:
    """Run a case, admitting it independently of the agent's answer.

    A partially completed deploy or injection still triggers cleanup. Backend
    failures are reported as case errors. Agent errors do not invalidate a case.
    """
    baseline: tuple[float, ...] = ()
    incident: tuple[float, ...] = ()
    recovery: tuple[float, ...] = ()
    report: ValidationReport | None = None
    window_report: ValidationReport | None = None
    error: str | None = None
    cleanup_errors: list[str] = []
    agent_status: Literal["not_run", "completed", "error"] = "not_run"
    agent_answer: Diagnosis | FlowDiagnosis | None = None
    agent_error: str | None = None
    deployment_attempted = False
    injection_attempted = False
    recovered = False
    validate = getattr(
        backend,
        "validate",
        lambda baseline, incident, recovery=None: validate_sustained_case(
            case.validation, baseline, incident, recovery
        ),
    )
    if case.partial_path is not None and evaluator is not None:
        raise ValueError("partial path cases need the flow diagnosis evaluator")

    try:
        deployment_attempted = True
        backend.deploy(case)
        baseline = tuple(backend.measure(case, "baseline"))
        injection_attempted = True
        backend.inject(case)
        incident = tuple(backend.measure(case, "incident"))
        active_report = validate(baseline, incident)
        if agent is not None and active_report.valid:
            start_window = getattr(backend, "start_agent_window", None)
            finish_window = getattr(backend, "finish_agent_window", None)
            if start_window is not None:
                start_window(case)
            try:
                agent_answer = agent.diagnose(case.incident)
                if case.partial_path is None:
                    validate_submission(agent_answer)
                else:
                    validate_flow_submission(agent_answer)
                if not audit_evidence(agent_answer, getattr(agent, "trace", []))["passed"]:
                    raise ValueError(
                        "agent cites missing, duplicate, or non-network observation IDs"
                    )
                agent_status = "completed"
            except Exception as exc:
                agent_status = "error"
                agent_error = f"{type(exc).__name__}: {exc}"
            finally:
                if finish_window is not None:
                    checked = finish_window(case, baseline)
                    if isinstance(checked, ValidationReport):
                        window_report = checked
        backend.recover(case)
        recovered = True
        recovery = tuple(backend.measure(case, "recovery"))
        report = validate(baseline, incident, recovery)
        if window_report is not None and not window_report.valid:
            report = replace(
                report,
                valid=False,
                reasons=(
                    *report.reasons,
                    *("agent_window:" + reason for reason in window_report.reasons),
                ),
            )
        if agent is not None and not audit_observations(getattr(agent, "trace", []))["passed"]:
            report = replace(
                report, valid=False, reasons=(*report.reasons, "private_artifact_exposed")
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if injection_attempted and not recovered:
            try:
                backend.recover(case)
            except Exception as exc:
                cleanup_errors.append(f"recover: {type(exc).__name__}: {exc}")
        if deployment_attempted:
            try:
                backend.teardown(case)
            except Exception as exc:
                cleanup_errors.append(f"teardown: {type(exc).__name__}: {exc}")

    status: Literal["valid", "invalid", "error"]
    if error or cleanup_errors:
        status = "error"
    elif report is not None and report.valid:
        status = "valid"
    else:
        status = "invalid"
    classification = None
    evaluation = None
    if evaluator is not None:
        evaluation = EvaluationResult(
            "semantic", "not_run", evaluator.identity, evaluator.identity_sha256
        )
    elif case.partial_path is None:
        evaluation = classification_evaluation()
    if status == "valid" and agent_status == "completed" and isinstance(agent_answer, Diagnosis):
        try:
            classification = score_classification(True, case.ground_truth, agent_answer)
        except ValueError as exc:
            agent_status = "error"
            agent_error = f"ValueError: {exc}"
        if agent_status == "completed" and evaluator is not None:
            evaluation = evaluator.evaluate(
                case.ground_truth,
                agent_answer,
                getattr(agent, "trace", []),
                incident=case.incident,
                environment=case.environment,
            )
    return CaseRun(
        case_id=case.case_id,
        status=status,
        baseline=baseline,
        incident=incident,
        recovery=recovery,
        validation=report,
        error=error,
        cleanup_errors=tuple(cleanup_errors),
        agent_status=agent_status,
        agent_answer=agent_answer,
        agent_error=agent_error,
        classification=classification,
        evaluation=evaluation,
    )


def validate_case(case: CaseDefinition, backend: NetworkBackend) -> CaseRun:
    """Run the same lifecycle without an agent, to validate a new case."""
    return run_case(case, backend)
