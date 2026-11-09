"""Host-side case definitions. Never mount these files into an agent container."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from graynet.paths import REPOSITORY_ROOT


class CaseFormatError(ValueError):
    """A case definition is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class GroundTruth:
    anomaly: bool
    component: str | None
    mechanism: str | None
    component_aliases: tuple[str, ...] = ()
    mechanism_aliases: tuple[str, ...] = ()
    diagnosis: str = ""


@dataclass(frozen=True)
class ValidationSpec:
    metric: str
    effect: Literal["degraded", "stable"]
    direction: Literal["increase", "decrease"] | None
    threshold: float
    min_samples: int
    max_baseline_cv: float
    recovery_tolerance: float
    min_affected_fraction: float
    pattern: Literal["sustained", "bursty", "gradual"] = "sustained"
    min_state_samples: int = 2
    min_cycles: int = 2
    min_trend_change: float = 0.15


@dataclass(frozen=True)
class PartialPathSpec:
    """Private flow selection and expected measured slowdown range."""

    flow: str | None
    min_slowdown: float
    max_slowdown: float


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    title: str
    directory: Path
    topology: Path
    incident: str
    incident_path: Path
    workload: dict[str, Any]
    fault: dict[str, Any]
    ground_truth: GroundTruth
    validation: ValidationSpec
    environment: str = ""
    network: str | None = None
    environment_path: Path | None = None
    partial_path: PartialPathSpec | None = None
    benign_fault: bool = False


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaseFormatError(f"{field} must be a mapping")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseFormatError(f"{field} must be a nonempty string")
    return value.strip()


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaseFormatError(f"{field} must be a number")
    result = float(value)
    if not 0 <= result < float("inf") or (positive and result == 0):
        raise CaseFormatError(
            f"{field} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _allowed_keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CaseFormatError(f"{field} field names must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise CaseFormatError(f"unknown {field} field(s): {', '.join(sorted(unknown))}")


def load_case(directory: Path | str) -> CaseDefinition:
    """Read a private manifest and its shared or explicitly overridden incident report."""
    case_dir = Path(directory).resolve()
    manifest = case_dir / "scenario.yaml"
    if not manifest.is_file():
        raise CaseFormatError("case directory must contain scenario.yaml")
    if not manifest.resolve().is_relative_to(case_dir):
        raise CaseFormatError("scenario.yaml must stay inside the case directory")
    try:
        raw = _object(yaml.safe_load(manifest.read_text(encoding="utf-8")), "scenario")
    except yaml.YAMLError as exc:
        raise CaseFormatError(f"invalid YAML in {manifest}: {exc}") from exc
    _allowed_keys(
        raw,
        {
            "schema_version",
            "id",
            "title",
            "topology",
            "workload",
            "fault",
            "ground_truth",
            "validation",
            "environment",
            "network",
            "incident",
            "partial_path",
            "benign_fault",
        },
        "scenario",
    )
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise CaseFormatError("schema_version must be 1")

    case_id = _text(raw.get("id"), "id")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id) is None:
        raise CaseFormatError("id must be a lowercase slug")
    title = _text(raw.get("title"), "title")
    network = raw.get("network")
    if network is not None and (
        not isinstance(network, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", network) is None
    ):
        raise CaseFormatError("network must be an environment profile slug")
    topology_name = Path(_text(raw.get("topology"), "topology"))
    if topology_name.is_absolute():
        raise CaseFormatError("topology must be a path inside the case directory")
    topology = (case_dir / topology_name).resolve()
    shared = REPOSITORY_ROOT / "environments"
    if not topology.is_relative_to(case_dir) and not topology.is_relative_to(shared):
        raise CaseFormatError(
            "topology must be inside the case directory or repository environments/"
        )
    if not topology.is_file():
        raise CaseFormatError(f"topology does not exist: {topology}")
    if "incident" in raw:
        incident_name = Path(_text(raw["incident"], "incident"))
        if incident_name.is_absolute():
            raise CaseFormatError(
                "incident must be a relative path inside the case directory or repository environments/"
            )
        incident_path = (case_dir / incident_name).resolve()
    elif network is not None:
        incident_path = (shared / network / "incident.md").resolve()
        if (case_dir / "incident.md").exists() or (case_dir / "incident.md").is_symlink():
            raise CaseFormatError(
                "a case-specific incident.md needs an explicit incident: incident.md reference"
            )
    else:
        incident_path = (case_dir / "incident.md").resolve()
    if not incident_path.is_relative_to(case_dir) and not incident_path.is_relative_to(shared):
        raise CaseFormatError(
            "incident must be inside the case directory or repository environments/"
        )
    if not incident_path.is_file():
        raise CaseFormatError(f"incident report does not exist: {incident_path}")
    incident = _text(incident_path.read_text(encoding="utf-8"), "incident report")
    environment = ""
    environment_path = None
    if "environment" in raw:
        environment_name = Path(_text(raw["environment"], "environment"))
        if environment_name.is_absolute():
            raise CaseFormatError("environment must be a relative path")
        environment_path = (case_dir / environment_name).resolve()
    elif network is not None:
        environment_path = (shared / network / "environment.md").resolve()
        if (case_dir / "environment.md").exists() or (case_dir / "environment.md").is_symlink():
            raise CaseFormatError(
                "a case-specific environment.md needs an explicit environment: reference"
            )
    if environment_path is not None:
        if (
            not (
                environment_path.is_relative_to(case_dir) or environment_path.is_relative_to(shared)
            )
            or not environment_path.is_file()
        ):
            raise CaseFormatError(
                "environment must be a file inside the case directory or repository environments/"
            )
        environment = _text(environment_path.read_text(encoding="utf-8"), "environment")

    workload = _object(raw.get("workload"), "workload")
    fault = _object(raw.get("fault"), "fault")
    _text(workload.get("kind"), "workload.kind")
    fault_kind = _text(fault.get("kind"), "fault.kind")
    partial_path = None
    if "partial_path" in raw:
        partial = _object(raw["partial_path"], "partial_path")
        _allowed_keys(partial, {"flow", "min_slowdown", "max_slowdown"}, "partial_path")
        flow = partial.get("flow")
        if flow is not None:
            flow = _text(flow, "partial_path.flow")
        if fault_kind != ("none" if flow is None else "link_loss"):
            raise CaseFormatError("partial paths require a healthy control or steady link loss")
        low = _number(partial.get("min_slowdown", 0), "partial_path.min_slowdown")
        high = _number(partial.get("max_slowdown", 0), "partial_path.max_slowdown")
        if (flow is None and (low or high)) or (flow is not None and not 0 < low < high < 1):
            raise CaseFormatError("invalid partial path slowdown range")
        partial_path = PartialPathSpec(flow, low, high)

    benign_fault = raw.get("benign_fault", False)
    if not isinstance(benign_fault, bool):
        raise CaseFormatError("benign_fault must be a boolean")

    expected = _object(raw.get("ground_truth"), "ground_truth")
    _allowed_keys(
        expected,
        {
            "anomaly",
            "component",
            "mechanism",
            "component_aliases",
            "mechanism_aliases",
            "diagnosis",
        },
        "ground_truth",
    )
    anomaly = expected.get("anomaly")
    if not isinstance(anomaly, bool):
        raise CaseFormatError("ground_truth.anomaly must be a boolean")
    component = expected.get("component")
    mechanism = expected.get("mechanism")
    if anomaly:
        component = _text(component, "ground_truth.component")
        mechanism = _text(mechanism, "ground_truth.mechanism")
        if fault_kind == "none":
            raise CaseFormatError("an anomalous case needs a fault")
    elif (
        component is not None
        or mechanism is not None
        or (fault_kind != "none" and partial_path is None and not benign_fault)
    ):
        raise CaseFormatError(
            "a healthy control needs fault.kind: none and no component or mechanism"
        )
    if benign_fault and (anomaly or fault_kind == "none" or partial_path is not None):
        raise CaseFormatError("benign_fault needs a healthy two-path case with an active fault")
    aliases = {}
    for field in ("component_aliases", "mechanism_aliases"):
        values = expected.get(field, [])
        if not isinstance(values, list) or (values and not anomaly):
            raise CaseFormatError(f"ground_truth.{field} must be a list for an anomalous case")
        aliases[field] = tuple(_text(value, f"ground_truth.{field}") for value in values)

    validation = _object(raw.get("validation"), "validation")
    _allowed_keys(
        validation,
        {
            "metric",
            "effect",
            "direction",
            "threshold",
            "min_samples",
            "max_baseline_cv",
            "recovery_tolerance",
            "min_affected_fraction",
            "pattern",
            "min_state_samples",
            "min_cycles",
            "min_trend_change",
        },
        "validation",
    )
    effect = validation.get("effect")
    if effect not in ("degraded", "stable"):
        raise CaseFormatError("validation.effect must be degraded or stable")
    if (effect == "degraded") != anomaly:
        raise CaseFormatError("validation.effect must agree with ground_truth.anomaly")
    direction = validation.get("direction")
    if effect == "degraded" and direction not in ("increase", "decrease"):
        raise CaseFormatError("degraded cases need validation.direction: increase or decrease")
    if effect == "stable" and direction is not None:
        raise CaseFormatError("healthy controls must omit validation.direction")
    threshold = _number(
        validation.get("threshold"), "validation.threshold", positive=effect == "degraded"
    )
    if partial_path is not None and (
        not 0 < threshold < 1
        or partial_path.flow is not None
        and (
            (anomaly and partial_path.min_slowdown <= threshold)
            or (not anomaly and partial_path.max_slowdown >= threshold)
        )
    ):
        raise CaseFormatError("the partial path slowdown band must agree with the service limit")
    samples = validation.get("min_samples", 3)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise CaseFormatError("validation.min_samples must be an integer of at least 3")
    max_cv = _number(validation.get("max_baseline_cv", 0.1), "validation.max_baseline_cv")
    tolerance = _number(validation.get("recovery_tolerance", 0.1), "validation.recovery_tolerance")
    affected = _number(
        validation.get("min_affected_fraction", 0.8), "validation.min_affected_fraction"
    )
    if affected > 1:
        raise CaseFormatError("validation.min_affected_fraction cannot exceed 1")
    pattern = validation.get("pattern", "sustained")
    if pattern not in ("sustained", "bursty", "gradual"):
        raise CaseFormatError("validation.pattern must be sustained, bursty, or gradual")
    fault_pattern = fault.get("pattern", {})
    if not isinstance(fault_pattern, dict) or fault_pattern.get("kind", "sustained") != pattern:
        raise CaseFormatError("fault and validation time patterns must agree")
    if not anomaly and pattern != "sustained":
        raise CaseFormatError("healthy controls must use sustained validation")
    if partial_path is not None and (pattern != "sustained" or samples != 5):
        raise CaseFormatError("partial path cases require sustained validation and five samples")
    counts = {}
    for field in ("min_state_samples", "min_cycles"):
        value = validation.get(field, 2)
        if type(value) is not int or value < 2:
            raise CaseFormatError(f"validation.{field} must be an integer of at least 2")
        counts[field] = value
    trend = _number(
        validation.get("min_trend_change", 0.15), "validation.min_trend_change", positive=True
    )

    return CaseDefinition(
        case_id=case_id,
        title=title,
        directory=case_dir,
        topology=topology,
        incident=incident,
        incident_path=incident_path,
        workload=dict(workload),
        fault=dict(fault),
        ground_truth=GroundTruth(
            anomaly,
            component,
            mechanism,
            **aliases,
            diagnosis=_text(expected["diagnosis"], "ground_truth.diagnosis")
            if "diagnosis" in expected
            else "",
        ),
        validation=ValidationSpec(
            metric=_text(validation.get("metric"), "validation.metric"),
            effect=effect,
            direction=direction,
            threshold=threshold,
            min_samples=samples,
            max_baseline_cv=max_cv,
            recovery_tolerance=tolerance,
            min_affected_fraction=affected,
            pattern=pattern,
            min_trend_change=trend,
            **counts,
        ),
        environment=environment,
        network=network,
        environment_path=environment_path,
        partial_path=partial_path,
        benign_fault=benign_fault,
    )
