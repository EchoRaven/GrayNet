"""Evidence-based case admission, independent of any agent's answer."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from graynet.scenarios.definitions import PartialPathSpec, ValidationSpec


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    reasons: tuple[str, ...]
    baseline_median: float | None
    incident_median: float | None
    recovery_median: float | None
    baseline_cv: float | None
    relative_change: float | None
    affected_fraction: float | None
    recovery_deviation: float | None
    temporal: dict | None = None


@dataclass(frozen=True)
class FlowValidation:
    """A measured slowdown and a separate service requirement check."""

    result: ValidationReport
    service_min_mbps: float | None
    service_violation_fraction: float | None
    service_violation: bool | None


def _report(reasons: list[str], **measurements: float | None) -> ValidationReport:
    return ValidationReport(
        valid=not reasons,
        reasons=tuple(reasons),
        baseline_median=measurements.get("baseline_median"),
        incident_median=measurements.get("incident_median"),
        recovery_median=measurements.get("recovery_median"),
        baseline_cv=measurements.get("baseline_cv"),
        relative_change=measurements.get("relative_change"),
        affected_fraction=measurements.get("affected_fraction"),
        recovery_deviation=measurements.get("recovery_deviation"),
    )


def validate_sustained_case(
    spec: ValidationSpec,
    baseline: Sequence[float],
    incident: Sequence[float],
    recovery: Sequence[float] | None = None,
) -> ValidationReport:
    """Check a sustained effect against repeated healthy and recovered samples.

    This validator deliberately does not admit bursty or gradual cases. Those
    need time-window-aware validators rather than a median over the full run.
    """
    if spec.pattern != "sustained":
        raise ValueError("temporal cases cannot use the sustained validator")
    phases = {"baseline": baseline, "incident": incident}
    if recovery is not None:
        phases["recovery"] = recovery
    reasons: list[str] = []
    for phase, samples in phases.items():
        if len(samples) < spec.min_samples:
            reasons.append(f"insufficient_{phase}_samples")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in samples
        ):
            reasons.append(f"invalid_{phase}_samples")
    if reasons:
        return _report(reasons)

    baseline_median = statistics.median(baseline)
    incident_median = statistics.median(incident)
    recovery_median = statistics.median(recovery) if recovery is not None else None
    if baseline_median <= 0:
        return _report(
            ["nonpositive_baseline"],
            baseline_median=baseline_median,
            incident_median=incident_median,
            recovery_median=recovery_median,
        )

    baseline_cv = statistics.pstdev(baseline) / statistics.mean(baseline)
    if baseline_cv > spec.max_baseline_cv:
        reasons.append("unstable_baseline")

    if spec.effect == "degraded":
        if spec.direction == "decrease":
            changes = [(baseline_median - value) / baseline_median for value in incident]
        else:
            changes = [(value - baseline_median) / baseline_median for value in incident]
        relative_change = statistics.median(changes)
        affected_fraction = sum(value >= spec.threshold for value in changes) / len(changes)
        if relative_change < spec.threshold or affected_fraction < spec.min_affected_fraction:
            reasons.append("no_repeatable_degradation")
    else:
        changes = [abs(value - baseline_median) / baseline_median for value in incident]
        relative_change = statistics.median(changes)
        affected_fraction = None
        if relative_change > spec.threshold:
            reasons.append("unexpected_degradation")

    recovery_deviation = None
    if recovery_median is not None:
        recovery_deviation = abs(recovery_median - baseline_median) / baseline_median
        if recovery_deviation > spec.recovery_tolerance:
            reasons.append("recovery_not_healthy")

    return _report(
        reasons,
        baseline_median=baseline_median,
        incident_median=incident_median,
        recovery_median=recovery_median,
        baseline_cv=baseline_cv,
        relative_change=relative_change,
        affected_fraction=affected_fraction,
        recovery_deviation=recovery_deviation,
    )


def validate_partial_paths(
    spec: ValidationSpec,
    selection: PartialPathSpec,
    baseline: dict[str, Sequence[float]],
    incident: dict[str, Sequence[float]],
    recovery: dict[str, Sequence[float]] | None = None,
) -> dict[str, FlowValidation]:
    """Admit the selected severity band only if the other flow stays healthy."""
    if (
        len(baseline) != 2
        or baseline.keys() != incident.keys()
        or (recovery is not None and baseline.keys() != recovery.keys())
        or (selection.flow is not None and selection.flow not in baseline)
    ):
        raise ValueError("partial path validation needs measurements for both named flows")
    reports = {}
    for name, healthy in baseline.items():
        affected = name == selection.flow
        tolerance = spec.recovery_tolerance
        report = validate_sustained_case(
            replace(spec, effect="stable", threshold=math.inf if affected else tolerance),
            healthy,
            incident[name],
            recovery[name] if recovery is not None else None,
        )
        median = report.baseline_median
        if median is None or median <= 0:
            reports[name] = FlowValidation(report, None, None, None)
            continue
        ratios = [(median - value) / median for value in incident[name]]
        slowdown = statistics.median(ratios)
        service_min = median * (1 - spec.threshold)
        violation_fraction = sum(value < service_min for value in incident[name]) / len(ratios)
        violation = report.incident_median < service_min
        reasons = list(report.reasons)
        expected_violation = affected and spec.effect == "degraded"
        repeat_fraction = violation_fraction if expected_violation else 1 - violation_fraction
        if violation != expected_violation or repeat_fraction < spec.min_affected_fraction:
            reasons.append("service_state_not_repeatable")
        if affected:
            if not selection.min_slowdown <= slowdown <= selection.max_slowdown:
                reasons.append("slowdown_outside_band")
            if sum(ratio >= selection.min_slowdown for ratio in ratios) / len(ratios) < (
                spec.min_affected_fraction
            ):
                reasons.append("slowdown_not_repeatable")
        elif sum(abs(ratio) <= tolerance for ratio in ratios) / len(ratios) < (
            spec.min_affected_fraction
        ):
            reasons.append("unaffected_flow_not_stable")
        reports[name] = FlowValidation(
            replace(
                report,
                valid=not reasons,
                reasons=tuple(reasons),
                relative_change=slowdown,
                affected_fraction=repeat_fraction if affected else None,
            ),
            service_min,
            violation_fraction,
            violation,
        )
    return reports


def validate_temporal_case(
    spec: ValidationSpec,
    baseline: Sequence[float],
    records: Sequence[dict],
    timeline: Sequence[dict],
    recovery: Sequence[float] | None = None,
) -> ValidationReport:
    """Match service samples to independently recorded injector intervals.

    Transition-spanning samples cannot establish either an on/off effect or a
    trend. We do not infer the injected state from the observed throughput.
    """
    if spec.pattern not in ("bursty", "gradual"):
        raise ValueError("temporal validation requires a bursty or gradual case")
    sustained = replace(spec, pattern="sustained")
    values = [record.get("throughput_mbps") for record in records]
    # Reuse baseline/data/recovery checks, not the sustained incident criterion.
    base = validate_sustained_case(sustained, baseline, values, recovery)
    reasons = [reason for reason in base.reasons if reason != "no_repeatable_degradation"]
    if (
        any(reason.startswith(("invalid_", "insufficient_")) for reason in reasons)
        or base.baseline_median is None
        or base.baseline_median <= 0
    ):
        return replace(base, valid=False, reasons=tuple(reasons))
    matched = []
    previous_end = None
    for event in timeline:
        left, right, cycle = (
            event.get("started_monotonic"),
            event.get("finished_monotonic"),
            event.get("cycle"),
        )
        if (
            type(left) not in (int, float)
            or type(right) not in (int, float)
            or not math.isfinite(left)
            or not math.isfinite(right)
            or right <= left
            or type(cycle) is not int
            or cycle < 0
        ):
            reasons.append("invalid_fault_timeline")
            continue
        if previous_end is not None and left < previous_end:
            reasons.append("overlapping_fault_intervals")
        previous_end = right
    previous_sample_end = None
    for record in records:
        start, end = record.get("started_monotonic"), record.get("finished_monotonic")
        if (
            type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
        ):
            reasons.append("invalid_sample_timestamps")
            continue
        if previous_sample_end is not None and start < previous_sample_end:
            reasons.append("overlapping_service_samples")
        previous_sample_end = end
        overlaps = []
        for event in timeline:
            left, right = event.get("started_monotonic"), event.get("finished_monotonic")
            if (
                not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not math.isfinite(left)
                or not math.isfinite(right)
                or right <= left
            ):
                reasons.append("invalid_fault_timeline")
                continue
            overlap = max(0, min(end, right) - max(start, left)) / (end - start)
            if overlap >= 0.8:
                overlaps.append(event)
        if len(overlaps) > 1:
            reasons.append("overlapping_fault_intervals")
        elif len(overlaps) == 1:
            matched.append((overlaps[0], record["throughput_mbps"]))
    grouped: dict[float, list[float]] = {}
    for event, value in matched:
        level = event.get("rate_mbps")
        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float))
            or not math.isfinite(level)
            or level < 0
        ):
            reasons.append("invalid_fault_level")
            continue
        grouped.setdefault(level, []).append(value)
    medians = {str(level): statistics.median(samples) for level, samples in grouped.items()}
    if spec.pattern == "bursty":
        on = [value for event, value in matched if event.get("rate_mbps", 0) > 0]
        off = grouped.get(0, [])
        on_cycles = {event.get("cycle") for event, _ in matched if event.get("rate_mbps", 0) > 0}
        off_cycles = {event.get("cycle") for event, _ in matched if event.get("rate_mbps", 0) == 0}
        if (
            len(on) < spec.min_state_samples
            or len(off) < spec.min_state_samples
            or len(on_cycles & off_cycles) < spec.min_cycles
        ):
            reasons.append("insufficient_on_off_coverage")
        if on:
            effect = sum(
                (base.baseline_median - value) / base.baseline_median >= spec.threshold
                for value in on
            ) / len(on)
            if effect < spec.min_affected_fraction:
                reasons.append("on_windows_not_degraded")
        if (
            off
            and sum(
                abs(value - base.baseline_median) / base.baseline_median <= spec.recovery_tolerance
                for value in off
            )
            / len(off)
            < spec.min_affected_fraction
        ):
            reasons.append("off_windows_not_healthy")
    else:
        levels = sorted(grouped)
        if len(levels) < 3 or any(len(grouped[level]) < spec.min_state_samples for level in levels):
            reasons.append("insufficient_ramp_coverage")
        if len(levels) >= 3:
            med = [statistics.median(grouped[level]) for level in levels]
            if any(
                second > first + base.baseline_median * 0.05
                for first, second in zip(med, med[1:], strict=False)
            ):
                reasons.append("not_a_worsening_trend")
            if (med[0] - med[-1]) / base.baseline_median < spec.min_trend_change:
                reasons.append("ramp_change_too_small")
            if (base.baseline_median - med[-1]) / base.baseline_median < spec.threshold:
                reasons.append("final_ramp_level_not_degraded")
    return replace(
        base,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        temporal={
            "pattern": spec.pattern,
            "matched_samples": len(matched),
            "transition_samples": len(records) - len(matched),
            "level_medians": medians,
        },
    )
