from dataclasses import replace
from pathlib import Path

import pytest

from graynet.evaluation.validation import validate_sustained_case, validate_temporal_case
from graynet.scenarios.definitions import load_case

SPEC = replace(
    load_case(Path(__file__).parent / "fixtures/sustained_congestion").validation, min_samples=6
)
BASE = [100, 101, 99, 100, 100, 100]


def samples(values):
    return [
        {"started_monotonic": i * 5 + 1, "finished_monotonic": i * 5 + 4, "throughput_mbps": value}
        for i, value in enumerate(values)
    ]


def burst_timeline():
    return [
        {
            "started_monotonic": i * 5,
            "finished_monotonic": i * 5 + 5,
            "rate_mbps": 35 if i % 2 == 0 else 0,
            "cycle": i // 2,
        }
        for i in range(6)
    ]


def test_burst_validation_uses_on_off_intervals_not_full_run_median():
    spec = replace(SPEC, pattern="bursty")
    result = validate_temporal_case(
        spec, BASE, samples([20, 100, 20, 100, 20, 100]), burst_timeline(), BASE
    )
    assert result.valid and result.temporal["matched_samples"] == 6
    with pytest.raises(ValueError, match="sustained"):
        validate_sustained_case(spec, BASE, [20, 100] * 3)


def test_bursty_all_bad_or_all_good_is_not_a_valid_burst_case():
    spec = replace(SPEC, pattern="bursty")
    assert (
        "off_windows_not_healthy"
        in validate_temporal_case(spec, BASE, samples([20] * 6), burst_timeline()).reasons
    )
    assert (
        "on_windows_not_degraded"
        in validate_temporal_case(spec, BASE, samples([100] * 6), burst_timeline()).reasons
    )


def test_probe_aliasing_is_rejected_not_mistaken_for_a_valid_burst():
    records = samples([20] * 6)
    timeline = [{"started_monotonic": 0, "finished_monotonic": 40, "rate_mbps": 35, "cycle": 0}]
    assert (
        "insufficient_on_off_coverage"
        in validate_temporal_case(replace(SPEC, pattern="bursty"), BASE, records, timeline).reasons
    )


def test_gradual_cases_require_a_measured_worsening_trend():
    timeline = [
        {
            "started_monotonic": i * 10,
            "finished_monotonic": i * 10 + 10,
            "rate_mbps": rate,
            "cycle": 0,
        }
        for i, rate in enumerate([10, 20, 35])
    ]
    spec = replace(SPEC, pattern="gradual")
    assert validate_temporal_case(
        spec, BASE, samples([90, 89, 70, 71, 30, 31]), timeline, BASE
    ).valid
    assert (
        "ramp_change_too_small"
        in validate_temporal_case(spec, BASE, samples([30] * 6), timeline).reasons
    )
    assert (
        "not_a_worsening_trend"
        in validate_temporal_case(spec, BASE, samples([50, 51, 90, 91, 20, 21]), timeline).reasons
    )


@pytest.mark.parametrize("mutation", ["overlap", "timestamps", "duplicate", "bad_cycle"])
def test_temporal_ground_truth_rejects_bad_or_overlapping_intervals(mutation):
    timeline = burst_timeline()
    records = samples([20, 100] * 3)
    if mutation == "overlap":
        timeline[1]["started_monotonic"] = 4
    elif mutation == "timestamps":
        records[0]["finished_monotonic"] = float("nan")
    elif mutation == "duplicate":
        records[1] = records[0]
    else:
        timeline[0]["cycle"] = True
    assert not validate_temporal_case(
        replace(SPEC, pattern="bursty"), BASE, records, timeline
    ).valid


def test_temporal_cases_still_require_healthy_recovery_and_stable_baselines():
    spec = replace(SPEC, pattern="bursty")
    records = samples([20, 100] * 3)
    assert (
        "recovery_not_healthy"
        in validate_temporal_case(spec, BASE, records, burst_timeline(), [20] * 6).reasons
    )
    assert (
        "unstable_baseline"
        in validate_temporal_case(spec, [50, 150] * 3, records, burst_timeline()).reasons
    )
