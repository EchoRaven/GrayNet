"""Repeat incident workloads without an agent to check measurement reliability.

Healthy and recovery phases keep their normal sample counts. The incident gets
the explicitly requested count; all original admission thresholds still apply.
This records a validation run, not a model-performance episode. No failed
workloads are retried or replaced.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from graynet.artifacts.records import RunArtifacts
from graynet.environments.containerlab import ContainerlabBackend
from graynet.execution.runner import validate_case
from graynet.scenarios.definitions import load_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.samples <= 100:
        parser.error("samples must be between 5 and 100")
    case = load_case(args.case)
    if case.validation.pattern != "sustained":
        parser.error("use the normal temporal-case lifecycle for scheduled faults")

    class StressBackend(ContainerlabBackend):
        def measure(self, case, phase):
            if phase == "incident":
                case = replace(case, validation=replace(case.validation, min_samples=args.samples))
            return super().measure(case, phase)

    backend = StressBackend(case)
    backend.reserve_agent_window(args.samples * (backend.duration + 40))
    artifacts = RunArtifacts.create(
        case.case_id,
        "stress-workloads",
        output=args.output,
        requested={"incident_samples": args.samples, "agent": None},
    )
    try:
        result = validate_case(case, backend)
        artifacts.finish(
            {
                "run": asdict(result),
                "ground_truth": asdict(case.ground_truth),
                "agent": {},
                "evidence": backend.evidence,
            }
        )
        waits = [
            record["wait_seconds"] for record in backend.evidence.get("workload_readiness", [])
        ]
        print(
            f"case {result.status}; completed incident samples: {len(result.incident)}/{args.samples}"
        )
        print(f"server readiness: {len(waits)} checks; longest wait {max(waits, default=0):.3f}s")
        if result.error:
            print(result.error)
        return 0 if result.status == "valid" else 1
    except BaseException as exc:
        artifacts.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
