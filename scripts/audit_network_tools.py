"""Linux integration regression for every allowed diagnostic view.

Run with the project's Linux Python environment from the repository root:
    python scripts/audit_network_tools.py cases/link_loss --output diagnostics/audits/tools

This checks visibility and usable outputs, not service-impact admission or
semantic diagnosis quality. The normal validate-case/run-case lifecycle checks
service impact separately. No model or credentials are needed here.
"""

import argparse
from pathlib import Path

from graynet.artifacts.records import RunArtifacts
from graynet.environments.containerlab import ContainerlabBackend
from graynet.evaluation.integrity import audit_observations
from graynet.scenarios.definitions import load_case
from graynet.tools.network import FRR_SHOW_COMMANDS, NetworkTools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = load_case(args.case)
    backend = ContainerlabBackend(case)
    artifacts = RunArtifacts.create(case.case_id, "audit-network-tools", output=args.output)
    tools = NetworkTools(
        case.environment,
        lab_name=backend.LAB_NAME,
        workload_lock=backend.workload_lock,
        policy=backend.network.policy,
    )
    trace = []
    checks = {}
    try:
        try:
            backend.deploy(case)
            for phase in ("healthy", "injected"):
                if phase == "injected":
                    backend.inject(case)
                for node, interfaces in tools.policy.interfaces.items():
                    commands = [
                        ["ip", "-j", "addr", "show"],
                        ["ip", "-j", "route", "show"],
                        ["ip", "-j", "-s", "link", "show"],
                        ["ss", "-tin"],
                        ["ip", "route", "get", tools.policy.server_ip],
                        ["ping", "-c", "3", "-W", "1", tools.policy.server_ip],
                    ]
                    for interface in interfaces:
                        commands += [
                            ["ip", "-j", "-s", "link", "show", "dev", interface],
                            ["tc", "-s", "qdisc", "show", "dev", interface],
                        ]
                    if node in tools.policy.routers:
                        commands += [
                            ["vtysh", "-c", command] for command in sorted(FRR_SHOW_COMMANDS)
                        ]
                    for command in commands:
                        observation_id = f"obs-{len(trace) + 1:04d}"
                        result = tools.call("run_command", {"node": node, "argv": command})
                        trace.append(
                            {
                                "observation_id": observation_id,
                                "phase": phase,
                                "status": "completed",
                                "tool": "run_command",
                                "arguments": {"node": node, "argv": command},
                                "result": result,
                            }
                        )
                        checks[f"{phase}:{node}:{' '.join(command)}"] = result["exit_code"] == 0
        finally:
            try:
                backend.recover(case)
            finally:
                backend.teardown(case)
        visibility = audit_observations(trace)
        report = {
            "passed": all(checks.values()) and visibility["passed"],
            "checks": checks,
            "visibility": visibility,
            "observations": trace,
            "scope": "Actual healthy/injected command views; not a service-impact or exhaustive shortcut audit.",
        }
        artifacts.finish_audit(report)
        print(
            f"{sum(checks.values())}/{len(checks)} command views usable; visibility passed={visibility['passed']}"
        )
        return 0 if report["passed"] else 1
    except BaseException as exc:
        artifacts.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
