"""Predefined visibility and evidence-provenance checks, not an LLM judge."""

import hashlib
import json
import re
from os.path import relpath
from typing import Any

from graynet.agents.base import Diagnosis, FlowDiagnosis
from graynet.environments.network import NetworkSpec
from graynet.scenarios.definitions import CaseDefinition

MECHANISMS = (
    "congestion",
    "packet_loss",
    "delay",
    "endpoint_overload",
    "capacity_limit",
    "mtu_mismatch",
    "routing",
    "multiple",
    "unknown",
)


def audit_observations(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag known simulator/private markers in returned observations only.

    Command arguments and inventory are excluded: an operational command may
    legitimately contain 'qdisc'. This is a regression guard, not a proof that
    an agent cannot recognize software emulation or exploit another shortcut.
    """
    markers = {
        "emulator_queue": r"\b(?:netem|tbf)\b",
        "lab_identity": r"clab-",
        "private_path": r"/(?:proc/|sys/fs/cgroup/|etc/frr/)",
        "namespace_metadata": r"\b(?:link_netnsid|parentIfindex|altnames)\b",
    }
    findings = []
    for entry in trace:
        if entry.get("status") != "completed" or entry.get("tool") == "network_inventory":
            continue
        public = json.dumps(entry.get("result", {}))
        for label, pattern in markers.items():
            if re.search(pattern, public, re.IGNORECASE):
                findings.append({"observation_id": entry.get("observation_id"), "marker": label})
    return {
        "passed": not findings,
        "findings": findings,
        "scope": "Known simulator/private markers in returned observations; not an exhaustive shortcut audit.",
    }


def audit_case(case: CaseDefinition, network: NetworkSpec) -> dict[str, Any]:
    public = case.incident + "\n" + case.environment
    truth = case.ground_truth
    components = {label.casefold() for label in network.policy.components}
    mechanisms = {label.casefold() for label in MECHANISMS}
    checks = {
        "public_input_has_no_case_id": case.case_id.casefold() not in public.casefold(),
        "public_input_has_no_private_title": case.title.casefold() not in public.casefold(),
        "public_input_has_no_private_manifest": not any(
            token in public
            for token in (
                "ground_truth:",
                "loss_percent:",
                "rates_mbps:",
                "scenario.yaml",
                "answer key:",
            )
        ),
        "controllers_not_managed": not {
            network.background.get("client"),
            network.background.get("server"),
        }
        & set(network.policy.interfaces),
        "reference_component_is_submittable": not truth.anomaly
        or any(
            label.casefold() in components for label in (truth.component, *truth.component_aliases)
        ),
        "reference_mechanism_is_submittable": not truth.anomaly
        or any(
            label.casefold() in mechanisms for label in (truth.mechanism, *truth.mechanism_aliases)
        ),
        "workload_endpoints_match_policy": case.workload["client"] == network.policy.client
        and case.workload["server_ip"] == network.policy.server_ip,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scope": "Static visibility and answer-contract checks; not proof of fidelity or semantic diagnosis quality.",
    }


def environment_fingerprint(case: CaseDefinition, network: NetworkSpec) -> str:
    # Excludes the selected fault and private reference so matched cases share
    # an environment cohort. Any operational/workload/config change separates it.
    value = {
        "profile": network.path.read_text(),
        "topology": case.topology.read_text(),
        "configs": {
            relpath(path, network.path.parent): path.read_text() for path in network.bound_files
        },
        "workload": case.workload,
        "incident": case.incident,
        "environment": case.environment,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_environment_fingerprint(evidence: dict[str, Any]) -> str:
    runtime = evidence.get("runtime", {})
    value = {
        "images": evidence.get("images", {}),
        "endpoint_resources": evidence.get("endpoint_resources", {}),
        "runtime": {
            key: runtime.get(key) for key in ("kernel", "architecture", "containerlab", "docker")
        },
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_evidence(
    answer: Diagnosis | FlowDiagnosis, trace: list[dict[str, Any]]
) -> dict[str, Any]:
    observed = {
        entry.get("observation_id"): entry
        for entry in trace
        if entry.get("status") == "completed"
        and entry.get("tool") != "network_inventory"
        and entry.get("result", {}).get("exit_code", 0) == 0
    }
    references = list(answer.evidence)
    invalid = [
        reference
        for reference in references
        if not re.fullmatch(r"obs-[0-9]{4,}", reference) or reference not in observed
    ]
    duplicate = len(set(references)) != len(references)
    current = any(
        reference in observed and observed[reference]["tool"] != "healthy_reference"
        for reference in references
    )
    return {
        "passed": bool(references) and not invalid and not duplicate and current,
        "references": references,
        "invalid_references": invalid,
        "duplicate_references": duplicate,
        "observed_tools": [
            observed[reference]["tool"] for reference in references if reference in observed
        ],
        "scope": "Confirms cited observations exist; does not grade whether the explanation follows from them.",
    }
