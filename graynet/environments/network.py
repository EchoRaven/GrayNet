"""Topology-derived diagnostic policy and private environment checks.

Profiles describe operational roles, not the answer to any incident. Fault
definitions never determine the tools or the public answer vocabulary.
"""

import re
from dataclasses import dataclass
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import yaml

from graynet.paths import REPOSITORY_ROOT
from graynet.scenarios.definitions import CaseDefinition


@dataclass(frozen=True)
class DiagnosticPolicy:
    interfaces: dict[str, tuple[str, ...]]
    routers: frozenset[str]
    addresses: frozenset[str]
    client: str
    server_ip: str
    flow_names: tuple[str, ...] = ()

    @property
    def components(self) -> list[str]:
        return [
            *self.interfaces,
            *[
                f"{node}:{interface}"
                for node, interfaces in self.interfaces.items()
                for interface in interfaces
            ],
        ]


@dataclass(frozen=True)
class ServiceFlow:
    """A service endpoint pair and its expected inter-router link."""

    client: str
    client_ip: str
    server: str
    server_ip: str
    sender: tuple[str, str]
    receiver: tuple[str, str]


@dataclass(frozen=True)
class NetworkSpec:
    path: Path
    topology: dict[str, Any]
    lab_name: str
    nodes: tuple[str, ...]
    policy: DiagnosticPolicy
    bgp_peers: dict[str, tuple[str, ...]]
    routes: dict[str, dict[str, str]]
    queue: dict[str, str]
    background: dict[str, str]
    bound_files: tuple[Path, ...]
    data_interfaces: dict[str, tuple[str, ...]]
    mtu: int
    endpoint_cpu_affinity: str = "unrestricted"
    flows: dict[str, ServiceFlow] | None = None
    service_min_fraction: float | None = None
    noise: dict[str, Any] | None = None


def _slug(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value) is None:
        raise ValueError(f"invalid {field}")
    return value


def _address(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("network addresses must be IPv4 strings")
    IPv4Address(value)
    return value


def _load_flows(
    profile: dict[str, Any],
    managed: dict[str, str],
    addresses: frozenset[str],
    routes: dict[str, dict[str, str]],
    links: list[dict[str, list[str]]],
) -> dict[str, ServiceFlow]:
    flows = profile["flows"]
    if not isinstance(flows, dict) or len(flows) != 2:
        raise ValueError("a partial path environment needs exactly two named flows")
    result = {}
    pairs = [tuple(link["endpoints"]) for link in links]
    peers = dict(pairs + [(right, left) for left, right in pairs])
    for name, flow in flows.items():
        _slug(name, "flow name")
        if not isinstance(flow, dict) or set(flow) != {
            "client",
            "client_ip",
            "server",
            "server_ip",
            "forward",
            "return",
        }:
            raise ValueError("invalid named flow fields")
        client = _slug(flow["client"], "flow client")
        server = _slug(flow["server"], "flow server")
        client_ip, server_ip = _address(flow["client_ip"]), _address(flow["server_ip"])
        if (
            client == server
            or managed.get(client) != "host"
            or managed.get(server) != "host"
            or client_ip == server_ip
            or not {client_ip, server_ip} <= addresses
        ):
            raise ValueError("named flows need distinct managed hosts and public addresses")
        forward = routes.get(_slug(flow["forward"], "forward route"))
        reverse = routes.get(_slug(flow["return"], "return route"))
        if (
            forward is None
            or reverse is None
            or forward["destination"] != server_ip
            or reverse["destination"] != client_ip
        ):
            raise ValueError("named flows need matching forward and return route checks")
        send = (forward["node"], forward["interface"])
        receive = (reverse["node"], reverse["interface"])
        if send[0] == receive[0] or peers.get(":".join(send)) != ":".join(receive):
            raise ValueError("forward and return routes must use the same inter-router link")
        result[name] = ServiceFlow(client, client_ip, server, server_ip, send, receive)
    first, second = result.values()
    if (
        len({first.client, first.server, second.client, second.server}) != 4
        or len({first.client_ip, first.server_ip, second.client_ip, second.server_ip}) != 4
        or first.sender[0] != second.sender[0]
        or first.receiver[0] != second.receiver[0]
        or first.sender == second.sender
    ):
        raise ValueError("named flows need separate endpoints and links between the same routers")
    return result


def load_network(case: CaseDefinition) -> NetworkSpec:
    if not case.network:
        raise ValueError("live cases need a network environment profile")
    root = REPOSITORY_ROOT
    path = (root / "environments" / case.network / "network.yaml").resolve()
    if not path.is_relative_to(root / "environments") or not path.is_file():
        raise ValueError("network profile does not exist inside environments/")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "managed_nodes",
        "public_addresses",
        "bgp_peers",
        "routes",
        "queue",
        "background",
        "data_plane_mtu",
    }
    if case.partial_path is not None:
        required = required - {"queue", "background"} | {"flows", "service_min_fraction"}
    if (
        not isinstance(profile, dict)
        or not required <= set(profile)
        or set(profile) - required - {"endpoint_cpu_affinity", "noise"}
        or type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("invalid network profile fields or schema_version")
    affinity = profile.get("endpoint_cpu_affinity", "unrestricted")
    if affinity not in ("single_cpu", "unrestricted"):
        raise ValueError("endpoint_cpu_affinity must be single_cpu or unrestricted")
    mtu = profile["data_plane_mtu"]
    if type(mtu) is not int or not 576 <= mtu <= 9216:
        raise ValueError("data_plane_mtu must be between 576 and 9216")
    topology = yaml.safe_load(case.topology.read_text(encoding="utf-8"))
    if not isinstance(topology, dict) or not isinstance(topology.get("topology"), dict):
        raise ValueError("topology must be a Containerlab mapping")
    lab = _slug(topology.get("name"), "lab name")
    nodes = topology["topology"].get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("topology needs nodes")
    for node, config in nodes.items():
        _slug(node, "node")
        if (
            not isinstance(config, dict)
            or config.get("kind") != "linux"
            or not isinstance(config.get("image"), str)
        ):
            raise ValueError("this backend supports Linux containers only")
    managed = profile["managed_nodes"]
    if (
        not isinstance(managed, dict)
        or not managed
        or not set(managed) <= set(nodes)
        or any(role not in ("router", "host") for role in managed.values())
    ):
        raise ValueError("managed nodes must be topology nodes with router or host roles")
    all_interfaces: dict[str, list[str]] = {node: [] for node in nodes}
    links = topology["topology"].get("links")
    if not isinstance(links, list):
        raise ValueError("topology needs a link list")
    for link in links:
        endpoints = link.get("endpoints") if isinstance(link, dict) else None
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError("only two-ended topology links are supported")
        for endpoint in endpoints:
            if not isinstance(endpoint, str) or endpoint.count(":") != 1:
                raise ValueError("invalid link endpoint")
            node, interface = endpoint.split(":")
            if (
                node not in nodes
                or not re.fullmatch(r"eth[1-9][0-9]*", interface)
                or interface in all_interfaces[node]
            ):
                raise ValueError("invalid or duplicate data-plane link endpoint")
            all_interfaces[node].append(interface)
    routers = frozenset(node for node, role in managed.items() if role == "router")
    addresses = profile["public_addresses"]
    if not isinstance(addresses, list) or not addresses:
        raise ValueError("public_addresses must be a nonempty list")
    addresses = frozenset(_address(address) for address in addresses)
    client, server = case.workload.get("client"), case.workload.get("server")
    server_ip = _address(case.workload.get("server_ip"))
    if (
        client not in managed
        or server not in managed
        or managed[client] != "host"
        or managed[server] != "host"
        or server_ip not in addresses
    ):
        raise ValueError("workload endpoints must be managed hosts with a public target address")
    peers = profile["bgp_peers"]
    if not isinstance(peers, dict) or set(peers) != set(routers):
        raise ValueError("bgp_peers must describe every managed router")
    for values in peers.values():
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or any(_address(value) not in addresses for value in values)
        ):
            raise ValueError("BGP peers must be distinct public addresses")
    routes = profile["routes"]
    if not isinstance(routes, dict) or not routes:
        raise ValueError("routes must contain expected forwarding checks")
    for route in routes.values():
        if not isinstance(route, dict) or set(route) != {"node", "destination", "via", "interface"}:
            raise ValueError("invalid route check fields")
        if route["node"] not in routers or route["interface"] not in all_interfaces[route["node"]]:
            raise ValueError("route check refers to an unknown router interface")
        _address(route["destination"])
        _address(route["via"])
    queue = profile.get("queue", {})
    if case.partial_path is None and (
        not isinstance(queue, dict)
        or set(queue) != {"node", "interface", "kind"}
        or queue["node"] not in routers
        or queue["interface"] not in all_interfaces[queue["node"]]
        or queue["kind"] != "tbf"
    ):
        raise ValueError("queue must identify an existing TBF-shaped router interface")
    background = profile.get("background", {})
    if case.partial_path is None and (
        not isinstance(background, dict)
        or set(background)
        != {
            "client",
            "server",
            "server_ip",
            "interface",
        }
    ):
        raise ValueError("invalid background workload fields")
    for role in ("client", "server"):
        if not background:
            break
        if (
            background[role] not in nodes
            or background[role] in managed
            or background["interface"] not in all_interfaces[background[role]]
        ):
            raise ValueError("background controllers must be unmanaged topology endpoints")
    if background:
        _address(background["server_ip"])
    noise = profile.get("noise")
    if noise is not None and (
        not isinstance(noise, dict)
        or set(noise) != {"client", "server", "server_ip", "rates_mbps", "step_seconds"}
        or noise["client"] not in nodes
        or noise["server"] not in nodes
        or {noise["client"], noise["server"]} & set(managed)
        or not noise["rates_mbps"]
        or any(type(rate) not in (int, float) or rate <= 0 for rate in noise["rates_mbps"])
        or type(noise["step_seconds"]) is not int
        or noise["step_seconds"] < 1
    ):
        raise ValueError("background noise needs unmanaged endpoints and a rate schedule")
    if noise is not None:
        _address(noise["server_ip"])
    flows = None
    service_min_fraction = None
    if case.partial_path is not None:
        service_min_fraction = profile["service_min_fraction"]
        if (
            type(service_min_fraction) not in (int, float)
            or not 0 < service_min_fraction < 1
            or abs(service_min_fraction + case.validation.threshold - 1) > 1e-9
        ):
            raise ValueError("private validation must agree with the public service requirement")
        flows = _load_flows(profile, managed, addresses, routes, links)
        if (client, server, server_ip) not in {
            (flow.client, flow.server, flow.server_ip) for flow in flows.values()
        }:
            raise ValueError("the default workload must be a named flow")
        if case.partial_path.flow is not None:
            impaired = flows.get(case.partial_path.flow)
            if impaired is None or impaired.sender != (
                case.fault.get("node"),
                case.fault.get("interface"),
            ):
                raise ValueError("loss must target the named flow's inter-router egress")
    bound_files = []
    for node in routers:
        mounts = set()
        for bind in nodes[node].get("binds", []):
            parts = bind.split(":") if isinstance(bind, str) else []
            source = (case.topology.parent / parts[0]).resolve() if len(parts) == 3 else None
            if (
                len(parts) != 3
                or parts[2] != "ro"
                or not source.is_file()
                or not source.is_relative_to(root)
            ):
                raise ValueError("router configs must be repository-owned read-only bind mounts")
            mounts.add(parts[1])
            bound_files.append(source)
        if mounts != {"/etc/frr/daemons", "/etc/frr/frr.conf", "/etc/frr/vtysh.conf"}:
            raise ValueError("router configuration mount targets do not match FRR")
    policy = DiagnosticPolicy(
        {node: tuple(sorted(all_interfaces[node])) for node in managed},
        routers,
        addresses,
        client,
        server_ip,
        tuple(flows or ()),
    )
    return NetworkSpec(
        path,
        topology,
        lab,
        tuple(nodes),
        policy,
        {node: tuple(values) for node, values in peers.items()},
        routes,
        queue,
        background,
        tuple(bound_files),
        {node: tuple(values) for node, values in all_interfaces.items()},
        mtu,
        affinity,
        flows,
        service_min_fraction,
        noise,
    )
