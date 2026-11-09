"""Private, bounded fault primitives. Never expose these controls to agents."""

import math
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from graynet.runtime.commands import run_process

if TYPE_CHECKING:
    from graynet.environments.containerlab import ContainerlabBackend


LINK_IMPAIRMENTS = {
    "link_loss": ("loss_percent", 50),
    "link_delay": ("delay_ms", 500),
    "link_jitter": ("delay_ms", 500),
    "link_rate_limit": ("rate_mbps", 10000),
    "link_burst_loss": ("enter_percent", 50),
    "link_police": ("rate_mbps", 10000),
    "link_mtu": ("mtu", 1400),
}
BGP_FAULTS = ("bgp_session_down", "bgp_weight", "bgp_flap")
ROUTE_FIELDS = {"kind", "node", "asn", "peer", "destination", "via", "duration_seconds"}


def loss_parameters(raw: str) -> tuple[float, int | None]:
    """Read effective loss and seed from a private netem queue snapshot."""
    loss = re.search(r"\bloss\s+(?:random\s+)?([\d.]+)%", raw)
    seed = re.search(r"\bseed (\d+)\b", raw)
    return float(loss[1]) if loss else 0.0, int(seed[1]) if seed else None


def positive(value: Any, name: str, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be positive and at most {maximum}")
    return float(value)


class FaultController:
    def __init__(self, backend: "ContainerlabBackend", spec: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.spec = backend.case.fault if spec is None else spec
        self.kind = self.spec["kind"]
        self.active = False
        self.started = False
        self.rate = 0.0
        allowed = {
            "none": {"kind"},
            "cross_traffic": {
                "kind",
                "client",
                "server",
                "server_ip",
                "rate_mbps",
                "duration_seconds",
                "pattern",
            },
            "link_loss": {"kind", "node", "interface", "loss_percent", "seed", "duration_seconds"},
            "link_delay": {"kind", "node", "interface", "delay_ms", "duration_seconds"},
            "link_jitter": {
                "kind",
                "node",
                "interface",
                "delay_ms",
                "jitter_ms",
                "duration_seconds",
            },
            "link_rate_limit": {"kind", "node", "interface", "rate_mbps", "duration_seconds"},
            "link_burst_loss": {
                "kind",
                "node",
                "interface",
                "enter_percent",
                "exit_percent",
                "duration_seconds",
            },
            "link_police": {"kind", "node", "interface", "rate_mbps", "duration_seconds"},
            "link_mtu": {
                "kind",
                "node",
                "interface",
                "mtu",
                "icmp_interface",
                "duration_seconds",
            },
            "bgp_session_down": ROUTE_FIELDS,
            "bgp_weight": ROUTE_FIELDS | {"weight", "restore_weight"},
            "bgp_flap": ROUTE_FIELDS | {"pattern"},
            "endpoint_overload": {"kind", "node", "workers", "duration_seconds"},
        }
        if self.kind not in allowed or set(self.spec) - allowed[self.kind]:
            raise ValueError("unsupported fault primitive or unexpected injector fields")
        if self.kind == "none":
            if set(self.spec) != {"kind"}:
                raise ValueError("healthy controls cannot contain injector parameters")
        elif self.kind == "cross_traffic":
            if (self.spec.get("client"), self.spec.get("server"), self.spec.get("server_ip")) != (
                backend.load_client,
                backend.load_server,
                backend.LOAD_SERVER_IP,
            ):
                raise ValueError(
                    "this backend only supports profile-defined cross traffic endpoints"
                )
            self.rate = positive(self.spec.get("rate_mbps"), "rate_mbps", 10000)
        elif self.kind == "endpoint_overload":
            self.node = self.spec.get("node")
            self.workers = self.spec.get("workers")
            if (
                self.node != backend.server
                or type(self.workers) is not int
                or not 1 <= self.workers <= 64
            ):
                raise ValueError("endpoint overload needs the service endpoint and 1..64 workers")
            if backend.case.workload.get("cpu_limit_cores") is None:
                raise ValueError("endpoint contention needs a fixed CPU limit in all phases")
        elif self.kind in LINK_IMPAIRMENTS:
            self.node, self.interface = self.spec.get("node"), self.spec.get("interface")
            if (
                self.node not in backend.network.policy.interfaces
                or self.interface not in backend.network.policy.interfaces[self.node]
            ):
                raise ValueError("link impairment needs an existing managed data-plane interface")
            if (self.node, self.interface) == (backend.queue_node, backend.queue_interface):
                raise ValueError("do not replace the operational shaper with a fault injector")
            field, maximum = LINK_IMPAIRMENTS[self.kind]
            self.value = positive(self.spec.get(field), self.kind, maximum)
            if self.kind == "link_jitter":
                self.jitter = positive(self.spec.get("jitter_ms"), "jitter_ms", 500)
            elif self.kind == "link_burst_loss":
                self.exit = positive(self.spec.get("exit_percent"), "exit_percent", 100)
            elif self.kind == "link_mtu":
                self.icmp_interface = self.spec.get("icmp_interface")
                if (
                    type(self.spec["mtu"]) is not int
                    or self.value < 576
                    or self.icmp_interface not in backend.network.policy.interfaces[self.node]
                ):
                    raise ValueError("link_mtu needs an integer MTU and a return interface")
            self.seed = self.spec.get("seed", 11)
            if type(self.seed) is not int or not 0 <= self.seed < 2**32:
                raise ValueError("fault seed must be an unsigned 32-bit integer")
        elif self.kind in BGP_FAULTS:
            self.node = self.spec.get("node")
            if self.node not in backend.network.bgp_peers or self.spec.get(
                "peer"
            ) not in backend.network.bgp_peers.get(self.node, ()):
                raise ValueError("BGP faults need a configured router and peer")
            if type(self.spec.get("asn")) is not int:
                raise ValueError("BGP faults need the router's AS number")
            self.rate = 1.0  # A flap schedule uses one level for "session down".
        else:
            raise ValueError(f"unsupported fault primitive: {self.kind}")

    def verify(self, phase: str) -> None:
        expected = phase == "incident" and self.kind != "none"
        if self.kind == "cross_traffic" or self.kind == "none":
            if self.backend._load_running() != expected:
                raise RuntimeError(f"unexpected cross traffic state during {phase}")
        elif self.kind in BGP_FAULTS:
            if expected and not self._on_alternate_route():
                raise RuntimeError("routing did not move to the expected alternate path")
        elif self.kind == "link_mtu":
            raw = self.backend._docker(self.node, "ip", "-o", "link", "show", "dev", self.interface)
            if (f"mtu {self.value:g} " in raw) != expected:
                raise RuntimeError(f"link MTU state changed during {phase}")
        elif self.kind in LINK_IMPAIRMENTS:
            partial = self.backend.case.partial_path is not None
            raw = self.backend._docker(
                self.node,
                "tc",
                *(("-d",) if partial else ()),
                "-s",
                "qdisc",
                "show",
                "dev",
                self.interface,
            )
            if partial:
                loss, seed = loss_parameters(raw)
                matched = "qdisc netem" in raw and (
                    math.isclose(loss, self.value, abs_tol=0.0001) and seed == self.seed
                    if expected
                    else loss == 0
                )
            else:
                marker = "qdisc tbf" if self.kind == "link_police" else "qdisc netem"
                matched = (marker in raw) == expected
            if not matched:
                raise RuntimeError(f"link impairment state changed during {phase}")
        elif self.kind == "endpoint_overload":
            result = run_process(
                ["docker", "exec", self.backend._container(self.node), "pgrep", "-x", "sha256sum"],
                check=False,
            )
            if result.returncode not in (0, 1) or (result.returncode == 0) != expected:
                raise RuntimeError(f"endpoint contention state changed during {phase}")

    def start(self) -> None:
        self.verify("baseline")
        if self.kind == "none":
            return
        self.started = True  # Cleanup is necessary even if the command partially fails.
        if self.kind == "cross_traffic":
            self.backend._docker(
                self.backend.load_client,
                "iperf3",
                "-c",
                self.backend.LOAD_SERVER_IP,
                "-u",
                "-b",
                f"{self.rate:g}M",
                "-l",
                "1200",
                "-t",
                str(self.backend.fault_duration),
                detached=True,
            )
        elif self.kind == "endpoint_overload":
            # All work stays inside the owned, CPU-limited endpoint container.
            # The shell is an evaluator control, never an agent tool.
            script = f"i=0; while [ $i -lt {self.workers} ]; do sha256sum /dev/zero >/dev/null & i=$((i+1)); done; wait"
            self.backend._docker(self.node, "sh", "-c", script, detached=True)
        elif self.kind in BGP_FAULTS:
            self._apply_bgp(active=True)
            self._wait_for_route(alternate=True)
        elif self.kind == "link_mtu":
            self.backend._docker(
                self.node, "ip", "link", "set", "dev", self.interface, "mtu", f"{self.value:g}"
            )
            # Suppress "fragmentation needed" so path MTU discovery cannot adapt.
            self.backend._docker(
                self.node, "tc", "qdisc", "add", "dev", self.icmp_interface, "clsact"
            )
            self.backend._docker(
                self.node,
                *("tc", "filter", "add", "dev", self.icmp_interface, "egress", "protocol", "ip"),
                *("u32", "match", "ip", "protocol", "1", "0xff"),
                *("match", "u8", "3", "0xff", "at", "20", "action", "drop"),
            )
        elif self.kind == "link_police":
            self.backend._docker(
                self.node,
                *("tc", "qdisc", "add", "dev", self.interface, "root", "tbf"),
                *("rate", f"{self.value:g}mbit", "burst", "16kb", "latency", "1ms"),
            )
        else:
            existing = self.backend._docker(self.node, "tc", "qdisc", "show", "dev", self.interface)
            uses_neutral_queue = self.backend.case.partial_path is not None
            if not uses_neutral_queue and existing.strip() and "qdisc noqueue" not in existing:
                self.started = False
                raise RuntimeError("refusing to replace an existing operational queue")
            if self.kind == "link_loss":
                option = ["loss", "random", f"{self.value:g}%", "seed", str(self.seed)]
            elif self.kind == "link_jitter":
                option = ["delay", f"{self.value:g}ms", f"{self.jitter:g}ms"]
            elif self.kind == "link_rate_limit":
                option = ["rate", f"{self.value:g}mbit"]
            elif self.kind == "link_burst_loss":
                option = ["loss", "gemodel", f"{self.value:g}%", f"{self.exit:g}%"]
            else:
                option = ["delay", f"{self.value:g}ms"]
            self.backend._docker(
                self.node,
                "tc",
                "qdisc",
                "change" if uses_neutral_queue else "add",
                "dev",
                self.interface,
                "root",
                "netem",
                "limit",
                "10000",
                *option,
            )
        self.active = True
        time.sleep(1)
        self.verify("incident")

    def _on_alternate_route(self) -> bool:
        route = self.backend._docker(
            self.node, "ip", "route", "get", self.spec["destination"]
        ).strip()
        return f"via {self.spec['via']} " in route

    def _wait_for_route(self, *, alternate: bool) -> None:
        for _ in range(45):
            if self._on_alternate_route() == alternate:
                return
            time.sleep(1)
        raise RuntimeError("routing did not converge after the BGP change")

    def _apply_bgp(self, *, active: bool) -> None:
        peer = self.spec["peer"]
        if self.kind == "bgp_weight":
            weight = self.spec["weight"] if active else self.spec["restore_weight"]
            commands = ["address-family ipv4 unicast", f"neighbor {peer} weight {weight}"]
        else:
            commands = [f"{'' if active else 'no '}neighbor {peer} shutdown"]
        arguments = ["conf t", f"router bgp {self.spec['asn']}", *commands, "end"]
        if self.kind == "bgp_weight":
            arguments.append("clear bgp ipv4 unicast * soft in")
        self.backend._docker(
            self.node, "vtysh", *[part for line in arguments for part in ("-c", line)]
        )

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "active": self.active}
        if self.kind in LINK_IMPAIRMENTS:
            raw = self.backend._docker(
                self.node, "tc", "-s", "qdisc", "show", "dev", self.interface
            )
            dropped = re.search(r"dropped (\d+)", raw)
            result.update(raw=raw.strip(), dropped=int(dropped.group(1)) if dropped else 0)
        return result

    def stop(self) -> None:
        if not self.started:
            return
        if self.kind == "cross_traffic":
            result = run_process(
                [
                    "docker",
                    "exec",
                    self.backend._container(self.backend.load_client),
                    "pkill",
                    "-x",
                    "iperf3",
                ],
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError("could not stop competing traffic")
        elif self.kind == "endpoint_overload":
            result = run_process(
                ["docker", "exec", self.backend._container(self.node), "pkill", "-x", "sha256sum"],
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError("could not stop endpoint competing work")
        elif self.kind in BGP_FAULTS:
            self._apply_bgp(active=False)
            self.backend._wait_for_routing()
            self._wait_for_route(alternate=False)
        elif self.kind == "link_mtu":
            self.backend._docker(
                self.node,
                *("ip", "link", "set", "dev", self.interface, "mtu"),
                str(self.backend.network.mtu),
            )
            self.backend._docker(
                self.node, "tc", "qdisc", "del", "dev", self.icmp_interface, "clsact"
            )
        elif self.backend.case.partial_path is not None:
            self.backend._docker(
                self.node,
                "tc",
                "qdisc",
                "change",
                "dev",
                self.interface,
                "root",
                "netem",
                "limit",
                "10000",
                "loss",
                "random",
                "0%",
                "seed",
                "0",
            )
        else:
            self.backend._docker(self.node, "tc", "qdisc", "del", "dev", self.interface, "root")
        self.active = self.started = False
        time.sleep(1)
        self.verify("recovery")


class ScheduledTraffic(FaultController):
    """Clock-driven cross traffic with recorded actual transition times."""

    def __init__(self, backend: "ContainerlabBackend") -> None:
        super().__init__(backend)
        pattern = self.spec["pattern"]
        self.pattern = pattern.get("kind")
        if self.pattern == "bursty" and set(pattern) == {"kind", "on_seconds", "off_seconds"}:
            self.stages = [
                (self.rate, positive(pattern["on_seconds"], "on_seconds", 300)),
                (0.0, positive(pattern["off_seconds"], "off_seconds", 300)),
            ]
        elif self.pattern == "gradual" and set(pattern) == {"kind", "step_seconds", "rates_mbps"}:
            step = positive(pattern["step_seconds"], "step_seconds", 300)
            rates = pattern["rates_mbps"]
            if not isinstance(rates, list) or not 3 <= len(rates) <= 10:
                raise ValueError("a gradual schedule needs 3..10 rate levels")
            rates = [positive(rate, "rates_mbps", self.rate) for rate in rates]
            if rates[-1] != self.rate or any(
                second <= first for first, second in zip(rates, rates[1:], strict=False)
            ):
                raise ValueError("gradual rates must strictly increase to rate_mbps")
            self.stages = [(rate, step) for rate in rates]
        else:
            raise ValueError("invalid cross traffic schedule")
        self.events: list[dict[str, Any]] = []
        self.epochs: list[list[dict[str, Any]]] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self.level = 0.0

    def _change(self, rate: float, cycle: int) -> None:
        with self._lock:
            if self.events and "finished_monotonic" not in self.events[-1]:
                self.events[-1]["finished_monotonic"] = time.monotonic()
            result = run_process(
                [
                    "docker",
                    "exec",
                    self.backend._container(self.backend.load_client),
                    "pkill",
                    "-x",
                    "iperf3",
                ],
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError("could not switch scheduled cross traffic")
            if rate:
                self.backend._docker(
                    self.backend.load_client,
                    "iperf3",
                    "-c",
                    self.backend.LOAD_SERVER_IP,
                    "-u",
                    "-b",
                    f"{rate:g}M",
                    "-l",
                    "1200",
                    "-t",
                    str(self.backend.fault_duration),
                    detached=True,
                )
            self.level = rate
            self.events.append(
                {"started_monotonic": time.monotonic(), "rate_mbps": rate, "cycle": cycle}
            )

    def start(self) -> None:
        if self.backend._load_running():
            raise RuntimeError("cross traffic was already active before the scheduled episode")
        self._stop.clear()
        self._error = None
        self.events = []
        self.epochs.append(self.events)
        self.started = self.active = True
        self._change(self.stages[0][0], 0)
        started = time.monotonic()

        def run():
            index = cycle = 0
            try:
                while not self._stop.wait(self.stages[index][1]):
                    if time.monotonic() - started >= self.backend.fault_duration:
                        raise RuntimeError("scheduled fault exceeded its reserved duration")
                    if index + 1 < len(self.stages):
                        index += 1
                    elif self.pattern == "bursty":
                        index = 0
                        cycle += 1
                    else:
                        continue  # Hold the final ramp level, without restarting traffic.
                    self._change(self.stages[index][0], cycle)
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(1)
        self.verify("incident")

    def verify(self, phase: str) -> None:
        with self._lock:
            if self._error:
                raise RuntimeError("fault scheduler failed: " + self._error)
            if self.backend._load_running() != (phase == "incident" and self.level > 0):
                raise RuntimeError("scheduled traffic stopped or changed unexpectedly")

    def timeline(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            return [
                {**event, "finished_monotonic": event.get("finished_monotonic", now)}
                for event in self.events
            ]

    def mean_rate(self, start: float, end: float) -> float:
        return sum(
            max(0, min(end, event["finished_monotonic"]) - max(start, event["started_monotonic"]))
            * event["rate_mbps"]
            for event in self.timeline()
        ) / (end - start)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=35)
            if self._thread.is_alive():
                raise RuntimeError("fault scheduler did not stop")
        with self._lock:
            if self.events and "finished_monotonic" not in self.events[-1]:
                self.events[-1]["finished_monotonic"] = time.monotonic()
            # Stop preserves errors in the record but still cleans up.
            error, self._error = self._error, None
            self.level = 0
            try:
                super().stop()
            finally:
                self.backend.evidence["fault_epochs"] = self.epochs
                self.backend.evidence["scheduler_error"] = error


class ScheduledBgpFlap(ScheduledTraffic):
    """Repeat a routing session failure; level 1 means the session is down."""

    def _change(self, rate: float, cycle: int) -> None:
        with self._lock:
            if self.events and "finished_monotonic" not in self.events[-1]:
                self.events[-1]["finished_monotonic"] = time.monotonic()
            self._apply_bgp(active=bool(rate))
            self.level = rate
            self.events.append(
                {"started_monotonic": time.monotonic(), "rate_mbps": rate, "cycle": cycle}
            )

    def verify(self, phase: str) -> None:
        with self._lock:
            if self._error:
                raise RuntimeError("fault scheduler failed: " + self._error)


class CompositeFault(FaultController):
    """Two or three independent primitives active over the same interval."""

    def __init__(self, backend: "ContainerlabBackend") -> None:
        self.backend, self.spec, self.kind = backend, backend.case.fault, "multiple"
        self.active = self.started = False
        self.rate = 0.0
        faults = self.spec.get("faults")
        if (
            set(self.spec) != {"kind", "faults", "duration_seconds"}
            or not isinstance(faults, list)
            or not 2 <= len(faults) <= 3
            or any(item.get("kind") in ("none", "multiple", *BGP_FAULTS) for item in faults)
        ):
            raise ValueError("a composite fault needs 2..3 link, traffic, or endpoint faults")
        self.children = [FaultController(backend, spec) for spec in faults]

    def verify(self, phase: str) -> None:
        for child in self.children:
            child.verify(phase)

    def start(self) -> None:
        self.started = True
        for child in self.children:
            child.start()
        self.active = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "active": self.active,
            "children": [child.snapshot() for child in self.children],
        }

    def stop(self) -> None:
        errors = []
        for child in reversed(self.children):
            try:
                child.stop()
            except Exception as exc:
                errors.append(f"{child.kind}: {exc}")
        self.active = self.started = False
        if errors:
            raise RuntimeError("; ".join(errors))


def build_fault(backend: "ContainerlabBackend") -> FaultController:
    if backend.case.fault.get("kind") == "multiple":
        return CompositeFault(backend)
    if backend.case.fault.get("kind") == "bgp_flap":
        return ScheduledBgpFlap(backend)
    if backend.case.fault.get("pattern"):
        return ScheduledTraffic(backend)
    return FaultController(backend)
