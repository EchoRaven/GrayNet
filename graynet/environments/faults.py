"""Private, bounded fault primitives. Never expose these controls to agents."""

import math
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from graynet.runtime.commands import run_process

if TYPE_CHECKING:
    from graynet.environments.containerlab import ContainerlabBackend


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
    def __init__(self, backend: "ContainerlabBackend") -> None:
        self.backend = backend
        self.spec = backend.case.fault
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
        elif self.kind in ("link_loss", "link_delay"):
            self.node, self.interface = self.spec.get("node"), self.spec.get("interface")
            if (
                self.node not in backend.network.policy.interfaces
                or self.interface not in backend.network.policy.interfaces[self.node]
            ):
                raise ValueError("link impairment needs an existing managed data-plane interface")
            if (self.node, self.interface) == (backend.queue_node, backend.queue_interface):
                raise ValueError("do not replace the operational shaper with a fault injector")
            self.value = positive(
                self.spec.get("loss_percent" if self.kind == "link_loss" else "delay_ms"),
                self.kind,
                50 if self.kind == "link_loss" else 500,
            )
            self.seed = self.spec.get("seed", 11)
            if type(self.seed) is not int or not 0 <= self.seed < 2**32:
                raise ValueError("fault seed must be an unsigned 32-bit integer")
        else:
            raise ValueError(f"unsupported fault primitive: {self.kind}")

    def verify(self, phase: str) -> None:
        expected = phase == "incident" and self.kind != "none"
        if self.kind == "cross_traffic" or self.kind == "none":
            if self.backend._load_running() != expected:
                raise RuntimeError(f"unexpected cross traffic state during {phase}")
        elif self.kind in ("link_loss", "link_delay"):
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
                matched = ("qdisc netem" in raw) == expected
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
        else:
            existing = self.backend._docker(self.node, "tc", "qdisc", "show", "dev", self.interface)
            uses_neutral_queue = self.backend.case.partial_path is not None
            if not uses_neutral_queue and existing.strip() and "qdisc noqueue" not in existing:
                self.started = False
                raise RuntimeError("refusing to replace an existing operational queue")
            option = (
                ["loss", "random", f"{self.value:g}%", "seed", str(self.seed)]
                if self.kind == "link_loss"
                else ["delay", f"{self.value:g}ms"]
            )
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

    def snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "active": self.active}
        if self.kind in ("link_loss", "link_delay"):
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


def build_fault(backend: "ContainerlabBackend") -> FaultController:
    if backend.case.fault.get("pattern"):
        return ScheduledTraffic(backend)
    return FaultController(backend)
