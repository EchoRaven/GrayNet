"""Resolve driver adapters without coupling execution code to a CLI vendor."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

from graynet.agents.base import Agent
from graynet.tools.network import NetworkTools


@dataclass(frozen=True)
class DriverConfig:
    model: str | None
    reasoning_effort: str | None
    timeout_seconds: int = 300
    max_tool_calls: int = 64

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "max_tool_calls"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DriverContext:
    tools: NetworkTools
    auth_file: Path
    artifacts: Path
    runtime_context: Callable[[], dict[str, Any]]


class ManagedDriver(Agent, Protocol):
    metadata: dict[str, Any]
    trace: list[dict[str, Any]]

    def preflight(self) -> None: ...


def _codex(context: DriverContext, config: DriverConfig) -> ManagedDriver:
    from graynet.agents.codex import CodexDriver

    return CodexDriver(
        context.tools,
        auth_file=context.auth_file,
        artifacts=context.artifacts,
        model=config.model,
        effort=config.reasoning_effort,
        timeout_seconds=config.timeout_seconds,
        max_tool_calls=config.max_tool_calls,
        runtime_context=context.runtime_context,
    )


def _factories() -> dict[str, Callable[..., ManagedDriver]]:
    factories = {"codex": _codex}
    for entry in entry_points(group="graynet.drivers"):
        if entry.name in factories:
            raise ValueError(f"duplicate driver registration: {entry.name}")
        # Defer plugin imports until the selected driver is instantiated.
        factories[entry.name] = entry.load
    return factories


def available_drivers() -> tuple[str, ...]:
    return tuple(sorted(_factories()))


def create_driver(name: str, context: DriverContext, config: DriverConfig) -> ManagedDriver:
    factories = _factories()
    if name not in factories:
        raise ValueError(f"unknown driver {name}; available: {', '.join(sorted(factories))}")
    factory = factories[name] if name == "codex" else factories[name]()
    driver = factory(context, config)
    if not all(callable(getattr(driver, method, None)) for method in ("preflight", "diagnose")):
        raise ValueError("driver must implement preflight and diagnose")
    if not isinstance(getattr(driver, "metadata", None), dict) or not isinstance(
        getattr(driver, "trace", None), list
    ):
        raise ValueError("driver must provide metadata and an observation trace")
    return driver
