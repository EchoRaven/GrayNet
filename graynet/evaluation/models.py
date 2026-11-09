"""Model-independent completion boundary for diagnosis evaluation."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Completion:
    text: str | None
    events: str = ""
    stderr: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    egress_log: str = ""


class CompletionError(RuntimeError):
    """Retain partial output when the evaluator model fails."""

    def __init__(self, message: str, completion: Completion):
        super().__init__(message)
        self.completion = completion


class CompletionInterrupted(KeyboardInterrupt):
    def __init__(self, completion: Completion):
        super().__init__("judge completion interrupted")
        self.completion = completion


class JudgmentModel(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def complete(self, prompt: str, schema: dict[str, Any]) -> Completion: ...
