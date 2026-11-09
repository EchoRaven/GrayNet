"""Network lifecycle interface shared by execution and backends."""

from collections.abc import Sequence
from typing import Literal, Protocol

from graynet.scenarios.definitions import CaseDefinition

Phase = Literal["baseline", "incident", "recovery"]


class NetworkBackend(Protocol):
    def deploy(self, case: CaseDefinition) -> None: ...

    def measure(self, case: CaseDefinition, phase: Phase) -> Sequence[float]: ...

    def inject(self, case: CaseDefinition) -> None: ...

    def recover(self, case: CaseDefinition) -> None: ...

    def teardown(self, case: CaseDefinition) -> None: ...
