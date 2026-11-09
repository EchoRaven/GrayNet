"""Tool-free evaluator model in a separate, read-only CLI container."""

from pathlib import Path
from typing import Any

from graynet.evaluation.models import Completion, CompletionError, CompletionInterrupted
from graynet.runtime.catalog import preflight_model
from graynet.runtime.codex import (
    IsolatedCodexSession,
    SessionError,
    SessionInterrupted,
    parse_events,
)


class CodexJudgmentModel:
    def __init__(
        self,
        *,
        model: str,
        effort: str,
        auth_file: Path,
        timeout_seconds: int = 120,
        image: str = "graynet-codex:0.1",
    ):
        if not model or not effort or type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError(
                "judge model, reasoning effort, and positive deadline must be explicit"
            )
        self.model, self.effort, self.auth_file = model, effort, auth_file
        self.timeout_seconds, self.image = timeout_seconds, image
        self._identity: dict[str, Any] | None = None

    def preflight(self) -> None:
        runtime = preflight_model(self.image, self.auth_file, self.model, self.effort)
        self.image, self.model, self.effort = runtime.image_id, runtime.model, runtime.effort
        self.provider_hosts = runtime.provider_hosts
        self._identity = {
            "driver": "codex",
            "model": self.model,
            "reasoning_effort": self.effort,
            "timeout_seconds": self.timeout_seconds,
            "image_id": runtime.image_id,
            "cli_version": runtime.cli_version,
        }

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            raise ValueError("judge model must pass preflight before evaluation")
        return dict(self._identity)

    def complete(self, prompt: str, schema: dict[str, Any]) -> Completion:
        identity = self.identity
        session = IsolatedCodexSession(
            image=identity["image_id"],
            model=self.model,
            effort=self.effort,
            auth_file=self.auth_file,
            provider_hosts=self.provider_hosts,
            timeout_seconds=self.timeout_seconds,
            role="judge",
        )
        error = None
        interrupted = False
        try:
            output = session.run(prompt, schema)
        except (SessionError, SessionInterrupted) as exc:
            output = exc.output
            error = str(exc)
            interrupted = isinstance(exc, SessionInterrupted)
        events, usage = parse_events(output.stdout)
        metadata = {**identity, **output.metadata, "usage": usage, "estimated_cost_usd": None}
        text = next(
            (
                event["item"].get("text")
                for event in reversed(events)
                if event.get("type") == "item.completed"
                and isinstance(event.get("item"), dict)
                and event["item"].get("type") == "agent_message"
            ),
            None,
        )
        tool_types = {"command_execution", "mcp_tool_call", "web_search", "file_change"}
        if any(
            event["item"].get("type") in tool_types
            for event in events
            if isinstance(event.get("item"), dict)
        ):
            error = "judge attempted to use a forbidden tool"
            metadata["status"] = "error"
        completion = Completion(text, output.stdout, output.stderr, metadata, output.proxy_log)
        if interrupted:
            raise CompletionInterrupted(completion)
        if error is not None:
            raise CompletionError(error, completion)
        return completion
