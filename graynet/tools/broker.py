"""Bounded MCP requests over a run-specific Unix socket."""

import json
import socketserver
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graynet.tools.network import NetworkTools


class ToolBroker:
    """Small MCP endpoint over a per-run Unix socket, never exposed on TCP."""

    def __init__(self, tools: NetworkTools, *, max_calls: int, timeout_seconds: int) -> None:
        self.tools = tools
        self.max_calls = max_calls
        self.deadline = time.monotonic() + timeout_seconds
        self.trace: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._calls = 0
        self.server: socketserver.UnixStreamServer | None = None
        self.thread: threading.Thread | None = None

    def dispatch(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid request"},
            }
        if "id" not in message:
            return None
        response = {"jsonrpc": "2.0", "id": message["id"]}
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(params, dict):
            return {**response, "error": {"code": -32602, "message": "Invalid parameters"}}
        if method == "initialize":
            version = params.get("protocolVersion")
            if version not in ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"):
                version = "2025-03-26"
            result = {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "network-diagnostics", "version": "0.1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tools.definitions()}
        elif method == "tools/call":
            started = time.monotonic()
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "tool": params.get("name"),
                "arguments": params.get("arguments", {}),
            }
            try:
                # Bound concurrent probes; shutdown rejects queued operations
                # before they can reach the managed network.
                with self._operation_lock:
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError("diagnostic time budget exhausted")
                    if self._calls >= self.max_calls:
                        raise RuntimeError("diagnostic tool budget exhausted")
                    self._calls += 1
                    entry["observation_id"] = f"obs-{self._calls:04d}"
                    output = self.tools.call(params.get("name"), params.get("arguments", {}))
                    output = {
                        **output,
                        "observation_id": entry["observation_id"],
                        "observed_at": datetime.now(UTC).isoformat(),
                    }
                entry.update(status="completed", result=output)
                result = {
                    "content": [{"type": "text", "text": json.dumps(output)}],
                    "isError": False,
                }
            except Exception as exc:
                # Never serialize privileged exception details, which may contain private host paths.
                error = (
                    str(exc)
                    if isinstance(exc, (ValueError, TimeoutError))
                    else "network operation failed or tool budget exhausted"
                )
                entry.update(status="error", error=error)
                result = {"content": [{"type": "text", "text": error}], "isError": True}
            entry["duration_seconds"] = time.monotonic() - started
            with self._lock:
                self.trace.append(entry)
        else:
            return {**response, "error": {"code": -32601, "message": "Method not found"}}
        return {**response, "result": result}

    def start(self, path: Path) -> None:
        broker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                self.request.settimeout(20)
                raw = self.rfile.readline(65537)
                if len(raw) > 65536:
                    return
                try:
                    response = broker.dispatch(json.loads(raw))
                except (ValueError, RecursionError):
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                if response is not None:
                    self.wfile.write(json.dumps(response).encode() + b"\n")

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = False
            block_on_close = True

        self.server = Server(str(path), Handler)
        path.chmod(0o666)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.deadline = 0
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
