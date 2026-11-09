# ATIF v1.7 reference models

The eleven model `.py` files in this directory are copied unchanged from
`/atif_converter/atif/` in SREGym, revision
`1a668ee5aab5c5622ae748fe92c51d9f80458021`:
https://github.com/SREGym/SREGym/tree/1a668ee5aab5c5622ae748fe92c51d9f80458021/atif_converter/atif

SREGym identifies these as vendored Harbor ATIF v1.7 models, originally from
Harbor revision `fd1a8ea6d411b336c9f377aafae1818fe7b18c8d`. Imports between the
models are relative; moving the package did not require changes. Do not edit
model logic to make a GrayNet record pass validation.

`validator.py` is copied from Harbor revision
`fd1521a1da6250d9ed8fc7505caa0b7a72f36c4b` (2026-10-02):
https://github.com/harbor-framework/harbor/blob/fd1521a1da6250d9ed8fc7505caa0b7a72f36c4b/src/harbor/utils/trajectory_validator.py
Its only code change is the model import, from `harbor.models.trajectories`
to this package. An attribution comment records that change.

Harbor is Apache-2.0 licensed; its license is preserved as `LICENSE.harbor`.
SREGym's MIT license is preserved as `LICENSE.sregym`.

The GrayNet adapter in `graynet/artifacts/trajectory.py` follows SREGym's Codex adapter
and its linked tool-call/observation pattern, but reads `codex exec --json`
events, not Codex rollout/session JSONL. Session-level API-call grouping and
per-inference metrics cannot be recovered from this stream and are not invented.
`llm_call_count` stays unknown for Codex steps. Scripted smoke dispatches use
zero. Aggregate usage is retained in `final_metrics`; cached tokens are a subset
of input tokens. Cost stays unknown.

Specification: https://docs.harborframework.com/core-concepts/agents/atif
