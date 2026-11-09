# GrayNet network degradation benchmark

GrayNet is a benchmark for LLM agents that diagnose network degradation.
The benchmark is under development. It uses a Containerlab network with FRR routers.
Agents investigate each incident through network tools.
A separate evaluator checks case validity and scores the agent's diagnosis.

## Code structure

The package separates responsibilities:

```text
graynet/
  scenarios/       private case definitions and validation settings
  environments/    network deployment, workloads, and fault controls
  agents/          diagnosis contract and registered driver adapters
  tools/           public diagnostics and the external tool broker
  execution/       case lifecycle and frozen experiment plans
  evaluation/      case admission, classification metrics, and semantic judgment
  artifacts/       run records, ATIF export, and historical import tools
  runtime/         subprocess, container-session, and isolation policy
  cli.py           command parsing and component assembly
```

Shared topology and operational inputs are in `environments/`.
Container images are in `docker/`.
Generated runs are not source code and are ignored by Git.

## Cases and shared environment

The original seven cases use one two-path network and the same operational context:

| Case | Injected mechanism | Independent checks |
| --- | --- | --- |
| `healthy_control` | None | Stable service performance |
| `primary_congestion` | Sustained cross traffic | Traffic delivery, queue buildup, throughput loss |
| `link_loss` | 2% loss on the server access link | Drop counters, throughput loss |
| `link_delay` | 200 ms added access-link delay | RTT increase, no unintended drops, throughput loss |
| `endpoint_overload` | Competing CPU work | CPU saturation, throughput loss |
| `bursty_congestion` | 10-second on/off cross traffic | Degraded on-windows, healthy off-windows, repeated cycles |
| `gradual_congestion` | Increasing cross-traffic rates | Coverage of rate levels and a worsening throughput trend |

Each case requires a stable healthy baseline, intact routing, basic reachability,
and recovery. The shared topology and router configurations are in
`environments/two_path/`.

The [partial path cases](docs/partial-path.md) use a separate network with
two active service flows. See the guide for case checks and the paired
comparison of fixed and selectable throughput probes.

The original cases use the operator's report in `environments/two_path/incident.md`.
The driver supplies general investigation instructions.
To use a different report, specify an `incident:` path in the case manifest.

The environment uses a 1500-byte data-plane MTU and a fixed healthy propagation delay.
The backend disables packet offloads and verifies their state.
It applies the same endpoint CPU quota and single-CPU placement in every phase.
It also verifies and records the actual placement.
Independent case validation determines these settings, not agent scores.

The runner reports case validity, agent completion, and diagnosis correctness separately.
It excludes invalid cases and backend errors.
Agent errors on valid cases count as unsuccessful evaluations.

## Local checks

Install Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
Run these commands from the repository root:

```sh
uv sync
uv run graynet check-case tests/fixtures/sustained_congestion
uv run pytest
uv run ruff check graynet tests scripts docker
uv run ruff format --check graynet tests scripts docker
```

`check-case` validates the manifest and its file references.
It does not deploy the network.
The sample in `tests/fixtures/sustained_congestion` is a format fixture, not a verified network case.

## Evaluate agents on Linux

The Linux host needs cgroup v2, Docker, Containerlab, `ethtool`, `nsenter`, and iproute2.
The `nsenter` command comes from util-linux.
The host also needs Python 3.11 or later, PyYAML, and Pydantic v2.

Run `uv sync --locked` on Linux to install the locked Python dependencies.
Use the Linux environment's Python for the commands below.
If Linux and macOS share the checkout, create a separate Linux environment.
For example, use `UV_PROJECT_ENVIRONMENT=/opt/graynet-runtime/venv uv sync --locked`.
Do not use a macOS virtual environment on Linux.

### Current driver example

The commands below use the currently implemented driver.
Supply its CLI login credentials as an `auth.json` file.
This driver supports subscription and API-key authentication.

Run these commands from the repository root:

```sh
sudo docker build -t graynet-host:0.1 docker/hosts
sudo docker build -t graynet-codex:0.1 -f docker/codex/Dockerfile .
sudo .venv/bin/python -m graynet audit-agent-isolation
sudo .venv/bin/python -m graynet run-case cases/primary_congestion --agent codex --judge-model gpt-6-astra --auth-file /absolute/path/to/auth.json
sudo .venv/bin/python -m graynet run-case cases/healthy_control --agent codex --judge-model gpt-6-astra --auth-file /absolute/path/to/auth.json
```

The commands use `.venv/bin/python`. Substitute your Linux environment's path if it differs.
Run cases one at a time because they use the same lab name.
The runner refuses to reuse an existing lab.

Optional arguments are `--model`, `--reasoning-effort`, `--timeout`,
`--max-tool-calls`, and `--output`.
This driver reads default model settings from the CLI account's catalog and records them.
It rejects unsupported explicit choices before deployment.
Use the same explicit model and effort for comparative runs.
Semantic evaluation requires an explicit `--judge-model`.
Use one fixed judge across agents in a comparison.
`--evaluation classification` skips the model judge and reports label agreement only.
It does not establish that an explanation is correct.

## Agent access

The current agent runtime uses a separate container with a non-root user and a read-only root filesystem.
It receives the incident report, operational context, and network tools.
It cannot access the repository, answer key, evaluator, injector, Docker socket,
or host session history.

An external tool broker checks diagnostic arguments through a Unix socket.
The agent has no direct internet route.
A restricted proxy permits model and authentication traffic.
The driver makes temporary credential copies and removes them after use.
It does not include credentials in images or results.

## Run details

Each agent attempt has a directory under `results/runs/<case-id>/<timestamp-command-id>/`.
Experiment batches use `results/experiments/<experiment>/`.
Each batch contains `runs/`, `experiment.json`, and `summary.json`.
The runner retains unsuccessful attempts and never overwrites a run directory.

Each agent run has this structure:

```text
run.json                identity, lifecycle status, artifact paths and hashes
result.json             case validity, agent outcome, classification, judgment, and metadata
agent/
  trajectory.md         readable investigation, including tool outputs
  trajectory.json       validated ATIF v1.7, with linked calls and observations
  tools.json            timestamped broker observations and denied calls
  diagnosis.json        submitted answer, if present
  metadata.json         resolved model, isolation, timing, and tokens
  input/                incident.md, complete prompt.txt, environment, config, schema
  logs/                 raw driver events, stderr, and proxy log
evaluator/evidence.json private measurements, checks, and source/runtime snapshots
evaluator/judge/         private judge input, rubric, raw output, usage, and verdict
```

`--output` and its alias `--run-dir` specify a new directory, not a JSON file.
A preflight failure keeps `run.json` with the error and stage.
It does not create a false evaluation result.

Development checks use separate directories:
`diagnostics/validation/`, `diagnostics/audits/`, and
`diagnostics/stress/`. Git ignores both `results/` and `diagnostics/`.
Isolation audits contain `audit.json`, not a scored `result.json`.

## Inspect and convert results

```sh
graynet summarize results/runs/<case-id>/<run-id>
graynet summarize results/runs/<case-id>
graynet import-results /path/to/old-run.json
graynet convert-trajectories diagnostics/imported
graynet check-trajectory results/runs/<case-id>/<run-id>/agent/trajectory.json
```

Legacy imports create copies under `diagnostics/imported/<case-id>/`.
They preserve original bytes, scores, timestamps, and provenance.
The import does not modify the source files.

Directory summaries read finished `result.json` records.
They do not treat raw logs, audits, or archived originals as separate evaluations.
They also remove duplicates from overlapping input selections.
Explicit legacy JSON files remain supported.

Select comparable runs for research estimates.
Do not combine development attempts with research episodes.
Summaries separate models, efforts, drivers, budgets, and answer-contract hashes.

The ATIF models come from SREGym's standalone converter.
`check-trajectory` uses Harbor's vendored validator.
Conversion preserves scores and raw logs, then updates artifact hashes in the manifest.
It archives previous custom views under `agent/legacy/pre-atif/`.
With `--refresh`, conversion also regenerates existing ATIF and archives the previous JSON and Markdown.

A valid, completed diagnosis returns exit code zero even when the answer is wrong.
Agent errors, evaluator errors, invalid cases, and backend errors return a nonzero exit code.
`validate-case` runs without an agent.

## Run repeated experiments

```sh
uv run graynet audit-case cases/link_loss
uv run graynet plan-experiment experiments/pilot.yaml
sudo .venv/bin/python -m graynet run-experiment experiments/pilot.yaml --output results/experiments/pilot --auth-file /absolute/path/to/auth.json
sudo .venv/bin/python -m graynet resume-experiment results/experiments/pilot --auth-file /absolute/path/to/auth.json
```

The YAML configuration specifies cases, models, efforts, budgets, repetitions, and seeds.
The runner creates a trial matrix before execution and shuffles its serial order.
It retains every attempt in `experiment.json`, `runs/`, `logs/`, and `summary.json`.

Packet-loss seeds change with each repetition.
The ordering seed controls trial order, not deterministic Linux execution.
The runner freezes hashes of scenario and runtime inputs.
It rejects changed inputs, overwrites, and silent retries of unfinished trials.
Resume executes only pending trials.
A cleanup failure stops further dispatch.

## Evidence and evaluation integrity

Each tool observation has an ID.
The agent must cite IDs from successful network observations.
Invented, duplicate, inventory-only, and failed-tool references violate the answer contract.

These checks establish evidence provenance, not causal correctness.
The separate semantic judge compares the explanation with a private reference diagnosis
and the actual cited observations. It evaluates detection, localization, mechanism,
and evidence support. Matching labels alone cannot pass this evaluation.
The judge runs after recovery and teardown. It has no network tools.
An evaluator error remains an evaluator error; there is no label-only fallback.
The model judgment still requires calibration and human review before research use.

Queue tools return counters, not simulator configuration.
Linux and FRR interface views exclude emulator and management metadata.
Route and socket views exclude management traffic.
Known private markers in returned observations invalidate an episode.
This guard does not cover every possible shortcut.

The controller collects CPU telemetry outside the affected container.
Thus, endpoint contention cannot starve the telemetry command.
`healthy_reference` returns five measured healthy observations under the same workload.
It reports loaded TCP RTT and idle ping RTT separately.

A healthy, saturating TCP flow can increase queue occupancy and cause some drops.
Historical measurements do not establish the current network's health.
A diagnosis also requires current observations.

## Summaries and study configurations

The summary reports per-case outcomes, failure categories, time, tool calls, and 95% Wilson intervals.
It checks recorded artifact hashes before aggregation.
Agent errors on valid cases remain in the success denominator.
Evaluator failures are not agent failures. Summaries report them separately and
exclude them from `scored_episodes`. Inspect this count before comparing success rates.
Historical label-only runs remain label-only. The evaluator does not rescore them.

Summaries separate environment and answer-contract revisions.
They also separate runtime code, tool definitions, network and agent image IDs,
host software versions, and actual endpoint resource settings.
Semantic cohorts also separate judge model, image, rubric, and prompt revisions.

The one-repetition pilot is an integration test.
Research comparisons need a trial matrix declared in advance and repeated runs.
They also need manual review of failures.

`experiments/comparison.yaml` defines two models with five repetitions.
A plan does not launch model evaluations.

## Development checks

Run these checks on Linux with Docker privileges:

```sh
sudo .venv/bin/python scripts/audit_network_tools.py cases/link_loss --output diagnostics/audits/diagnostic-views
sudo .venv/bin/python scripts/stress_workloads.py cases/endpoint_overload --samples 40 --output diagnostics/stress/endpoint-stress
python scripts/check_experiment_artifacts.py results/experiments/pilot
```

The command-view audit checks all permitted commands on healthy and impaired networks.
It does not spend model tokens.
The stress check repeats incident measurements with the original thresholds.
It keeps the normal healthy and recovery phases.
It does not run an agent, retry failed workloads, or estimate model performance.
The artifact checker also checks recorded semantic verdicts against raw judge
responses and cited observations. It does not rescore historical runs or establish
the model judge's reliability.

See [architecture](docs/architecture.md) and
[case details](cases/primary_congestion/README.md).
