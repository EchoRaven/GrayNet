# GrayNet architecture

GrayNet separates case validity from agent performance.
It scores a diagnosis only if the network case produces the intended behavior.
The design follows Terminal-Bench's separate verifier and SREGym's deploy, inject, recover, and independent validation cycle.
Driver adapters use these boundaries for network investigations.

## Package boundaries

`scenarios/` owns private case definitions.
`environments/` implements network deployment, measurement, fault control, and recovery.
`agents/` defines the diagnosis contract and driver registry.
`tools/` exposes public diagnostics through an external broker.
`execution/` coordinates the case lifecycle and experiment matrix.
`evaluation/` contains case admission, classification metrics, and semantic judgment.
`artifacts/` owns recorded attempts and ATIF conversion.
`runtime/` provides command execution and isolated model sessions.
The CLI assembles these components. It does not implement network diagnosis.

The driver registry includes one built-in adapter.
Additional packages can register a factory under the `graynet.drivers` entry-point group.
A factory receives `DriverContext` and `DriverConfig` and returns a managed driver.
The driver must implement `preflight()` and `diagnose()` and expose metadata and a trace.
Execution code does not import a vendor-specific driver.
Every driver must satisfy the same submission and evidence-provenance checks.

## Case files and public inputs

```text
case/
├── scenario.yaml        # host-private workload, fault, ground truth, and thresholds
├── topology.clab.yml    # optional case-specific topology; shared profiles live in environments/
└── environment.md       # optional, explicitly selected operational-context override
```

A case with a named network profile uses `environments/<profile>/incident.md` by default.
This file contains the operator's report of observed symptoms.
The driver supplies general agent instructions separately.

To use a case-specific report, set `incident: incident.md` or another permitted relative path in `scenario.yaml`.
A standalone case without a profile uses its own `incident.md`.
A report must remain inside the case directory or repository `environments/` after symlink resolution.
Cases in each network profile share one report, including the healthy control.

Cases with a network profile share `environments/<profile>/environment.md` by default.
An `environment:` reference selects an explicit override.
The same path and symlink restrictions apply to both public inputs.
The operational context describes managed devices, links, addresses, and service expectations.
It does not include fault selection, injector parameters, the reference diagnosis, or evaluation status.
The runner reads the private manifest on the host.
The agent receives the report, operational context, and network tools.

`check-case` validates the manifest and file references.
It does not check network behavior.
The sample in `tests/fixtures/sustained_congestion` is a format fixture.
It has no implemented fault and is not a validated benchmark case.

## Network profiles and fault mechanisms

Network profiles are in `environments/`.
A profile defines device roles, permitted addresses, BGP peers, expected routes,
the shaped interface, and background workload endpoints.
The topology defines data-plane interfaces, the lab name, and node names.

The backend and tool policy use these definitions instead of fixed node names or private fault labels.
The throughput probe and public answer vocabulary use them too.
The backend verifies router bind mounts and records their sources.
It currently supports Linux, FRR, and bounded iperf3 workloads.
A different router platform requires a backend implementation.

The [partial path profile](partial-path.md) uses two service flows and has
its own workload backend. It reuses deployment and cleanup from the
Containerlab backend. Its evaluator checks each flow separately.

The backend sets and verifies the data-plane MTU.
It disables and verifies TSO, GSO, and GRO on interfaces that it owns.
The two-path profile uses a healthy 10 ms propagation delay and a 40 Mbps shaper.
All cases use the same service endpoint CPU quota and single-CPU placement.
The backend verifies the actual cpuset and quota after deployment and records them.

`environments/faults.py` implements cross traffic, link loss, added delay, and competing CPU work.
A scheduler controls the time pattern of cross traffic.
The evaluator keeps raw injector configuration private.
Agents receive queue counters and CPU observations, not simulator parameters or controller processes.

The tool broker limits Linux and FRR interface views to managed data-plane state.
It removes queue configuration, namespace IDs, Containerlab aliases, management routes, and management sockets.
Failed commands return generic errors instead of private container names or paths.
An observation audit flags known private markers and excludes exposed episodes from scoring.
This guard does not establish that every shortcut is absent.

## Run lifecycle

```text
deploy → baseline samples → inject → incident samples → pre-agent check
                                      ↓ if case has a measured effect
                                  agent diagnosis
                                      ↓
                         recover → recovery samples → final case check
                                      ↓
                            score only if case is valid
```

The backend deploys the network, measures the workload, injects the fault, restores healthy state, and removes the lab.
It measures the same service workload in each phase.
Recovery can repeat safely.
If injection started but recovery did not finish, the runner retries recovery during cleanup.
An agent error on a valid case counts as an unsuccessful evaluation.
It does not exclude the episode.

During diagnosis, a background observer measures service performance and samples queue backlog while traffic is active.
It also checks routing sessions, preferred routes, reachability, and fault state.
Observer measurements and agent throughput probes run one at a time.
This prevents overlapping test flows from causing their own slowdown.

Before each flow, the controller waits for the workload server's next test-generation readiness record.
The [iperf3 server lifecycle](https://github.com/esnet/iperf/blob/3.17.1/src/iperf_server_api.c) closes and reopens listeners between tests.
Client completion does not establish that the server is ready.
This distinction matters when the endpoint has CPU contention.

The controller reads readiness outside the affected container, applies a deadline, and retains the record privately.
Unexpected generations and timeouts count as errors.
The controller does not silently retry failed workloads.

The observer takes periodic samples, so it can miss transient events.
A changed fault state or observer failure prevents scoring.
If service-impact validation fails during diagnosis, the evaluator invalidates the case but still collects recovery measurements.
Probe and observer execution errors remain separate backend errors.
The evaluator retains completed samples and server logs even when a later measurement fails.

## Healthy reference measurements

The agent can use `healthy_reference` to read measured normal throughput, loaded TCP RTT, idle ping RTT, and queue observations.
The returned data has the same structure across incident types.
It does not expose the injector, case admission decision, or answer key.

The public reference always contains five flows, regardless of the private validation sample count.
Its duration and queue counters cover only those flows.
Thus, a longer private history does not reveal the case type through public reference metadata.

A healthy TCP flow can fill the operational queue and increase RTT.
Idle ping RTT is not a target for loaded TCP latency.
Current throughput probes report the same TCP timing type as the healthy reference.
A diagnosis requires current observations as well as any historical reference.

## Case validation

Sustained cases require repeated, nonnegative measurements and a stable healthy baseline.
The median change must cross the specified threshold.
The required fraction of samples must also show degradation.
Recovery must return performance close to baseline.
Healthy controls require stable performance.

Temporal cases use a separate validator.
It matches service samples to actual injector intervals.
It excludes samples that cross transitions from state and trend evidence.

Bursty cases require multiple on/off cycles with degraded on-windows and healthy off-windows.
Gradual cases require enough samples at three or more rates and a worsening throughput trend.
The validator rejects missing coverage, overlapping intervals or service samples, and flat trends.

For a gradual case, the evaluator first runs a preview ramp to validate the incident.
It then checks recovery before a fresh ramp starts for the scored investigation.
This gives the agent access to the ramp, not only its final plateau.
The evaluator records both phases explicitly.
It does not silently restart the fault.
Agent probes use the observer's workload and add timestamped validation samples.

## Diagnosis and evaluation

`evaluation/classification.py` reports detection, localization, and mechanism label agreement.
These metrics do not establish explanation quality.
The private manifest can define equivalent labels, such as either endpoint of a link.
Cases share a public vocabulary derived from managed devices and fault categories,
not from the selected case's answer key.

`evaluation/judge.py` evaluates the explanation with a versioned semantic rubric.
Its private input contains the reference diagnosis, public incident report and context, submitted answer,
and actual cited observations. Uncited observations cannot fill gaps in an explanation.
The rubric checks detection, localization, mechanism, and evidence support.
Each dimension is marked `correct`, `incorrect`, or `insufficient_evidence`.
All dimensions must be correct for the overall diagnosis to pass.

The judge accepts equivalent explanations. It does not require private injector parameters.
The main score concerns causal diagnosis. A correct congestion explanation need
not reconstruct an unreported private on/off schedule. If the agent claims a time
pattern, its observations must support that claim. A reported intermittent symptom
must be explained. Rubric version 2 makes this boundary explicit; version 1 records
retain their original verdicts and remain in separate cohorts.
It must reject contradictory explanations, unsupported conclusions, and instructions
embedded in agent text or tool output. Prompt boundaries reduce this risk but do not
prove immunity to prompt injection. Calibration and human review remain required.

The model backend implements the `JudgmentModel` completion interface.
The current backend uses a separate tool-free container session.
It runs after network recovery and teardown and cannot send feedback to the agent.
The evaluator stores its exact prompt, response schema, raw completion, usage, and judgment.
Failed or malformed model responses are evaluator errors, not incorrect agent diagnoses.
The evaluator never falls back to label agreement.

`run-case` requires an explicit judge model unless classification-only mode is selected.
Version 2 experiment plans freeze the evaluation settings across all agents.
Old version 1 plans are classification-only; historical records are not rescored.
Static case audits remain visibility and answer-contract checks, not semantic judges.

## Agent access boundary

The [primary-path congestion case](../cases/primary_congestion/README.md) uses FRR routers and Linux hosts on two paths.
It checks both BGP peers, preferred routes, ping reachability, cross-traffic delivery,
throughput degradation, queue counters, and recovery.

`run-case --agent codex` starts Codex CLI in a disposable container.
The agent runs as a non-root user.
The container has a read-only root filesystem and public mounts.
It drops all capabilities and blocks privilege gain.

The container does not mount the checkout, evaluator code, Docker socket, or host session state.
An MCP transport connects through a run-specific Unix socket to an external tool broker.
The broker checks diagnostic arguments.
It exposes no arbitrary shell, file access, process access, router mutation, or injector endpoint.
Driver adapters use generic device and interface diagnostics.

The agent has a deadline and a tool-call budget.
Its internal-only Docker network reaches model and authentication hosts through a restricted CONNECT proxy.
It has no general internet route.
The driver checks runtime container policy before it sends the prompt.
`audit-agent-isolation` tests common shortcuts against the actual container and broker.

## Run records and trajectories

The evaluator records timestamps, workload and queue observations, raw iperf output,
image IDs and digests, runtime versions, and source snapshots.
It also records configuration hashes, the Git revision, and whether the checkout has local changes.
Snapshots include shared topology and nested runtime sources.
They exclude adjacent credentials and bytecode caches.

Agent records contain the exact prompt, answer schema, configuration, JSONL events, tool calls,
submission, elapsed time, and token usage.
Missing usage remains unknown.
The driver does not invent a dollar cost for subscription usage.

Each attempt has one directory.
`run.json` records identity, status, and artifact hashes.
`result.json` contains the evaluation summary.
`agent/` contains inputs, trajectories, and logs.
`evaluator/evidence.json` contains network measurements and reproducibility records.
`evaluator/judge/` contains the semantic evaluator's input, raw response, and verdict.
The artifact checker compares this verdict with the raw response and checks the
judge's input against the agent's submitted evidence. It does not rerun the model
or determine whether its judgment is reliable.

Individual evaluations default to `results/runs/<case-id>/<run-id>/`.
Experiment trials use `results/experiments/<experiment>/runs/`.
Validation and audits use the separate Git-ignored `diagnostics/` tree.
An explicit output directory overrides these defaults.

The canonical `agent/trajectory.json` uses ATIF v1.7 models from SREGym.
Observations reference their tool calls.
The `extra` fields contain custom metadata.
`trajectory.md` gives a readable view of the same ATIF record.

The summary does not duplicate large evidence records.
A preflight error keeps a failed attempt manifest without an evaluation result.
Historical imports preserve originals and record provenance.
An import is not a new execution.

## Repeated experiments and aggregation

`execution/experiments.py` freezes a case, model, effort, budget, judge, and repetition matrix before execution.
It runs trials one at a time.
It records trial order, loss seeds, source hashes, every attempt, and aggregate outcomes.

The runner refuses to resume an experiment with changed inputs.
It does not silently replace incomplete trials.
Summaries separate revisions and retain agent errors on valid cases.
They report per-case results and binomial intervals.
Evaluator errors are reported separately and do not enter the semantic success denominator.
Each summary records both admitted episodes and scored episodes.
Judge identities include the model, deadline, image, rubric, and prompt hash.
Historical classification-only cohorts remain separate from semantic cohorts.

Aggregation separates actual network image IDs and host resource settings.
Mutable image tags alone do not define a comparable environment.
A small integration pilot does not estimate general benchmark performance.

## Design references

- [Terminal-Bench 2 task format](https://docs.harborframework.com/core-concepts/tasks/overview) and [separate verifier](https://docs.harborframework.com/core-concepts/tasks/separate-verifier)
- [iproute2 netem behavior and limits](https://man7.org/linux/man-pages/man8/tc-netem.8.html)
- [SREGym's problem lifecycle](https://github.com/SREGym/SREGym/blob/045c1032f856f334a1caddc2a8892140be047384/sregym/conductor/problems/base.py) and [integration validator](https://github.com/SREGym/SREGym/blob/045c1032f856f334a1caddc2a8892140be047384/tests/integration/validate_problem.py)
- [SREGym Codex driver](https://github.com/SREGym/SREGym/blob/1a668ee5aab5c5622ae748fe92c51d9f80458021/clients/codex/driver.py), [adapter](https://github.com/SREGym/SREGym/blob/1a668ee5aab5c5622ae748fe92c51d9f80458021/clients/codex/codex_agent.py), and [container runner](https://github.com/SREGym/SREGym/blob/1a668ee5aab5c5622ae748fe92c51d9f80458021/sregym/service/container_runner.py)
