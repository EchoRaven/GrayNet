# Primary path congestion case

Two FRR routers have a preferred eBGP link and a backup link.
User traffic and load-generator traffic use the preferred link.
That link has a 40 Mbps shaper in every phase.

During the incident, a separate host sends 35 Mbps of UDP cross traffic.
The user-facing TCP flow should slow down while routing remains intact.
Recovery stops only the competing flow.

```text
user-client ── r1 ══ preferred 40 Mbps ══ r2 ── user-server
                ╚════ backup link ═══════╝
load-client ──┘                       └── load-server
```

The private manifest defines the workload, fault, reference diagnosis, and acceptance thresholds.
The agent receives the shared report and operational context in `environments/two_path/`.
The driver supplies general investigation instructions.
Deployment files and injector controls remain on the host.

## Run on Linux

Install the Linux dependencies and locked Python environment described in the repository README.
The test host used an arm64 Ubuntu 24.04 OrbStack VM.
It ran Containerlab 0.79.0, Docker 29.1.3, and FRR 10.4.1.
The shared topology and FRR configurations are in `environments/two_path/`.

Run these commands from the repository root:

```sh
sudo docker build -t graynet-host:0.1 docker/hosts
sudo docker build -t graynet-codex:0.1 -f docker/codex/Dockerfile .
sudo .venv/bin/python -m graynet check-case cases/primary_congestion
sudo .venv/bin/python -m graynet validate-case cases/primary_congestion
sudo .venv/bin/python -m graynet run-case cases/primary_congestion --agent codex --judge-model gpt-6-astra --auth-file /absolute/path/to/auth.json
```

## Validate the case

`validate-case` deploys the lab and waits for both BGP sessions.
It starts the two workload servers.
It collects five user-throughput samples in each phase and checks routing and reachability.
It removes the lab after recovery.

Validation outputs go under `diagnostics/validation/<case-id>/`.
Agent evaluations go under `results/runs/<case-id>/`.
In both layouts, `result.json` contains the evaluation summary.
`evaluator/evidence.json` contains raw samples and network checks.

The command returns a nonzero exit code if deployment fails or degradation is not repeatable.
It also fails if routing, basic connectivity, or recovery fails.

## Evaluate an agent

`run-case` uses an isolated agent container and an external tool broker.
The backend extends the sustained fault through the agent deadline and measurement windows.
It records the actual fault duration.
It rejects a fault that ends during diagnosis instead of silently restarting it.
The evaluator checks service impact throughout the agent window before scoring.

The [healthy control](../healthy_control/README.md) uses the same network, report, context, tools, and service workload.
It does not start competing traffic.
