# Partial path degradation

A network operator receives a report of a slow transfer. The report does not
identify the service. The agent checks whether either service falls below its
throughput requirement. It also reports which flow slowed down and where it
found the problem, if the observations support those conclusions.

## Network and case checks

The Containerlab network has two FRR routers and two TCP services, `alpha`
and `beta`. Both router links carry traffic. BGP route preferences direct
each service along a different link in both directions. Each router also
learns a route through the other link.

The healthy control leaves both links unchanged. The other cases apply
steady packet loss on one inter-router link. Both links use the same type
of queue, and the fault changes the loss setting on the selected link.
The broker returns queue counters without queue types, loss settings, or
random seeds.

Each phase measures five pairs of simultaneous four-second TCP transfers.
The validator checks BGP sessions, routes, and basic reachability for both
services. It also captures a packet from each flow on its expected receiving
interface. During a fault, it checks the loss rate, the random seed, and
the increase in drop counters. After it removes the fault, it measures
both flows again.

The service requirement is 75% of the healthy median throughput for that
flow. The validator records the measured slowdown and the service violation
separately. A mild fault slows the flow, but the service still meets its
requirement. A near or clear fault violates the requirement. The other flow
must stay within 8% of its healthy median in at least four of five samples.

## Probe comparison

Both trial arms use the same report, topology, healthy reference procedure,
and read-only diagnostic tools. The fixed arm can probe `alpha`. The
choice arm can probe `alpha` or `beta`. Each probe runs the same two
transfers and returns the measurement for the selected flow.

The plan pairs fresh deployments with the same case and loss seed. It
reverses arm order in the second repetition for each case. The fixed probe
covers the impaired flow in half of the fault cases. Both arms use nine tool
calls and a 60-second budget. The baseline reads the healthy reference,
samples queue counters before and after its tests, and makes four throughput
probes. It divides the probes evenly among the permitted flows.

The agent can state that the service status, affected flow, or location is
unknown. The evaluator checks each answer separately. It also checks whether
the cited measurements support each claim. To clear both services, the agent
must measure both. To support a location, it must cite a slowdown on the
corresponding flow and an increase in drops on the link during a probe.
The check covers these structured claims. It does not judge free text.

The baseline uses no model. Its results check the experiment procedure
and show what the available measurements can support. They do not estimate
LLM performance.

## Commands

Use `plan-probe-experiment` to check the plan without deploying a network.
Run the other two commands on a Linux host that has GrayNet, Docker,
Containerlab, and `tcpdump`. Use a new output directory for each command.

```sh
graynet plan-probe-experiment experiments/partial_path.yaml
sudo .venv/bin/python -m graynet validate-case cases/partial_path_beta_near --output /path/outside/repo/beta-near-check
sudo .venv/bin/python -m graynet run-probe-experiment experiments/partial_path.yaml --case-id partial-path-beta-clear --repetition 1 --output /path/outside/repo/probe-pilot
```

Omit the case and repetition filters to run the full plan. See `plan.json`
for the paired inputs and `summary.json` for the comparison. Each arm has
its own run directory with the diagnosis, tool calls, ATIF trajectory, and
private validation evidence.
