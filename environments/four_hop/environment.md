# Network operations context

The managed routers are FRR devices `r1`, `r2`, `r3`, and `r4`, connected in a chain.
The managed Linux hosts are `user-client` and `user-server`.
The service sends TCP traffic from `10.1.0.2` to `10.2.0.2`.

Under the standard single-flow workload, healthy throughput is normally approximately 40 Mbps.
The service expects throughput to remain within 20% of its measured healthy reference under the same workload.
Compare repeated samples to distinguish degradation from normal variation.
Idle ping RTT is normally approximately 10 ms.
The standard TCP workload can increase RTT and queue occupancy during normal operation.

Use `healthy_reference` to compare measurements under the same workload.
Compare idle ping RTT with idle ping RTT, and loaded TCP RTT with loaded TCP RTT.
Do not use idle RTT as a target for loaded latency.
An established routing session or successful ping alone does not establish acceptable service performance.

| Device/interface | Address | Neighbor |
| --- | --- | --- |
| user-client:eth1 | 10.1.0.2/24 | r1:eth1 |
| r1:eth1 | 10.1.0.1/24 | user-client:eth1 |
| r1:eth2 | 10.12.0.1/30 | r2:eth1 |
| r2:eth1 | 10.12.0.2/30 | r1:eth2 |
| r2:eth2 | 10.23.0.1/30 | r3:eth1 |
| r2:eth3 | 10.3.0.1/24 | background-traffic network |
| r3:eth1 | 10.23.0.2/30 | r2:eth2 |
| r3:eth2 | 10.34.0.1/30 | r4:eth1 |
| r3:eth3 | 10.4.0.1/24 | background-traffic network |
| r4:eth1 | 10.34.0.2/30 | r3:eth2 |
| r4:eth2 | 10.2.0.1/24 | user-server:eth1 |
| user-server:eth1 | 10.2.0.2/24 | r4:eth2 |

The routers use eBGP between adjacent routers, with AS numbers 65001 through 65004.
Use the network tools to inspect managed interfaces.
Background workload controllers are not managed devices.
Interface counters still show their traffic.

Identify the affected component with a device/interface identifier, such as `r1:eth1`.
You can also identify a link by its two endpoints.
Do not change network configuration.
This task is diagnosis only.
Describe the mechanism in plain language and cite observations from the investigation.
