# Network operations context

The managed routers are FRR devices `r1` and `r2`.
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
| r1:eth2 | 10.3.0.1/24 | background-traffic network |
| r1:eth3 | 10.12.0.1/30 | r2:eth1 |
| r1:eth4 | 10.12.1.1/30 | r2:eth2 |
| r2:eth1 | 10.12.0.2/30 | r1:eth3 |
| r2:eth2 | 10.12.1.2/30 | r1:eth4 |
| r2:eth3 | 10.2.0.1/24 | user-server:eth1 |
| r2:eth4 | 10.4.0.1/24 | background-traffic network |
| user-server:eth1 | 10.2.0.2/24 | r2:eth3 |

The routers use eBGP with AS 65001 and AS 65002 over both inter-router links.
Normal routing prefers the `10.12.0.0/30` link. The `10.12.1.0/30` backup link has a lower capacity of about 15 Mbps.
Use the network tools to inspect managed interfaces.
Background workload controllers are not managed devices.
Interface counters still show their traffic.

Identify the affected component with a device/interface identifier, such as `r1:eth1`.
You can also identify a link by its two endpoints.
Do not change network configuration.
This task is diagnosis only.
Describe the mechanism in plain language and cite observations from the investigation.
