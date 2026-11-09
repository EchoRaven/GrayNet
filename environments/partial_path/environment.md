# Network and services

Two TCP services share the routers r1 and r2. Both routers keep BGP sessions
on both inter-router links.

| Service | Client address | Server address | Preferred link |
| --- | --- | --- | --- |
| alpha | alpha-client (10.1.0.2) | alpha-server (10.2.0.2) | r1:eth3 to r2:eth1 |
| beta | beta-client (10.3.0.2) | beta-server (10.4.0.2) | r1:eth4 to r2:eth2 |

A service meets its requirement when its standard TCP transfer reaches at
least 75% of its own healthy median receiver throughput. The healthy
reference contains measurements from the same workload. A single probe
is a sample; report uncertainty when the evidence does not establish a
service violation, which flow is affected, or where the problem is.

Use the available read-only network tools. Report a diagnosis with
observation IDs. Do not change the network.
