# Matched healthy control

This case uses the same topology, FRR configurations, workload, context, incident report, and tools as `primary_congestion`.
It does not start a competing flow.
Throughput must stay close to baseline during the nominal incident, investigation, and recovery.

The case tests false alarms.
A reported slowdown or a normal link shaper does not establish a fault.
This control does not introduce a separate topology or fault mechanism.

```sh
sudo .venv/bin/python -m graynet run-case cases/healthy_control --agent codex --judge-model gpt-6-astra --auth-file /absolute/path/to/auth.json
```

Run this case separately from the congestion case.
Both use the same lab name.
