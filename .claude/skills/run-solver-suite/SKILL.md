---
name: run-solver-suite
description: >-
  Run the full M1 gated regression suite (solver, accuracy bench, swerve,
  collision, trajectory) with the correct interpreter and report a pass/fail
  table with metrics. Use after changing the solver or when asked for the
  regression metrics.
disable-model-invocation: true
---

# Run the M1 solver regression suite

Run every gated suite from the **repo root** with **`/usr/bin/python3`** (ROS 2
Jazzy, Python 3.12 — bare `python3` is conda 3.13 and is wrong). Allow generous
timeouts; the cold multi-seed IK suites are slow. Do not stop early on the first
failure — run them all, then summarize.

```bash
/usr/bin/python3 _solver_test.py
/usr/bin/python3 _solver_test_positions.py
/usr/bin/python3 _solver_test_tracking.py
/usr/bin/python3 _solver_test_pathing.py
/usr/bin/python3 _accuracy_bench.py
/usr/bin/python3 _swerve_test.py
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.collision
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.trajectory
```

Optional (ROS import; no DDS needed): `/usr/bin/python3 _quest_position_test.py`.

Each script prints `[PASS]/[FAIL]` gate lines, an `N/N gates passed` total, and
exits 0 only if all gates pass.

## Report
Produce one table: **suite → gates (N/N) → PASS/FAIL → headline metrics** (sub-mm
%, mean/max error in mm, worst-case `solve_step` ms, p99 latency, % over the
60 Hz budget — whatever that suite prints). End with a one-line overall verdict,
and for any failure quote the exact failing gate name and the metric that missed.
Distinguish a suite that *errored* (import/dep problem) from one that *failed a
gate*.

Consider delegating the actual run to the `solver-suite-runner` subagent if you
want it off the main context.
