# M1 ROS 2 — project guide for Claude

The full handoff doc lives in **@AGENTS.md** (architecture, every node, test
status, gotchas). Read it before non-trivial work. This file pins the few rules
agents break most and the commands you'll reuse.

## Hard rules (agents keep breaking these)

1. **Interpreter.** Run ROS / solver scripts with **`/usr/bin/python3`** (ROS 2
   Jazzy, Python 3.12). `python3` on PATH is **conda 3.13** — wrong numpy, no
   `rclpy`, no Jazzy messages. Write the command right the first time. Standalone
   solver/collision modules need `PYTHONPATH=ros2_ws/src/m1_control`.
2. **The arm reach is POSITION-ONLY.** A target is a 3D point (or a dict with
   `"pos"`); any `"quat"`/`"R"` is ignored. Don't reintroduce orientation rows
   into the solve unless explicitly asked.
3. **60 Hz is a goal, not a hard cutoff.** Accuracy first; don't re-tighten
   latency gates into a hard real-time cutoff.

## Solver backend: Drake

The Cartesian reach IK in `kinematics.py` is solved by **Drake** (`pydrake`):
a `MultibodyPlant` + `InverseKinematics` position-cost solve with an amortized
multi-start. It's installed for the ROS interpreter with:

```bash
/usr/bin/python3 -m pip install --user --break-system-packages drake
```

`pydrake` is imported lazily/at controller construction; the FK utilities
(`UrdfModel`/`ArmChain`) stay dependency-free, so FK-only viz nodes still work
even without Drake.

## Common commands

```bash
# Full solver/robot regression (gated, prints N/N gates passed):
/usr/bin/python3 _solver_test.py
/usr/bin/python3 _solver_test_positions.py
/usr/bin/python3 _solver_test_tracking.py
/usr/bin/python3 _solver_test_pathing.py
/usr/bin/python3 _accuracy_bench.py
/usr/bin/python3 _swerve_test.py
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.collision
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.trajectory

# Hardware bridge + CAN tool tests (hardware-free):
PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 ros2_ws/src/m1_control/_bridge_test.py
PYTHONPATH=ros2_ws/src/m1_can_tools /usr/bin/python3 -m pytest ros2_ws/src/m1_can_tools/test -v

# Build the ROS workspace:
source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon build --symlink-install
source install/setup.bash

# Real-hardware / mock ros2_control bring-up (replaces the Isaac sim driver):
ros2 launch m1_bringup hardware.launch.py use_mock:=true   # offline mock_components
ros2 launch m1_bringup hardware.launch.py use_mock:=false can_interface:=can0 motor_map:=...  # real Damiao
ros2 run m1_can_tools m1_hwconfig                            # motor config/test page :8090
```

**Real-hardware deployment** lives in **@ros2_ws/HARDWARE.md** (Damiao CAN motors +
AgileX base; the `m1_hardware` C++ ros2_control plugin, `m1_can_tools` config page,
and the bridge nodes). Standalone `m1_can_tools` modules need
`PYTHONPATH=ros2_ws/src/m1_can_tools`. **Cleaning up a launched ros2 stack:** SIGINT
the `ros2 launch` then sweep leftover PIDs — never `pkill -f <node-name>` (the
pattern matches your own shell's command line and SIGKILLs your shell).

Use `/run-solver-suite` to run the whole gated suite at once, and `/colcon-build`
to build. The `kinematics-reviewer` and `solver-suite-runner` subagents exist for
reviewing the numerical solver and running regressions.

## Don't commit build artifacts

`__pycache__/`, `*.pyc`, and `ros2_ws/{build,install,log}/` are gitignored. Don't
re-add them.
