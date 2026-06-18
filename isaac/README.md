# Ranger Air in Isaac Sim

Scripts to import the Ranger Air mobile manipulator (mobile base + lift + dual
OpenArm arms) into NVIDIA Isaac Sim and run physics simulations.

## Environment (this machine)

- **Isaac Sim 5.1.0** installed natively at `/home/jerry/isaac-sim`
- Platform: DGX Spark, `aarch64`, NVIDIA GB10 GPU (driver 580 / CUDA 13)
- Also available: Isaac Sim 5.0 / Isaac Lab 2.3 Docker images, and IsaacLab
  source at `/home/jerry/IsaacLab`

All scripts run with Isaac Sim's bundled Python via `python.sh` (do **not** use
the base conda env). Run everything from the repo root.

## 1. Convert the URDF to USD

```bash
cd /home/jerry/Downloads/M1-visualizer
/home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py
```

This reads `assets/ranger_air_description/urdf/ranger_air_description.urdf` and
writes a layered USD to `assets/usd/ranger_air.usd` (geometry, physics,
articulation, and sensor layers live under `assets/usd/configuration/`). The
`package://` mesh references for both `ranger_air_description` and
`openarm_description` are resolved automatically.

Useful flags:

- `--fix-base` — pin the base in place (default is a free/mobile base)
- `--urdf PATH` / `--usd PATH` — override input/output paths

## 2. Run a simulation

Headless smoke test (no window):

```bash
/home/jerry/isaac-sim/python.sh isaac/run_sim.py --headless --steps 300
```

Interactive window with a wheel-drive demo (needs a display):

```bash
/home/jerry/isaac-sim/python.sh isaac/run_sim.py --demo
```

Flags:

- `--headless` — no GUI window
- `--steps N` — number of frames to simulate
- `--demo` — apply an angular velocity drive to the four wheels
- `--spawn-height M` — height above the ground the base is dropped from
- `--usd PATH` — load a different robot USD

A status summary (articulation path, DOF count/names) is written to
`isaac/last_run_report.txt` because Isaac Sim redirects Python stdout.

## 3. Teleoperate every joint

`isaac/teleop.py` opens an interactive window and lets you drive all 27 DOFs
from the keyboard (wheels, steering, lift, both 7-DOF arms, both grippers):

```bash
/home/jerry/isaac-sim/python.sh isaac/teleop.py
```

The window must have focus to receive keystrokes. Keyboard map:

- **Base (swerve):** `W`/`S` drive forward/reverse, `A`/`D` turn left/right (in
  place when stopped, or arc while driving), `Q`/`E` strafe (crab) left/right,
  `C` re-center wheels & stop, `SPACE` stop the base
- **Lift:** `R`/`V` raise/lower
- **Arms:** `TAB` switch active arm (LEFT/RIGHT), `1`–`7` select a joint of the
  active arm (`[`/`]` cycle), `UP`/`DOWN` jog the selected joint
- **Grippers:** `O`/`K` open/close the active arm's gripper
- **Misc:** `H` reset to the default pose, `ESC` quit

Flags:

- `--fix-base` — pin the base so it can't roll while you work the arms
- `--spawn-height M` / `--usd PATH` — same as `run_sim.py`

The base is driven as a swerve platform: a body-velocity command (forward,
strafe, yaw) is solved into a heading + spin for each of the four steer/wheel
modules, so the same controls give straight driving, crab strafing, turning in
place, and arcing while moving. Wheels use a velocity drive; everything else
uses position drives with per-group gains. If a wheel spins or a corner steers
the wrong way, flip its sign in the `WHEEL_DIR` / `STEER_DIR` dicts near the top
of `teleop.py`; base speed/turn limits live in the `MAX_*` constants there. A
status summary is written to `isaac/last_teleop_report.txt`.

## Robot summary

The imported articulation has **27 DOFs**:

- 4 steering joints (`*_steering_joint`)
- 4 wheels (`*_wheel_joint`)
- 1 prismatic lift (`lift_joint`)
- 2 × 7 arm joints (`openarm_{left,right}_joint1..7`)
- 4 gripper finger joints (`openarm_{left,right}_finger_joint{1,2}`; the
  `finger_joint2` joints mimic `finger_joint1`)

Articulation root prim: `/World/RangerAir/base_link/base_link`.

## Next steps / ideas

- Drive the base by writing velocity targets to the wheel/steering joints.
- Control the arms with joint position targets via the `Articulation` API.
- For RL / learning workflows, wrap the USD as an Isaac Lab asset
  (`/home/jerry/IsaacLab`).
