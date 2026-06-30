# M1 real-hardware bring-up

How to drive the real robot (Damiao CAN motors for arms + lift; AgileX base)
instead of Isaac Sim. The control brain (`m1_control`) and every operator
interface (`m1_web`, `m1_quest`, `m1_teleop`, `m1_send_pose`) are **unchanged** —
they speak only `/m1/*` + `/joint_states`. On hardware those topics are served by
a `ros2_control` stack + bridge nodes instead of `isaac/ros_sim.py`.

See `docs/superpowers/specs/2026-06-23-real-hardware-deployment-design.md` for the
full design and the OpenArm-vs-Drake comparison that motivated it.

## Architecture (the seam)

```
operators ──/m1/*──► controller_node (Drake IK + swerve, 60 Hz) ──► /m1/joint_command
                                                                          │
              ┌───────────────────────────────────────────────────────────┤
              ▼ upper body (lift+arms+grippers)                            ▼ base
   m1_joint_bridge ─► ros2_control                  m1_base_bridge ─/cmd_vel(Twist)─► agx_bringup (vendored)
   forward_position_controller (+ JTC for planned moves)                   │  steering_angles/wheel_speeds
              │ loaned-memory                                              │ CAN feedback
              ▼                                                            ▼
   m1_hardware/M1SystemInterface (Damiao MIT, SocketCAN) + lift   ranger_state_shim ─► /joint_states (base)
              │                                                            ▲
   joint_state_broadcaster ─► /joint_states (upper body) ─────────────────┘
```

The brain is unchanged: it still publishes the full 27-DOF `/m1/joint_command`.
`m1_joint_bridge` forwards the **17 commanded** upper-body joints to the position
controller; the base entries are ignored (the base is driven over its own Twist
path). `/joint_states` is the union of the broadcaster (upper body) + the base
shim.

## Quick start

```bash
cd ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --symlink-install && source install/setup.bash

# 1) MOCK (no hardware): validate the whole stack with ros2_control mock_components.
ros2 launch m1_bringup hardware.launch.py use_mock:=true use_rviz:=true

# 2) REAL motors (arms + lift), base off:
#    (bring up the CAN bus first -- see "CAN setup" below)
ros2 launch m1_bringup hardware.launch.py use_mock:=false \
     can_interface:=can0 can_fd:=true motor_map:=$HOME/.config/m1/motor_map.yaml

# 3) REAL + base:
ros2 launch m1_bringup hardware.launch.py use_mock:=false use_base:=true ...
```

Then the operator interfaces, exactly as in sim:
```bash
ros2 run m1_control m1_web       # or m1_quest / m1_teleop / m1_send_pose
```

## Bus-ownership model (IMPORTANT)

Motor **configuration** and live **ros2_control** are mutually exclusive on a CAN
bus — never run both at once:

- **Maintenance mode** — the `m1_hwconfig` Python tool owns the bus. Use it to
  scan, assign IDs, set zero, edit limits, and jog/test motors. `ros2_control`
  must be DOWN.
- **Run mode** — the `ros2_control` stack owns the bus (the launch above). The
  config page goes read-only (telemetry from `/joint_states`).

## CAN setup

Two transports are supported (the choice decides the host stack):

- **SocketCAN (recommended, CAN-FD):** a CANable/candleLight, Innomaker, or PCAN
  adapter presents as `can0`. Bring it up:
  ```bash
  sudo ip link set can0 up type can bitrate 1000000                      # classic CAN
  sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on   # CAN-FD
  ```
  Both `m1_hardware` (C++, openarm_can) and `m1_hwconfig` (`transport:=socketcan`)
  use this. Install python-can for the config tool's real path:
  ```bash
  /usr/bin/python3 -m pip install --user --break-system-packages python-can
  ```
- **DAMIAO USB2CAN serial dongle:** presents as `/dev/ttyACM0` @ 921600 with the
  vendor 0xAA/0x55 framing — NOT SocketCAN. Use `m1_hwconfig transport:=serial`.
  (The C++ `m1_hardware` path targets SocketCAN; the serial dongle is a bench/
  bring-up convenience via the Python tool.)

## Motor configuration workflow (`m1_hwconfig`)

```bash
# maintenance mode (default), fake transport for a dry run, or socketcan/serial:
ros2 run m1_can_tools m1_hwconfig                       # -> http://localhost:8090
ros2 run m1_can_tools m1_hwconfig --ros-args -p transport:=socketcan -p can_channel:=can0
```
The page lets you: **scan** the bus, set/verify each motor's **CAN ID + master
ID**, **map** each motor → logical joint (e.g. `openarm_left_joint3`), edit
per-joint **limits** (writes a `joint_limits.yaml`), **jog/test** a motor (clamped
slider + dead-man), **set-zero**, and watch **live telemetry** (pos/vel/torque/
MOS+rotor temp/error). The motor→joint map is saved to a YAML you then pass to the
launch as `motor_map:=...`.

## Wiring the motor map into the controller (bring-up TODO)

The `m1_hardware/M1SystemInterface` plugin sources each joint's `can_id` /
`master_id` / `motor_model` / `kp` / `kd` / `dir` / `offset` with this precedence:
**URDF `<ros2_control><joint><param>` (if present) > the `motor_map:=` YAML >
built-in default** (sequential CAN IDs `0x01..`, model DM4310, kp/kd 0). Two ways
to supply the real per-joint values:

1. **Edit `urdf/m1.ros2_control.xacro`** to emit `<param name="can_id">…` etc. per
   joint (highest precedence), or
2. **(preferred) pass `motor_map:=<path>.yaml`** — the plugin now loads it via
   yaml-cpp in `parse_joints`/`load_motor_map`. It is the SAME schema
   `m1_hwconfig` writes (`{joint: {id, master_id, model, kp, kd, dir, offset,
   soft_limits}}`), so the config-page map is the source of truth for ids/models/
   gains with no xacro edit. (`soft_limits` is consumed by the controllers' limit
   enforcement, not the plugin.)

**Limp-arm safety guard:** if any COMMANDED joint ends up with `kp == 0` (e.g. a
missing/empty motor_map on a `use_mock:=false` launch), the plugin logs a
prominent `UNSAFE CONFIG` error and **refuses to open the CAN bus** (motors are
never enabled) so the arms can't silently go limp/mis-scaled. The component still
LOADS and runs no-bus/mock I/O — only LIVE driving is refused until kp is fixed.

**Mimic joints (resolved):** `parse_joints` now skips any joint with no `position`
command interface, so the two state-only mimic `*_finger_joint2` get **no motor
and no command interface** (their STATE interfaces are still exported, so they
appear in `/joint_states`). Result: **17 commanded motors, not 19**. One gripper
motor per arm drives the parallel fingers via the URDF `<mimic>`; the brain
commands `finger_joint1` and the mimic propagates.

**Order-independent device wiring:** the openarm CAN device collection is a
`std::map` keyed by `master_id`, so it iterates in ascending-master_id order — NOT
URDF joint order. The plugin matches each motor to ITS device **by master_id**
(`mit_control_one`/`get_motor` keyed on the resolved device index), and a startup
check refuses to drive if the configured master_ids don't match the live device
set. Operators can therefore reassign IDs via the config page without cross-wiring
commands/feedback.

## AgileX base integration (vendored: `agx_bringup`)

The AgileX Ranger-Air base driver is **vendored** into the workspace at
`ros2_ws/src/vendor/agx_bringup` (AgileX `ranger_ros2` @ **`air_delta`**, the
Ranger-Air-specific driver — the `jazzy`/`ranger_base`/`ugv_sdk` driver supports
only Ranger / Ranger Mini, *not* the Air; Apache-2.0; see `agx_bringup/VENDOR.md`
for origin + the two minimal local patches). It is **self-contained** (raw Linux
SocketCAN, **no `ugv_sdk`/libasio**) and builds clean on Jazzy with no API port.
`hardware.launch.py use_base:=true` launches it (`agx_bringup_node`) alongside the
two bridges.

- `m1_base_bridge`: `/m1/cmd_vel` → a single-intent body `geometry_msgs/Twist` on
  `/cmd_vel` (the driver's `/sub_cmd_vel`, remapped). The driver **auto-selects the
  motion mode itself** from the Twist (PARALLEL when `linear.y≠0`; SPINNING when
  turn radius `|vx/yaw|<0.5 m`; else DUAL_ACKERMANN) and emits the enable (`0x421`)
  + mode (`0x141`) + motion (`0x111`) CAN frames — so there is **no motion-mode
  topic/service** (the old `/m1/base/motion_mode` Int8 was dropped; nothing
  consumed it). The bridge still collapses to one mode's components via
  `select_motion_mode` so we never ask the firmware to blend strafe+yaw (it's
  mode-switched, never holonomic — memory `agilex-ranger-no-per-module-cmd`;
  `swerve.py`'s per-module output cannot drive this base). Dead-man'd (`BASE_HOLD`);
  the actual mode is readable on the driver's `/motion_mode_feedback`.
- `ranger_state_shim`: subscribes the driver's `/steering_angles`
  (`agx_bringup/SteeringAngles`, rad) + `/wheel_speeds` (`agx_bringup/WheelSpeeds`,
  m/s) → `/joint_states` for the 8 base joints, applying the swerve sign
  conventions and converting wheel m/s → rad/s (÷ `wheel_radius`).

**Build note:** building the vendored *message* package needs the **conda base env
off** — scrub `/home/jerry/miniconda3` from `PATH` (or `conda deactivate`) before
`colcon build`. The active conda Python 3.13 otherwise shadows ROS's `empy` and
breaks `rosidl` message generation (`em.TransientParseError: not enough data to
read`). Use system Python 3.12 (matches CLAUDE.md's `/usr/bin/python3` rule).

Deferred to hardware (cannot be derived from the driver source — see
`ranger_state_shim`'s "HARDWARE CHECKPOINTS" docstring):
1. **Module → corner mapping**: the driver labels modules `01..04` with no
   FL/FR/RR/RL legend. Default `corner_order:=[3,0,1,2]` (AgileX motor-ID order
   RF/RR/LR/LF → our fl/fr/rr/rl); confirm by jogging one module at a time and
   watching which `/joint_states` entry moves, then override the param.
2. **Wheel radius**: `/wheel_speeds` is linear m/s; set `wheel_radius` to the real
   Ranger-Air rolling radius (default `swerve.WHEEL_RADIUS` = 0.055 m).
3. **Steering sign / zero** vs our URDF (same class of check as the arm
   `dir`/`offset`).
4. **Base CAN bus**: a *separate* adapter from the Damiao arm bus (classic CAN,
   ~500 kbps — the base driver is **not** CAN-FD). Bring it up, then
   `base_can_interface:=canX` (default `can1`) selects it (the vendor patch wires
   the driver's `interface` param through).
5. **Per-module vs stock**: still **assumed STOCK** (Twist + internal auto-mode). If
   your specific unit instead exposes per-module control, `swerve.py` could drive it
   directly instead of this Twist path.

## Safety

- `enforce_command_limits: true` (in `m1_controllers.yaml`) clamps every streamed
  setpoint to the URDF position/velocity/effort limits in the framework
  (velocity/accel-bounded slew). Keep the brain's `IK_MAX_DQ` too (defense in
  depth).
- **E-stop must be a hardware mechanism.** Software does stop + controller reset on
  resume so the first post-resume command doesn't jump the arm (do NOT route
  e-stop through ROS topics).
- The teleop deadman / `BASE_HOLD` watchdog stays in the operator nodes (the
  forward controller has no command timeout).
- The plugin enables motors in `on_activate`, disables in `on_deactivate`/
  `on_error`, and treats transient CAN faults inside read/write (never escalates
  to ERROR, which would finalize the component).

## Deferred live-validation checkpoints (need real motors)

Validated offline today: byte-exact CAN codec (34 tests), the full mock
ros2_control loop (controllers active, brain reach flows through), the real plugin
loads + activates with no bus, the config page serves, the vendored AgileX base
driver (`agx_bringup`) builds clean on Jazzy and its `SteeringAngles`/`WheelSpeeds`
msgs import, the base bridges' pure mappers pass (`_bridge_test.py`), and **all
brain gated suites stay green**. On hardware, additionally verify:

1. Per-joint **sign/direction** (`dir`) and **offset** match the real motors
   (the `dir`/`offset` math is implemented but untested against real encoders).
2. The **live closed loop**: the controller's command-fingertip vs the measured
   `/joint_states` fingertip ≈ 0 mm (the discipline in memory
   `drake-solver-backend`). The offline suites use perfect feedback + zeroed
   fingers and can't catch a live mimic/sign bug.
3. **Gains** (`config/control_gains.yaml` kp/kd) — expect gravity-comp tuning,
   like OpenArm's open issues.
4. The base Twist path on the real chassis: `/m1/cmd_vel` → `m1_base_bridge` →
   `/cmd_vel` → `agx_bringup` drives, and the driver's auto-selected mode (read on
   `/motion_mode_feedback`) matches intent. Plus the `ranger_state_shim` checkpoints
   (module→corner `corner_order`, `wheel_radius`, steering sign/zero — see the
   "AgileX base integration" section).
```
