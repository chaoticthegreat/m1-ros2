# M1 Real-Hardware Deployment — Design Spec

**Date:** 2026-06-23
**Branch:** `real-hardware-deployment`
**Status:** Approved design → implementation

## 1. Goal

Take the M1 stack (AgileX Ranger swerve base + shared prismatic lift + dual 7-DOF
OpenArm arms, 27 DOF), today driven only by Isaac Sim over DDS, and make it drive
**real hardware**: Damiao (DM-series) CAN motors for the arms + lift, and the
AgileX base. Plus a **hardware configuration/test web page** (assign motors, edit
limits, jog/test motors, calibrate zero, live telemetry).

Same control code must keep driving sim; hardware is an alternate backend, not a
fork.

## 2. Architecture decision (approved)

**Hybrid**, motor layer built **phased (Python bring-up → C++ ros2_control)**.

- **Reuse OpenArm's hardware layer** (`openarm_can` + `openarm_hardware`, Apache-2.0)
  as the Damiao MIT-mode SocketCAN driver — the tedious hardware code we'd otherwise
  write from scratch, and the arms are customized-OpenArm so it fits closely.
- **Keep our Drake position-only IK** (`kinematics.py`) as the reactive Cartesian
  brain. It natively does the four things MoveIt structurally cannot:
  position-only reach, the **shared lift as a least-squares compromise serving both
  grippers**, 60 Hz reactive teleop, and coupled dual-arm.
- **MoveIt is optional/later**, used only for what it genuinely wins: planned,
  collision-aware point-to-point motion. It is NOT on the reactive/teleop path.

### Why not pure-MoveIt (the comparison verdict)

MoveIt's bundled IK (OpenArm uses `KDLKinematicsPlugin`) requires one serial chain →
one tip per planning group. The lift is one physical joint feeding **both** arms;
MoveIt has no concept of a single DOF resolved as the least-squares compromise across
two single-tip IK groups (`{lift+left}` and `{lift+right}` would each command a
different lift height and fight over the actuator; `{lift+left+right}` is not a serial
chain and KDL rejects it). OpenArm's MoveIt config is also bimanual-but-independent
(no combined group), has **no MoveIt Servo configured**, defaults to full 6-DOF pose
IK (position-only fights it), and is self-described "🚧 under active development."
Adopting it would forfeit requirements 1–4 to gain only requirement 5 (planned
collision-aware motion), which we can add later through the same controllers.

Full research + file-level findings: the six research-agent reports and the OpenArm
deep-dive in this session's transcript.

## 3. The invariant contract (every backend must preserve)

This is the contract Isaac honors today and the real stack must reproduce bit-for-bit,
so the Drake brain, swerve, RViz markers, and ALL gated suites stay unchanged.

**Topics (the only interface the brain + operators use):**

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/joint_states` | sensor_msgs/JointState | brain ← robot | measured pos/vel (+effort) |
| `/m1/joint_command` | sensor_msgs/JointState | brain → robot | unified command |
| `/m1/{left,right}_arm/target_pose` | geometry_msgs/PoseStamped | brain ← op | position-only reach |
| `/m1/cmd_vel` | geometry_msgs/Twist | brain ← op | base (vx,vy,yaw) |
| `/m1/{left,right}_arm/gripper` | std_msgs/Float64 | brain ← op | 0=closed..1=open |

**The 27-DOF canonical name order** (from `controller_node.ALL_JOINTS`):
`fl/fr/rr/rl_steering_joint`, `lift_joint`, `openarm_left_joint1..7`,
`openarm_right_joint1..7`, `openarm_left_finger_joint1/2`,
`openarm_right_finger_joint1/2`, `fl/fr/rr/rl_wheel_joint`.

**Conventions:** wheels = velocity (rad/s), everything else = position (rad / m for
lift); fingers mimic-coupled (right side negated); message is NaN-free.

## 4. The new seam (replaces `isaac/ros_sim.py` on real hardware)

`controller_node` is **unchanged** — `command_topic` is already a parameter and it
already merges partial `/joint_states`. The new ROS-side pieces are bridges; the
brain and operator nodes (`web_node`, `quest_node`, `teleop_node`, `send_pose`) and
every solver test are untouched.

```
operators ──/m1/*──► controller_node (Drake IK + swerve, 60 Hz) ──► /m1/joint_command
                                                                          │
                          ┌───────────────────────────────────────────────┤
                          ▼ (upper body: lift+arms+fingers)                ▼ (base: steer/wheel entries ignored)
              m1_joint_bridge ──► ros2_control                  m1_base_bridge ──/cmd_vel(Twist)──► AgileX ranger driver
              forward_position_controller (+JTC available)                 │
                          │ loaned-memory                                  │ CAN feedback
                          ▼                                                ▼
              m1_hardware SystemInterface (Damiao MIT, SocketCAN) + lift   AgileX firmware (swerve IK in firmware)
                          │                                                │
              joint_state_broadcaster ──┐                  ranger_state_shim ──┐
                                        └────────► /joint_states ◄────────────┘  (union by name; both publish)
```

- **`m1_joint_bridge`** (Python): subscribes `/m1/joint_command` (JointState), writes
  the **lift + 14 arm + 4 finger** positions to the `forward_position_controller`
  command topic (Float64MultiArray, name→index map). Steer/wheel entries are ignored
  on real HW (the base is not per-joint-commandable; see §3 base note). One thin,
  testable mapping node. (In sim, this bridge is not launched — Isaac consumes
  `/m1/joint_command` directly, exactly as today.)
- **ros2_control stack**: `m1_hardware` plugin + `forward_position_controller`
  (the right controller for a streamed 60 Hz setpoint per UR/MoveIt-Servo precedent;
  **not** JTC, which splices time-parameterized waypoints) + `joint_state_broadcaster`
  + a gripper path. JTC is also loaded for planned moves (hot-swapped via
  `controller_manager`). `enforce_command_limits` ON.
- **`m1_base_bridge`** (Python): subscribes `/m1/cmd_vel`, maps to the AgileX body
  `Twist` + motion-mode the stock Ranger firmware accepts (mode-switched, not free
  holonomic — see memory `agilex-ranger-no-per-module-cmd`). Keeps the deadman.
- **`ranger_state_shim`** (Python): AgileX per-wheel feedback → JointState for the 8
  base joints so RSP animates the URDF (the real driver publishes no `/joint_states`).

### Bus-ownership model (safety)

Motor **configuration** (assign CAN/master ID, set-zero, mode, flash limits) and live
**ros2_control** are mutually exclusive on a CAN bus (OpenArm separates its config GUI
from the control loop the same way):

- **Maintenance mode** — `m1_can_tools` Python driver owns the bus; the config/test
  web page can enable/jog/zero/assign/configure. ros2_control is down.
- **Run mode** — the ros2_control stack owns the bus; reactive control + optional
  MoveIt. The config page is read-only (live telemetry from `/joint_states` +
  diagnostics).

## 5. Components / package layout

```
ros2_ws/src/
  m1_control/        (exists) UNCHANGED brain + operators + gated tests.
                     ADD: m1_joint_bridge, m1_base_bridge, ranger_state_shim nodes
                     (new entry points) + the m1_hwconfig web node.
  m1_hardware/       (NEW, C++/ament_cmake) Damiao SystemInterface plugin
                     (forked from openarm_hardware), vendored openarm_can, a CAN
                     transport abstraction (SocketCAN default / serial fallback),
                     the lift motor, a motor-ID→joint map loaded from YAML.
  m1_can_tools/      (NEW, Python/ament_python) Phase-0 bring-up driver:
                     DM CAN frame codec (MIT / pos-vel / vel, enable 0xFC /
                     disable 0xFD / set-zero 0xFE / clear-err 0xFB), per-model limit
                     tables, transport abstraction (shared with m1_hardware's intent),
                     a maintenance-mode bus owner. Pure-python + python-can; importable
                     and unit-testable with NO hardware.
  m1_bringup/        (exists) ADD hardware bringup launch: ros2_control_node +
                     controllers + RSP + bridges, with mock_components↔real plugin
                     switchable by one launch arg; AgileX driver optional include.
```

## 6. Phased delivery plan

Phases are ordered by dependency; ✦ marks work parallelizable within/across a phase.
Each phase ends green (its tests pass) and is committed.

### Phase 0 — Motor bring-up driver + config/test web page (no ros2_control)
- ✦ `m1_can_tools`: DM CAN frame **codec** (encode MIT/pos-vel/vel commands + special
  frames; decode feedback incl. temp + error nibble) with the verified per-model limit
  tables. Pure functions, exhaustively unit-tested vs the documented protocol.
- ✦ Transport abstraction: `SocketCanTransport` (python-can `can0`) + `SerialTransport`
  (vendor `/dev/ttyACM*` 0xAA/0x55 framing) behind one interface; a `FakeTransport`
  for tests.
- ✦ Maintenance-mode bus owner: enumerate/scan motors, enable/disable, jog (clamped),
  set-zero, read telemetry; a motor-ID→joint map persisted to YAML.
- ✦ `m1_hwconfig` web node (see §7) on top of it.
- **Gate:** codec unit tests; transport round-trip via FakeTransport; web data-path
  test (headless, like `_quest_position_test.py`). No hardware needed.

### Phase 1 — ros2_control skeleton on mock_components + bridges + control re-wire
- ✦ URDF `<ros2_control>` xacro for lift+arms+fingers (mock_components/GenericSystem
  default; real plugin by arg); `controllers.yaml` (forward_position_controller,
  joint_state_broadcaster, JTC, gripper); `joint_limits.yaml` from the URDF limit table.
- ✦ `m1_joint_bridge` + `m1_base_bridge` + `ranger_state_shim`.
- ✦ Hardware bringup launch; verify the full reach loop in mock + against Isaac.
- **Gate:** mock round-trip (command in → /joint_states out, contract preserved);
  bridge mapping unit tests; existing solver suites still green; an end-to-end
  mock reach check (like `_ros_reach_check.py`).

### Phase 2 — Real C++ SystemInterface (`m1_hardware`)
- Vendor `openarm_can`; fork `openarm_hardware`→`m1_hardware`; **port Humble→Jazzy**
  (interface-export migration); generalize the hardcoded 7-DOF / dual-bus assumptions;
  **add the lift motor**; wire the motor-ID map + CAN transport abstraction.
- Swap mock→real in the same URDF; turn on `enforce_command_limits`.
- ✦ Base: integrate AgileX `ranger_ros2` (`air_delta` branch) or `ugv_sdk` Twist path,
  port to Jazzy; finish `ranger_state_shim`.
- **Gate:** builds clean on Jazzy (colcon); CAN codec/HW interface unit tests; on
  hardware (when available) the **live closed-loop** check — command-fingertip vs
  measured ~0 mm, the discipline from memory `drake-solver-backend`.

### Phase 3 — MoveIt (optional, additive)
- A MoveIt config (adapted from OpenArm's bimanual, + the lift as a planning-only
  joint-space group) for planned collision-aware moves, executed via JTC, hot-swapped
  with the forward controller. Reactive teleop + shared lift stay on Drake.
- **Gate:** a planned move executes in mock/sim; reactive path unaffected.

## 7. Hardware configuration/test web page (`m1_hwconfig`)

New node, stdlib `http.server` + embedded HTML/JS (no new deps), themed like
`web_node` (warm cream / clay accent), bound to the maintenance-mode driver. Sections:

1. **Motor inventory & assignment** — scan the bus; list responding motors (ID,
   master ID, model, firmware/temp); set/verify CAN ID + master ID; map each motor →
   logical joint (e.g. `openarm_left_joint3`); persist the map to YAML the hardware
   layer reads. Flag duplicate/missing IDs.
2. **Per-joint limits editor** — position/velocity/effort + soft limits per joint,
   seeded from the URDF; writes `joint_limits.yaml` (consumed by ros2_control) and,
   where applicable, motor-flash limits. Validates against per-model `[P,V,T]MAX`.
3. **Motor test / jog** — per-motor enable/disable; MIT/pos-vel jog with a
   hardware-limit-clamped slider + live setpoint; **set-zero** calibration; a dead-man
   (motion only while held) + per-action confirm.
4. **Live telemetry** — position/velocity/torque, MOS + rotor temperature, decoded
   error nibble, CAN bus health; color-coded fault states.
5. **Mode guard** — write actions only in maintenance mode (ros2_control down);
   otherwise read-only telemetry. A clear banner shows the current bus owner.

## 8. Safety layer

- Joint limits in URDF + `joint_limits.yaml`; **`enforce_command_limits` ON** (Jazzy
  clamps streamed setpoints to velocity/accel-bounded slews — the framework version
  of our `IK_MAX_DQ`). Keep `IK_MAX_DQ` too (defense in depth).
- E-stop as a **hardware** mechanism; software does stop + controller reset on resume
  so the first post-resume command doesn't jump the arm (per ros2_control guidance —
  do not route e-stop through topics).
- Keep the existing teleop deadman / `BASE_HOLD` watchdog; `forward_command_controller`
  has no timeout, so staleness guarding stays upstream in the brain/bridges.
- Motor enable sequence in `on_activate`; disable in `on_deactivate`/`on_error`;
  decode temp/error each cycle and back off, treating transient CAN faults inside
  read/write (don't escalate to ERROR, which finalizes the component).

## 9. Validation strategy

- **All existing gated suites stay green, unchanged** (contract preserved):
  `_solver_test*.py`, `_accuracy_bench.py`, `_swerve_test.py`, `collision`, `trajectory`.
- **New suites:** DM CAN codec encode/decode vs documented protocol + per-model limit
  tables; transport round-trip (FakeTransport); bridge name→index mapping;
  mock_components contract round-trip; limit-enforcement; `m1_hwconfig` data path
  (headless). All runnable with NO hardware (the project's offline-gate philosophy).
- **Live-loop discipline** (memory `drake-solver-backend`): when hardware is present,
  validate the closed loop (command-fingertip ≈ measured), not just offline gates.
  Interpreter rule: standalone scripts use `/usr/bin/python3` (Jazzy 3.12); the C++
  builds via colcon.

## 10. Risks / things to verify on the real robot

- **Base type** — stock AgileX (Twist-only, mode-switched) vs a custom swerve base
  exposing per-module control. Design defaults to the AgileX Twist path; if per-module
  is available, `swerve.py` can drive it directly. (memory `agilex-ranger-no-per-module-cmd`)
- **Motor part numbers + CAN/master IDs** — customized OpenArm; confirm models
  (affects the `[P,V,T]MAX` scaling) and the ID→joint map before live control.
- **CAN adapter** — SocketCAN (recommended, CAN-FD) vs vendor serial dongle; the
  transport abstraction makes this swappable.
- **Jazzy port** — OpenArm's HW interface targets Humble; budget the
  interface-export migration + a Jazzy colcon build of `openarm_can`.

## 11. Out of scope (YAGNI)

- Full autonomous navigation / SLAM / perception.
- `/odom` + TF publisher wired into `controller_node` (separate next step already noted
  in AGENTS.md).
- Re-introducing orientation (6-DOF) control — the reach stays position-only.
- Coordinated dual-arm planning in MoveIt (its IK can't; not needed for teleop).
