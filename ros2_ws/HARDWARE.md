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
   m1_joint_bridge ─► ros2_control                              m1_base_bridge ─/cmd_vel(Twist)─► AgileX driver
   forward_position_controller (+ JTC for planned moves)                   │
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

The `m1_hardware/M1SystemInterface` plugin reads per-joint `can_id` / `master_id`
/ `motor_model` / `kp` / `kd` / `dir` / `offset` from the URDF `<ros2_control>`
`<joint><param>` tags, falling back to **sequential CAN IDs `0x01..` and model
DM4310** when absent (fine for a no-bus load; NOT correct for live control). To
drive real motors you must supply the real per-joint values. Two ways:

1. **Edit `urdf/m1.ros2_control.xacro`** to emit `<param name="can_id">…` etc. per
   joint (from the `m1_hwconfig` map), or
2. **(preferred, TODO)** finish the `motor_map` hook in
   `m1_hardware/src/m1_system_interface.cpp` (`parse_joints`) so the plugin loads
   the `motor_map:=` YAML (the same schema `m1_hwconfig` writes:
   `{joint: {id, master_id, model, dir, offset, soft_limits}}`) as the source of
   truth for ids/models.

**Known wart:** the plugin currently creates a motor for all 19 ros2_control
joints, including the two state-only mimic `*_finger_joint2`. Those have **no
physical motor** (one gripper motor per arm drives the parallel fingers via the
URDF `<mimic>`). With no bus this is harmless; before live control, either omit
the two `finger_joint2` from the motor map or update `parse_joints` to skip joints
without a `position` command interface. The brain commands `finger_joint1` and the
mimic propagates.

## AgileX base integration (path, not yet vendored)

The ROS-side base bridges are implemented and unit-tested:
- `m1_base_bridge`: `/m1/cmd_vel` → body `geometry_msgs/Twist` on `/cmd_vel` +
  motion-mode (`/m1/base/motion_mode`, Int8). Stock Ranger firmware is
  **mode-switched, not free-holonomic** — it never blends strafe + rotate (see the
  memory note `agilex-ranger-no-per-module-cmd`). The bridge picks PARALLEL
  (strafe), SPINNING (yaw), or DUAL_ACKERMANN per command.
- `ranger_state_shim`: AgileX per-wheel feedback → `/joint_states` (8 base joints)
  so RViz/RSP animate the base.

To finish on hardware:
1. Clone + build the AgileX driver for **your** base on Jazzy — `ranger_ros2`
   (`air_delta` branch for the Ranger Air) + `ugv_sdk`. There is no official Jazzy
   branch; budget a recompile/port (plain rclcpp + tf2, low risk).
2. Bring up the base CAN (separate adapter, 500 kbps): the AgileX `setup_can2usb`
   scripts.
3. Point `m1_base_bridge`'s output `/cmd_vel` at the driver, and set
   `ranger_state_shim`'s `steer_topic`/`wheel_topic` params to the driver's
   per-wheel feedback topics (the `air_delta` branch publishes `/steering_angles`
   + `/wheel_speeds`). Map the motion-mode Int8 to the driver's `SetMotionMode`.
4. **Confirm whether your base is stock AgileX (Twist-only) or exposes per-module
   control.** If per-module, `swerve.py` can drive it directly instead of the
   Twist path.

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
loads + activates with no bus, the config page serves, and **all brain gated
suites stay green (113/113)**. On hardware, additionally verify:

1. Per-joint **sign/direction** (`dir`) and **offset** match the real motors
   (the `dir`/`offset` math is implemented but untested against real encoders).
2. The **live closed loop**: the controller's command-fingertip vs the measured
   `/joint_states` fingertip ≈ 0 mm (the discipline in memory
   `drake-solver-backend`). The offline suites use perfect feedback + zeroed
   fingers and can't catch a live mimic/sign bug.
3. **Gains** (`config/control_gains.yaml` kp/kd) — expect gravity-comp tuning,
   like OpenArm's open issues.
4. The base Twist path + motion-mode on the real chassis.
```
