# Vendored: `agx_bringup` (AgileX Ranger Air base driver)

This package is **vendored third-party source** — the AgileX Ranger Air / Delta
ROS 2 chassis driver. It is the real CAN driver for the M1 robot's AgileX base
(replacing the placeholder topic plumbing the bridges used to assume).

## Origin

- Repo: <https://github.com/agilexrobotics/ranger_ros2>
- Branch: **`air_delta`** (AgileX's own Ranger **Air** driver; the `jazzy` branch
  is `ranger_base`/`ugv_sdk` and supports only Ranger / Ranger Mini, **not** the
  Air — its kinematics model has no Air variant, so it is the wrong driver here).
- Upstream commit: `fb96124fa68c49bab03d8ddd2294c2afd45af925` (depth-1 clone, 2026-06-30).
- License: Apache-2.0 (see `package.xml`).
- Self-contained: raw Linux SocketCAN, **no `ugv_sdk` / libasio dependency** —
  only stock ROS 2 deps (`rclcpp`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `tf2*`,
  `rosidl_*`). Builds clean on **ROS 2 Jazzy** (C++14) with no API changes.

## Interface (what our nodes wire to)

- **Command:** subscribes `geometry_msgs/Twist` on `/sub_cmd_vel` (the stock
  `robot.launch.py` remaps `/sub_cmd_vel` → `/cmd_vel`). `linear.x`=vx (m/s),
  `linear.y`=vy (m/s), `angular.z`=yaw (rad/s). The driver **auto-selects** the
  motion mode internally from the Twist (`Handler111`: `linear.y!=0` → PARALLEL;
  else turn-radius `|vx/yaw|<0.5 m` → SPINNING; else DUAL_ACKERMANN) and emits the
  enable (`0x421`) + mode (`0x141`) + motion (`0x111`) CAN frames itself. There is
  **no external motion-mode topic/service** — mode is set purely by Twist shape.
  `m1_base_bridge` therefore publishes a single collapsed `Twist` and does **not**
  publish a separate motion-mode message.
- **Feedback:** publishes `agx_bringup/msg/SteeringAngles` on `/steering_angles`
  (`steering_01..04`, **rad**) and `agx_bringup/msg/WheelSpeeds` on `/wheel_speeds`
  (`wheel_01..04`, **m/s linear**). `m1_ranger_shim` consumes these. Also publishes
  `/motion_mode_feedback`, `/system_status`, `/chassis_motion_feedback`,
  `/front_wheel_odom`, `/back_wheel_odom` (informational; not used by the shim).

## Local modifications (kept minimal — grep `[M1 vendor patch]`)

1. `src/can/socket_can.cpp` — the `SocketCan` constructor now reads the node's
   `interface` parameter (default `can0`) before opening the socket. Upstream
   declared `interface` in `config/agx_bringup.yaml` but never read it (the device
   was hardcoded to `can0`); this makes the launch's `can_interface:=` argument
   actually select the base CAN bus.
2. `include/nlohmann/json.hpp` — **added** (nlohmann/json single-header v3.12.0,
   MIT). Upstream `#include <nlohmann/json.hpp>` requires the system
   `nlohmann-json3-dev` package, which is not installed here; vendoring the single
   header keeps the build self-contained and offline (no `apt`). It resolves via
   the package's existing `target_include_directories(... include)`.

## Re-vendoring

```bash
git clone --depth 1 -b air_delta https://github.com/agilexrobotics/ranger_ros2.git
cp -r ranger_ros2/agx_bringup ros2_ws/src/vendor/agx_bringup
# re-apply the two local modifications above, then rebuild.
```
