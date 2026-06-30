# AgileX Ranger Air base — setup & bring-up

Step-by-step guide to get the **AgileX Ranger Air** swerve base driving under the
M1 stack. The base is driven by the **vendored AgileX driver**
(`ros2_ws/src/vendor/agx_bringup`, AgileX `ranger_ros2 @ air_delta`) plus two thin
bridge nodes; the control brain and every operator UI are unchanged.

For the architecture/reference, see `HARDWARE.md` → "AgileX base integration". This
file is the hands-on runbook. (For full-robot deployment onto a Jetson, see the
project's `DEPLOY_AGX_ORIN.md` if present.)

```
/m1/cmd_vel ──► m1_base_bridge ──/cmd_vel (Twist)──► agx_bringup_node ──CAN──► base
                                                          │
base ──CAN──► agx_bringup_node ──/steering_angles + /wheel_speeds──► m1_ranger_shim ──► /joint_states
```

The driver takes a **body Twist** and **auto-picks the motion mode** itself
(PARALLEL / SPINNING / DUAL_ACKERMANN) — there is no per-module or motion-mode
command. It is mode-switched: it will **not** strafe and rotate at the same time.

---

## 0. What you need

- An AgileX **Ranger Air** chassis, powered, with its CAN connector broken out.
- A **separate** USB-CAN adapter for the base (CANable/candleLight, Innomaker,
  PCAN, or the AgileX USB2CAN) — *separate from the Damiao arm bus*. The base is
  **classic CAN** (not CAN-FD); the arm bus is CAN-FD, so they must be two adapters.
- ROS 2 Jazzy + this workspace.

---

## 1. Wire & power

1. Connect the base CAN adapter to the chassis CAN port (CAN-H / CAN-L / GND).
2. Plug the adapter into the host. Confirm a `canX` netdev appears:
   ```bash
   ip -details link show | grep -A2 can
   dmesg | grep -i can | tail
   ```
   It will typically be `can0` (arms) and `can1` (base), but the order depends on
   plug order — note which is the base. The launch arg `base_can_interface:=` (default
   `can1`) selects it.
3. Power the chassis on. Make sure the **physical E-stop is released** and the
   chassis is in a mode that accepts CAN control (the driver sends the enable frame
   `0x421` automatically; a hardware RC override will still win).

---

## 2. Bring up the base CAN interface

The base bus is **classic CAN at 500 kbps** (AgileX Ranger default — confirm against
your chassis manual):

```bash
sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 up type can bitrate 500000
ip -s -d link show can1            # state should be UP/ERROR-ACTIVE
```

Sanity-check raw traffic (needs `can-utils`: `sudo apt install can-utils`): with the
chassis on you should see periodic feedback frames (e.g. `0x211`, `0x271`, `0x281`):
```bash
candump can1
```
If `candump` shows nothing, fix the wiring / bitrate / termination before going on.

> Persist across reboots with a systemd-networkd `.network`/`.link` or a udev rule;
> the base CAN must be up **before** launching the stack.

---

## 3. Build the workspace

**IMPORTANT — deactivate conda first.** The conda base env (Python 3.13) shadows
ROS's `empy` and breaks `rosidl` message generation for `agx_bringup`
(`em.TransientParseError`). Use system Python 3.12:

```bash
conda deactivate 2>/dev/null      # or: export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install     # builds the vendored agx_bringup + the bridges
source install/setup.bash
```

Verify the driver + messages are present:
```bash
ros2 pkg executables agx_bringup            # -> agx_bringup agx_bringup_node
ros2 interface show agx_bringup/msg/SteeringAngles
```

---

## 4. (Optional) test the base driver alone first

Before wiring it into the whole stack, confirm the driver talks to the chassis:

```bash
# terminal A: run just the AgileX driver on the base bus
ros2 run agx_bringup agx_bringup_node --ros-args -p interface:=can1

# terminal B: watch feedback (should update as you push the robot by hand)
ros2 topic echo /steering_angles
ros2 topic echo /wheel_speeds
ros2 topic echo /motion_mode_feedback

# terminal C: gently command a slow spin (CAUTION: robot will move!)
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist '{angular: {z: 0.3}}'
```

If feedback updates and the gentle command moves the base, the driver↔chassis link
is good. Stop terminal C (Ctrl-C) to halt. **Keep clear of the robot.**

---

## 5. Launch the full stack with the base

```bash
cd ros2_ws && source install/setup.bash      # (conda still deactivated)

# real arms + base:
ros2 launch m1_bringup hardware.launch.py \
    use_mock:=false use_base:=true \
    can_interface:=can0 can_fd:=true \
    base_can_interface:=can1 \
    motor_map:=$HOME/.config/m1/motor_map.yaml

# base only on the bench (mock arms): handy for base bring-up
ros2 launch m1_bringup hardware.launch.py use_mock:=true use_base:=true base_can_interface:=can1
```

This launches `agx_bringup_node` (base CAN), `m1_base_bridge`
(`/m1/cmd_vel`→`/cmd_vel`), and `m1_ranger_shim` (feedback→`/joint_states`) in
addition to the upper-body stack.

Then drive with any operator UI — all publish only `/m1/cmd_vel`:
```bash
ros2 run m1_control m1_web        # http://localhost:8080  (base drive pad)
ros2 run m1_control m1_teleop     # keyboard
ros2 run m1_control m1_quest      # Quest thumbstick base drive
```

---

## 6. Calibrate (REQUIRED — these can't be derived from the driver source)

The driver labels its modules `01..04` with **no FL/FR/RR/RL legend**, and the
steering sign / wheel radius depend on your unit. Calibrate once, on blocks (wheels
off the ground), with the base bus up.

### 6a. Module → corner mapping (`corner_order`)

With the base up and RViz/`m1_web` showing the base model, **jog one motion at a
time** and watch which `/joint_states` entry moves:

```bash
ros2 topic echo /joint_states --field name     # see fl_/fr_/rr_/rl_ steering+wheel
# gentle forward, then a slow spin, then a slow strafe — observe which corner reacts
ros2 topic pub -r 20 /m1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.1}}'
```

If the rendered corners are swapped/rotated relative to the real wheels, set the
`corner_order` parameter of `m1_ranger_shim` — a length-4 list mapping
**our** corner order `[fl, fr, rr, rl]` to the driver's `steering_0X / wheel_0X`
index (0-based). Default `[3, 0, 1, 2]` assumes the AgileX motor order RF, RR, LR,
LF. Edit it in `m1_bringup/launch/hardware.launch.py` (the `m1_ranger_shim` node's
`parameters`), e.g.:

```python
parameters=[{"corner_order": [0, 1, 2, 3], "wheel_radius": 0.055}],
```

### 6b. Wheel radius (`wheel_radius`)

`/wheel_speeds` is **linear m/s**; the shim converts to the wheel joint's **rad/s**
by dividing by the rolling radius. Set `wheel_radius` (m) to your Ranger Air's real
rolling radius (default `0.055`). Check: command a known `vx` and confirm the
`*_wheel_joint` velocity in `/joint_states` ≈ `vx / wheel_radius`.

### 6c. Steering sign / zero

If a steering corner renders mirrored or offset vs. the real wheel, flip that
joint's sign in `m1_control/m1_control/swerve.py` (`STEER_DIR` / `WHEEL_DIR`) — the
shim reuses those conventions. (Same class of check as the arm `dir`/`offset`.)

---

## 7. How it drives (mode behavior)

The base is **mode-switched** — it does one of these at a time, chosen by the driver
from the Twist you send (via `m1_base_bridge`):

| You command (`/m1/cmd_vel`)            | Mode            | What happens |
|----------------------------------------|-----------------|--------------|
| strafe (`vy` ≠ 0)                       | **PARALLEL**    | translate along (vx, vy); **yaw ignored** |
| pure rotate (`vx`≈0, `yaw`≠0)           | **SPINNING**    | spin in place; linear forced 0 |
| drive / drive+turn (`vx`, `yaw`, no vy) | **DUAL_ACKERMANN** | car-like; tight turns may auto-switch to SPINNING |

You **cannot** strafe and rotate in the same instant — `m1_base_bridge` drops the
weaker component so the firmware gets a clean single-mode command. The driver's
actual chosen mode is published on `/motion_mode_feedback`.

---

## 8. Bus ownership (don't run two owners)

A CAN bus has exactly one owner at a time:
- **Run mode:** the launch above owns the bus (driver + ros2_control).
- **Maintenance:** `m1_hwconfig` owns the *arm* bus for motor config — that's the
  Damiao bus, not the base. The base has no config tool; it's stock firmware.

Never run the AgileX driver and another base CAN program on the same `canX` at once.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `colcon build` fails with `em.TransientParseError` | conda base env active — `conda deactivate` (see §3), rebuild. |
| Driver logs `SocketCan初始化失败` / "no bus" | `canX` not up or wrong name — redo §2; check `base_can_interface:=`. |
| Base doesn't move | E-stop engaged; RC override on; bus down; or another program owns the bus. `candump canX` to confirm frames. |
| Base moves but RViz/Quest shows wrong corners | `corner_order` wrong — recalibrate §6a. |
| Wheel spin in `/joint_states` looks too fast/slow | `wheel_radius` wrong — §6b. |
| A steering corner mirrored/offset | sign convention — §6c. |
| Strafe + turn "ignores the turn" | expected — base is mode-switched (§7). |
| No `/joint_states` for base joints | `m1_ranger_shim` not getting `/steering_angles`+`/wheel_speeds`; confirm the driver publishes them (`ros2 topic hz /steering_angles`). |

---

## Deferred / known limits

- The base is assumed **stock** AgileX (Twist + auto-mode). If your unit exposes
  per-module control, `swerve.py` could drive it directly instead.
- Open-loop only: no `/odom` is published from the base yet (the Quest viz
  dead-reckons the *command*). Wiring `/wheel_speeds`+`/steering_angles` (or
  `/chassis_motion_feedback`) into a real odometry publisher is a good next step.
- Validate the live closed loop on hardware: drive a known `/m1/cmd_vel` and confirm
  `/motion_mode_feedback` + `/steering_angles`/`/wheel_speeds` match the intent.
