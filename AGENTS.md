# M1 ROS 2 — handoff notes for the next agent

Quick orientation for whoever picks this up next. Read this first, then
`ros2_ws/README.md` (full ROS interface) and `isaac/README.md` (Isaac details).

## What this repo is

An AgileX Ranger Air swerve base + prismatic lift + dual 7-DOF OpenArm arms
(27 DOF), simulated in Isaac Sim and controlled over ROS 2. Goal: give an arm a
Cartesian target pose and the arm joints + shared lift reach toward it; drive
the base with a velocity command. Same control code is meant to later run on the
real robot.

## Environment (this machine)

- Isaac Sim **5.1** native at `/home/jerry/isaac-sim`, bundled Python **3.11**.
- ROS 2 **Jazzy** at `/opt/ros/jazzy`, Python **3.12**. MoveIt, PyKDL,
  robot_state_publisher, xacro, rviz2 are installed.
- Platform: DGX Spark, aarch64, GB10 GPU.
- ⚠️ `python3` on PATH is **conda 3.13** — does NOT match ROS. Use
  `/usr/bin/python3` (3.12) for standalone ROS scripts. `ros2 run` launchers use
  the correct interpreter via their shebang.

## Architecture (two processes, talk over DDS)

```
Isaac Sim (python.sh, isaac/ros_sim.py)        ROS 2 control (Jazzy)
ROS 2 bridge OmniGraph  ── /joint_states,/clock ──▶  m1_control "brain"
= simulated robot driver ◀── /m1/joint_command ──    (IK + swerve) + RSP + RViz
```

The split exists because Isaac's Python (3.11) ≠ Jazzy's (3.12), so `rclpy`
cannot run inside Isaac. The Isaac side uses the OmniGraph ROS 2 bridge (ships
its own ROS libs). To deploy on hardware, replace `isaac/ros_sim.py` with the
real driver; everything in `ros2_ws/` is unchanged.

## Key files

- `isaac/ros_sim.py` — robot as a ROS 2 bridge node (clock + joint_states pub,
  joint_command sub → IsaacArticulationController). Sets drive gains (stiff arms,
  velocity wheels). Logs to `isaac/last_ros_sim_report.txt`.
- `ros2_ws/src/m1_control/m1_control/kinematics.py` — dependency-free URDF FK +
  geometric Jacobian + damped-least-squares (DLS) Cartesian reach. Jacobian
  verified against finite differences. The reach is a full iterated Gauss-Newton
  IK: each call iterates the model to the optimal joint configuration for the
  target(s) with adaptive (singularity-aware) damping and multi-seed restarts to
  avoid local minima, then leads the measured pose toward it by a bounded step.
  Reachable targets converge sub-mm; unreachable ones settle at the closest the
  joints allow. The solved goal is cached while the target holds, so the heavy
  search runs only on a target change (steady ticks cost one bounded step;
  benchmark `_solver_bench.py`: 100% of reachable single-arm targets <1 mm).
- `.../swerve.py` — swerve base kinematics (vx, vy, yaw → steer + wheel spin).
- `.../controller_node.py` — the only node you talk to. Subscribes to pose
 targets / cmd_vel / gripper + /joint_states; publishes unified
 `/m1/joint_command` at 60 Hz. Recruits the shared lift for reaching: each arm
 with a target is solved to its optimal joint configuration, and when both arms
 have targets they share one stacked solve so the single lift is the
 least-squares compromise that best serves both grippers. Also publishes
 `/m1/target_markers` (visualization_msgs/MarkerArray: target sphere + label,
 current fingertip, error line per arm) for RViz.
- `.../send_pose.py` — `ros2 run m1_control m1_send_pose --arm left --xyz x y z`.
- `.../teleop_node.py` — `ros2 run m1_control m1_teleop`: interactive keyboard
 console. Publishes only to the controller's `/m1/*` topics, so the SAME
 interface drives sim and the real robot (hardware-agnostic, unlike
 `isaac/teleop.py` which drives the Isaac articulation directly and is sim-only).
 Keeps a per-arm Cartesian target seeded from `/joint_states` FK so the arm
 doesn't jump on connect.
- `.../quest_node.py` — `ros2 run m1_control m1_quest`: Meta Quest WebXR teleop.
 Serves ONE self-contained WebXR page over **HTTPS** (self-signed cert auto-made
 with openssl into `~/.cache/m1_quest/`; WebXR needs a secure context over LAN
 IP). The Quest browser opens it, enters immersive-ar (passthrough) and POSTs
 both controllers' grip-space poses + buttons to `/api/xr`. The node maps each
 hand to the same-side arm with **clutched relative** motion (hold Grip to move,
 release to recenter), Trigger→gripper, thumbsticks→base, A/X→re-seed. Publishes
 only to `/m1/*`, so sim + real, exactly like `web_node`/`teleop_node`. WebXR
 axes (x right, y up, −z fwd) → ROS base_link (x fwd, y left, z up) in
 `_webxr_to_ros`. No extra deps (stdlib HTTPS + embedded HTML/JS).
- `.../web_node.py` — `ros2 run m1_control m1_web`: browser control panel on
 http://localhost:8080. Same `/m1/*`-only bridge (sim + real). Stdlib HTTP
 server + embedded HTML/JS (no extra deps); base drive pad, per-arm Cartesian
 targets + gripper, live fingertip/dist readout, dead-man'd base. A status dot
 reflects whether `/joint_states` is live. UI is themed after anthropic.com
 (warm cream, serif headings, clay accent). An untouched arm's panel target
 re-syncs to its live fingertip so nudges stay relative to where the arm
 actually is (the lift is a shared compromise when both arms reach).
- `ros2_ws/src/m1_bringup/launch/bringup.launch.py` — RSP + controller + RViz.
- `assets/{ranger_air,openarm}_description` — also ROS 2 packages (symlinked into
  `ros2_ws/src/`); hold URDF + meshes.

## How to run

```bash
# build
cd ros2_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install
source install/setup.bash

# Terminal 1: sim
source /opt/ros/jazzy/setup.bash
/home/jerry/isaac-sim/python.sh isaac/ros_sim.py        # --headless to skip GUI

# Terminal 2: brain + rviz
ros2 launch m1_bringup bringup.launch.py
ros2 run m1_control m1_send_pose --arm left --xyz 0.30 0.20 0.95

# Terminal 3 (optional): operator interfaces (sim OR real robot)
ros2 run m1_control m1_web        # browser panel at http://localhost:8080
ros2 run m1_control m1_teleop     # or a terminal keyboard console
```

Note: the controller stays idle until `/joint_states` arrives (it seeds its
command pose from feedback first). If "nothing moves", Terminal 1 (sim/robot)
is not running — the web panel's status dot makes this obvious.

## Status / what's verified

- `colcon build` clean (4 pkgs). All Python syntax-checked.
- IK converges to the optimal reachable configuration, recruits the lift, and
  shares the lift across both arms. Benchmarked standalone (`_solver_bench.py`,
  no ROS) against targets generated as the FK of real joint configs: **100% of
  reachable single-arm targets reach <1 mm** (mean ~0.3 mm) and **100% of
  jointly-feasible dual-arm targets reach <5 mm** (mean ~0.4 mm). Steady-state
  `solve_step` cost ~0.7 ms; the heavy iterate-and-restart search runs only on a
  target change (cached otherwise).
- Lift recruitment verified standalone: commanding one arm high drives the lift
 up so the gripper reaches; commanding both arms together shares the lift as the
 least-squares compromise. Both arms with targets are solved together in one
 stacked system, so the lift serves both grippers optimally.
- End-to-end ROS test (fake /joint_states, real DDS): both arms reach FK-derived
  targets to ~0 cm and recruit the shared lift; /m1/cmd_vel → correct wheel
  velocities + steer angles. (`_e2e_check.py` has a pre-existing unrelated
  web-node import error; the reach path itself is verified.)
- `isaac/ros_sim.py` itself was NOT run end-to-end by the agent (no GPU/display
  in the sandbox). The user confirmed it loads after the `set_target_prims` fix.

## Gotchas already hit (don't re-debug these)

- `og.Controller.set_target_prims` does NOT exist in this Isaac build. We set the
  `inputs:targetPrim` relationship via the USD API directly (see `_build_ros2_graph`).
- DDS needs to open loopback UDP sockets — in a restricted sandbox this fails with
  "Error creating socket: Permission denied"; run with real network access.
- ROS log dir defaults to `~/.ros` (not writable in sandbox); set `ROS_LOG_DIR`
  to a writable path when testing in a restricted shell.
- `/m1/joint_command` is intentionally **NaN-free**: wheels get velocity (their
  drive stiffness is 0 so position is ignored), everything else gets position +
  0 target velocity. Don't reintroduce NaNs — the Isaac controller is happier.

## Known limitations / good next steps

- Reaching is **position-only**: the gripper fingertip is driven onto the target
  point; the PoseStamped's **orientation is ignored**. Next: extend the task to
  6-DOF (stack orientation error + use the angular Jacobian rows already computed
  in `fk`). The solver structure (`_stack` / `_dls`) already generalises to it.
- No collision avoidance / planning — it's a reactive Jacobian controller solved
  to convergence each tick. MoveIt is installed if you want planned motion (would
  need SRDF + kinematics config + a controller bridge).
- Shared-lift tradeoff: the lift is one prismatic joint feeding both arms, so
 when both arms have targets the stacked solve picks the single lift height that
 minimises the combined (equal-weight) fingertip error. Two arms with targets at
 very different heights therefore share the lift as a compromise; clear one
 arm's target if you want the lift to commit entirely to the other.
- Worst-case `solve_step` latency: a target *change* triggers the iterate-and-
  restart search (bounded, ~tens of ms in the rare hard/cold case); steady ticks
  reuse the cached solution. If you ever need a hard real-time bound, amortise
  the restart seeds across ticks.
- Base `/m1/cmd_vel` is open-loop swerve; no odometry is published yet.
- The Isaac graph publishes `/joint_states` + `/clock` only; TF comes from
  robot_state_publisher on the ROS side. No camera/lidar/IMU bridged yet (the
  URDF has those frames if you want to add sensor graphs).
- Node/attribute names in `ros_sim.py` target the Isaac Sim 5.x
  `isaacsim.ros2.bridge`; if Isaac is upgraded, re-check them.
