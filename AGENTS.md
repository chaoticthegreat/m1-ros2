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
  target(s) with adaptive (singularity-aware) damping, then leads the measured
  pose toward it by a bounded step. Reachable targets converge sub-mm;
  unreachable ones settle at the closest the joints allow. `solve_step`
  distinguishes two regimes so it is smooth for teleop AND globally optimal for
  cold targets (this split fixed the Quest "arm snaps to a random pose / moving
  one arm wrecks the other / sometimes slow" bugs):
  * **Tracking** (same arms active, target moved < `_IK_TRACK_JUMP`≈6 cm, i.e. a
    bridge nudging the goal each tick): warm-start from the cached goal, refine a
    few in-branch iterations, **no global restart**. The whole config (lift
    included) is gently regularized toward the previous goal, so the redundant
    DOFs can't drift between elbow/base branches. Result: no random snaps, the
    held arm stays put when the other moves, and ~1 ms/tick.
  * **Cold** (first solve / arm-set change / big jump): full multi-seed restart
    search, but (a) the primary solve regularizes arm joints to mid-range while
    the **lift stays free** so very high/low targets still solve, while the
    **restart probes run pure-task** on the re-searched DOFs (the mid-range pull
    drags an extreme near-boundary target short, so the probes drop it and reach
    sub-mm), (b) any arm whose target barely moved is *pinned* to its cached
    branch (and the shared lift is anchored toward its cached height) so a big
    move on one arm doesn't drag the held one, and (c) candidates are chosen by
    residual with a proximity tie-break. The cold search is **amortized across
    ticks**: each tick spends a bounded iteration budget (`_IK_COLD_BUDGET`) and
    carries the unfinished primary/probe state in the cache (`job`, resumable via
    `_pump_restart`), so the worst-case `solve_step` stays ~7 ms (was ~120 ms when
    the whole search ran in one tick) while the *total* search is more thorough.
  Every tick also applies a small capped per-arm Cartesian **hold correction** to
  the command (each arm's own joints, shared lift fixed) so the arms are
  decoupled through the lift. The solved goal is cached while the target holds.
  Benchmarks (`_solver_test.py`, the full suite — 15/15 gates): **100% of
  reachable single-arm targets <1 mm from cold** (was 22 mm worst), dual-arm
  <1.2 mm, **worst-case `solve_step` ~6.6 ms / 0% over the 60 Hz budget** (was
  ~122 ms), held-arm far-jump disturbance 25 mm → ~10 mm; `_teleop_stress.py`
  continuous tracking still 0.17 mm at ~1 ms.
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
 `_webxr_to_ros`. **In-headset RViz-like 3D viz:** the page renders the real
 robot meshes over passthrough with **three.js** (vendored locally under
 `m1_control/web_assets/vendor/`, served by the node — no CDN, no build step on
 device), posing each link every frame from per-link FK transforms streamed in
 the `/api/xr` response (`UrdfModel.link_transforms` + `mat_to_quat`). A target
 sphere per arm turns green→amber→red by fingertip distance (so impossible goals
 are obvious); B/Y recenters the model, which is anchored ~1.1 m in front on the
 floor. The glTF meshes (`web_assets/meshes/*.glb`, ~5 MB, decimated to ~3.5k
 faces/link, visual origin+scale baked in, mirrored winding flipped) are produced
 offline by `tools/convert_meshes.py` (needs a throwaway trimesh venv — see its
 docstring; re-run + rebuild only when meshes/URDF change). Each solid's real CAD
 **material colour is baked in as glTF vertex colours** (sRGB→linear), so the base
 reads as the actual robot — white body, black tyres/trim, red accents — instead
 of the old flat grey; the client renders COLOR_0 with a vertex-colour material
 (`ROBOT_MAT_VC`), falling back to grey for any colourless solid. The node serves
 `/manifest.json`, `/vendor/*`, `/meshes/*` as static files. Falls back to a
 wireframe (from `arms.points`) if the manifest is missing. three.js is the only
 third-party dep and it's vendored, so it stays a self-contained web app.
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
- **Full solver suite `_solver_test.py` (no ROS): 15/15 gates pass.** Covers
  reachability (single+dual), continuous tracking, hold-under-disturbance,
  latency distribution, and stress. Key results: **100% of reachable single-arm
  targets <1 mm *from a cold start*** (mean 0.21 mm, was 95% / 22 mm worst before
  the amortized pure-task restart), dual-arm 100% <5 mm (max 1.2 mm), **worst-case
  `solve_step` ~6.6 ms with 0% over the 60 Hz budget** (was ~122 ms), held-arm
  far-jump disturbance ~10 mm (was 25 mm). `_solver_bench.py` agrees (single/dual
  100%, max `solve_step` 6.5 ms). Steady-state `solve_step` ~0.6–1 ms; the cold
  iterate-and-restart search is amortized across ticks so no tick blows the budget.
- **Teleop tracking verified** (`_teleop_stress.py`, no ROS): streams a smoothly
  moving target like the Quest does. Continuous single-arm tracking holds <1 mm
  with goal jumps <0.03 rad (was: tens-to-hundreds of mm with ~5 rad branch
  flips — the reported "snaps to a random pose"); a held arm stays at ~0 mm while
  the other sweeps (was: >130 mm); per-tick time ~1 ms, 0% over the 60 Hz budget
  (was ~19 ms mean, 22% over). A separate scenario covers a large single-arm
  jump (cold re-solve) keeping the held arm's transient small.
- Lift recruitment verified standalone: commanding one arm high drives the lift
 up so the gripper reaches; commanding both arms together shares the lift as the
 least-squares compromise. Both arms with targets are solved together in one
 stacked system, so the lift serves both grippers optimally.
- End-to-end ROS test (`_ros_reach_check.py`, fake /joint_states, real DDS,
  timer-driven so the executor isn't GIL-starved): static target converges to
  <1 mm, a smoothly moving target tracks with <5 mm fingertip steps and no
  jumps, and the held arm stays within ~2 mm while the other sweeps. Stable
  across repeated runs. (NB: drive target/metric loops from ROS timers, not a
  main-thread `sleep` loop — the latter starves callbacks and makes the
  controller see the target *teleport*, which falsely looks like a solver bug.)
- `_e2e_check.py` (web-panel path): its web-node import was repaired
  (`_make_handler(node)`; `_resolve_web_dir` no longer exists), but it still has
  further stale web-panel API assumptions (`/api/state` no longer returns
  `njoints`); the reach path is covered by `_ros_reach_check.py` instead.
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
- Worst-case `solve_step` latency is now bounded ~7 ms: the cold iterate-and-
  restart search is **amortized across ticks** (`_IK_COLD_BUDGET` + a resumable
  `job` in the cache), so no single tick runs the whole multi-seed search.
  Continuous tracking / steady ticks remain ~1 ms. The trade-off is that a hard
  cold target now *converges over a handful of ticks* (~100–200 ms wall) instead
  of one slow tick — the command leads toward the best-so-far meanwhile, so the
  arm starts moving immediately and the goal only sharpens.
- A big *discontinuous* target jump on one arm (e.g. typing a far XYZ in the web
  panel, or `send_pose`) re-solves the coupled two-arm system; the held arm is
  pinned, the shared lift is anchored toward its cached height, and a per-arm
  Cartesian hold correction keeps the held gripper planted — its transient ride
  is down to ~10 mm (was ~25 mm) and recovers. Quest teleop never does this (the
  clutch moves the target continuously), so it stays in the smooth tracking path.
- Base `/m1/cmd_vel` is open-loop swerve; no odometry is published yet.
- The Isaac graph publishes `/joint_states` + `/clock` only; TF comes from
  robot_state_publisher on the ROS side. No camera/lidar/IMU bridged yet (the
  URDF has those frames if you want to add sensor graphs).
- Node/attribute names in `ros_sim.py` target the Isaac Sim 5.x
  `isaacsim.ros2.bridge`; if Isaac is upgraded, re-check them.
