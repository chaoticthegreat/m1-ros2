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
  verified against finite differences. **The reach is POSITION-ONLY**: a target
  is a 3D point (or a dict carrying `"pos"`; any `"quat"`/`"R"` is ignored) and
  the gripper fingertip is driven to it — orientation is not part of the task.
  (FK/quaternion utilities `pose_jacobian`/`gripper_pose`/`_so3_log`/`mat_to_quat`
  remain for the viz + tests to *report* gripper rotation, but the solve never
  constrains it.) The reach is a full iterated Gauss-Newton IK: each call iterates
  the model to the optimal joint configuration for the target(s) with adaptive
  (singularity-aware) damping, then leads the measured pose toward it by a bounded
  step. Reachable targets converge sub-mm;
  unreachable ones settle at the closest the joints allow. `solve_step`
  distinguishes three regimes so it is smooth for teleop AND globally optimal for
  cold targets (this split fixed the Quest "arm snaps to a random pose / moving
  one arm wrecks the other / sometimes slow" bugs):
  * **Tracking** (same arms active, target moved < `_IK_TRACK_JUMP`≈6 cm, i.e. a
    bridge nudging the goal each tick): warm-start from the cached goal, refine a
    few in-branch iterations, **no global restart**. The whole config (lift
    included) is gently regularized toward the previous goal, so the redundant
    DOFs can't drift between elbow/base branches. The in-branch refine uses a
    **backtracking line search** (`_solve_from(line_search=True)`): every step is
    shrunk until the task cost strictly decreases, so it can't overshoot into a
    worse configuration — that is what lets it **ride a wrist singularity through**
    instead of "getting stuck and not tracking" (the operator-reported failure).
    Result: no random snaps, the held arm stays put when the other moves, ~1 ms/tick.
  * **Re-acquire** (tracking refined but an arm's *solved* residual stays large for
    `_IK_REACQUIRE_TICKS` — a needed branch/elbow flip or recovering from a
    boundary saturation, which a small per-tick move never trips the cold path for):
    launch the same amortized multi-seed restart the cold path uses, but free *only*
    the stuck arm(s) (the healthy arm stays pinned), so it re-acquires the target
    instead of staying stuck forever. Gated on a *sustained* position residual
    (`_IK_REACQUIRE_POS` for `_IK_REACQUIRE_TICKS` ticks), so a brief singularity
    crossing recovers in-branch and never triggers a spurious branch jump.
  * **Cold** (first solve / arm-set change / big jump): full multi-seed restart
    search, but (a) the primary solve regularizes arm joints to mid-range while
    the **lift stays free** so very high/low targets still solve, while the
    **restart probes run pure-task** on the re-searched DOFs (the mid-range pull
    drags an extreme near-boundary target short, so the probes drop it and reach
    sub-mm), (b) any arm whose target barely moved is *pinned* to its cached
    branch (and the shared lift is anchored toward its cached height) so a big
    move on one arm doesn't drag the held one, and (c) candidates are chosen by
    residual with a proximity tie-break. The cold/probe solves use the **fixed-step**
    `_solve_from(line_search=False)` (NOT the line search): a seed must be free to
    step *over* a saddle — e.g. the shared-lift compromise of a partly-unreachable
    dual target, where one arm swings the lift the "wrong" way briefly so the held
    arm can recompense — which a strict line search would refuse, leaving the held
    arm short. Diverse seeds + posture pinning supply robustness here instead of
    monotonicity. The cold search is **amortized across ticks**: each tick spends a
    bounded iteration budget (`_IK_COLD_BUDGET`) and carries the unfinished
    primary/probe state in the cache (`job`, resumable via `_pump_restart`), so the
    worst-case `solve_step` stays ~7 ms (was ~120 ms when the whole search ran in
    one tick) while the *total* search is more thorough.
  Every tick also applies a small capped per-arm Cartesian **hold correction** to
  the command (each arm's own joints, shared lift fixed) so the arms are
  decoupled through the lift. The solved goal is cached while the target holds.
  Benchmarks (`_solver_test.py`, the full suite — 15/15 gates): **100% of
  reachable single-arm targets <1 mm from cold** (was 22 mm worst), dual-arm
  <1.2 mm, **worst-case `solve_step` ~6.7 ms / 0% over the 60 Hz budget** (was
  ~122 ms), held-arm far-jump disturbance 25 mm → ~10 mm; `_teleop_stress.py`
  continuous tracking still 0.17 mm at ~1 ms. **`_solver_test_positions.py`** (new,
  19/19) simulates MANY positions: 300 single-arm + 200 dual reachable points, a
  7³ workspace grid sweep, and a guard that supplying an orientation has **zero**
  effect on the joint solution (bit-identical to the bare point — proves the
  rotation component is gone). **`_solver_test_tracking.py`** (20/20) is the hard
  position-tracking suite: large continuous Cartesian sweeps that cross internal
  singularities (typical p95 sub-mm; an aggressive sweep rides a singularity
  through with a bounded, recovering transient — never stuck), boundary excursion
  + re-acquire (stuck-forever → settles home sub-mm), cold-hard and dual-arm. This
  is the suite that reproduces and guards the "gets stuck and can't keep tracking"
  report. (Removing orientation dropped the incidental redundancy regularization
  it provided, so an aggressive sweep can briefly lag at an internal singularity;
  it recovers — mean/p95 stay sub-mm.)
- `.../swerve.py` — swerve base kinematics. `module_states` is the pure inverse
  map (vx, vy, yaw → per-module heading+speed); `SwerveSolver.solve` adds heading
  low-pass smoothing, the ≤90° flip-and-reverse optimisation, and **wheel-speed
  desaturation** (a command that would over-speed a module is scaled down
  uniformly, preserving the travel *direction*), then the per-joint sign fixups.
  `forward_kinematics` recovers (vx, vy, yaw) from the module states (4-module
  least-squares; exact inverse of `module_states`), and `SwerveOdometry`
  dead-reckons an (x, y, θ) base pose with the **exact SE(2) arc integration** of
  the body twist (turn-while-driving traces the true arc; pure spin keeps x,y
  fixed). Verified by `_swerve_test.py` (16/16). Used by `controller_node` (the
  solver) and `quest_node` (the odometry, to drive the headset model).
- `.../controller_node.py` — the only node you talk to. Subscribes to pose
 targets / cmd_vel / gripper + /joint_states; publishes unified
 `/m1/joint_command` at 60 Hz. Recruits the shared lift for reaching: each arm
 with a target is solved to its optimal joint configuration, and when both arms
 have targets they share one stacked solve so the single lift is the
 least-squares compromise that best serves both grippers. Also publishes
 `/m1/target_markers` (visualization_msgs/MarkerArray: target sphere + label,
 current fingertip, and error line, per arm) for RViz. The reach is position-only,
 so the target_pose orientation is ignored (no orientation triad is drawn).
- `.../send_pose.py` — `ros2 run m1_control m1_send_pose --arm left --xyz x y z`.
- `.../teleop_node.py` — `ros2 run m1_control m1_teleop`: interactive keyboard
 console. Publishes only to the controller's `/m1/*` topics, so the SAME
 interface drives sim and the real robot (hardware-agnostic, unlike
 `isaac/teleop.py` which drives the Isaac articulation directly and is sim-only).
 Keeps a per-arm Cartesian target seeded from `/joint_states` FK so the arm
 doesn't jump on connect.
- `.../quest_node.py` — `ros2 run m1_control m1_quest`: Meta Quest WebXR teleop.
 NOTE: the quest node still *computes and streams* a target orientation (for its
 own in-headset triad viz), but the controller's reach is now **position-only**, so
 that orientation no longer moves the gripper — only the target *point* does. The
 rotation-lock / absolute-orientation machinery below is therefore inert on the
 arm (kept so the node + `_quest_orientation_test.py` are unchanged); if you want a
 purely-positional Quest UX, strip the orientation streaming/lock/triad here too.
 Serves ONE self-contained WebXR page over **HTTPS** (self-signed cert auto-made
 with openssl into `~/.cache/m1_quest/`; WebXR needs a secure context over LAN
 IP). The Quest browser opens it, enters immersive-ar (passthrough) and POSTs
 both controllers' grip-space poses (position **and orientation**) + buttons to
 `/api/xr`. The node maps each hand to the same-side arm with **clutched relative
 translation** but **ABSOLUTE orientation** (hold Grip to translate; the gripper's
 rotation mirrors the controller's *actual* orientation —
 `R_target = C·hand_R·Cᵀ·ori_align` — so controller-up-90° → gripper-up-90°, it
 does NOT add 90° each grab like the old relative-twist did; `ori_align` is zeroed
 to the live pose on first grab / A-X so nothing snaps, see `_calibrate_ori`).
 **Thumbstick click toggles a per-arm rotation LOCK** (`ori_locked`) that freezes
 the gripper orientation so the wrist can be re-oriented / the hand translated
 without rotating the gripper; unlocking re-zeros `ori_align` so it resumes without
 a jump. Trigger→gripper, A/X→re-seed (target + rotation).
 Validated headless by `_quest_orientation_test.py` (8/8: absolute + no
 accumulation + lock, driving the real `on_xr_frame`).
 **Base drive (thumbstick push):** LEFT stick fwd/back→drive fwd/back (vx),
 left/right→strafe (vy); RIGHT stick left/right→turn (yaw); smooth rescaled
 deadzone; cmd_vel is set every frame so centring the stick stops at once
 (BASE_HOLD now only guards a lost connection). The headset **robot model
 actually drives through the room**: `SwerveOdometry` dead-reckons the commanded
 cmd_vel into a base pose streamed as `viz["base"]`, and the page hangs all link
 meshes/markers (and the orientation triads) off a `robotBase` group (inside the
 room-anchored `robotRoot`) carrying that pose, so the body translates/rotates
 while the wheels steer/spin from `/joint_states`. B/Y recenters and zeroes the
 odom (the `place` flag is sticky so a recenter is never dropped on an in-flight
 POST). Publishes
 only to `/m1/*`, so sim + real, exactly like `web_node`/`teleop_node`. WebXR
 axes (x right, y up, −z fwd) → ROS base_link (x fwd, y left, z up) in
 `_webxr_to_ros`. **In-headset RViz-like 3D viz:** the page renders the real
 robot meshes over passthrough with **three.js** (vendored locally under
 `m1_control/web_assets/vendor/`, served by the node — no CDN, no build step on
 device), posing each link every frame from per-link FK transforms streamed in
 the `/api/xr` response (`UrdfModel.link_transforms` + `mat_to_quat`). A target
 sphere per arm turns green→amber→red by fingertip distance (so impossible goals
 are obvious), with an **RGB orientation triad** at the target (commanded gripper
 rotation) plus a smaller faint triad on the live gripper so alignment is visible;
 B/Y recenters the model, which is anchored ~1.1 m in front on the floor. Mesh
 loading is hardened (parallel fetch with retry, explicit per-link failure
 reporting, per-arm wireframe fallback, frustum-culling off) so a dropped mesh
 load can't silently leave an arm link missing. The glTF meshes (`web_assets/meshes/*.glb`, ~5 MB, decimated to ~3.5k
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

- `colcon build` clean (4 pkgs; `m1_control` rebuilt after the position-only +
  swerve changes). All Python syntax-checked.
- **The reach is POSITION-ONLY.** The solver, `controller_node`, and its markers
  were stripped of the orientation (6-DOF) path: a target is a 3D point and the
  gripper's rotation is not constrained. `_solver_test_positions.py` guards that a
  supplied orientation is bit-for-bit ignored (identical joint solution to the bare
  point). The position-solve code itself is unchanged, so its accuracy/latency are
  the same as before.
- **Swerve suite `_swerve_test.py` (no ROS): 16/16 gates pass.** Covers IK↔FK
  round-trip (≈1e-15), solver geometry (drive/strafe/turn-in-place point the
  modules right; reverse uses the ≤90° flip), desaturation (caps + preserves
  direction), settled-command→body-velocity round-trip, and odometry (straight,
  strafe, turn-in-place, arc vs analytic, full-circle closure). The arm-solver
  suite (`_solver_test.py` 15/15) and `_teleop_stress.py` are unaffected by the
  swerve refactor.
- **Full solver suite `_solver_test.py` (no ROS): 15/15 gates pass.** Covers
  reachability (single+dual), continuous tracking, hold-under-disturbance,
  latency distribution, and stress. Key results: **100% of reachable single-arm
  targets <1 mm *from a cold start*** (mean 0.21 mm), dual-arm 100% <5 mm (max
  1.2 mm), **worst-case `solve_step` ~6.7 ms with 0% over the 60 Hz budget**,
  held-arm far-jump disturbance ~10 mm. `_solver_bench.py` agrees (single/dual
  100%). Steady-state `solve_step` ~0.6–1 ms; the cold iterate-and-restart search
  is amortized across ticks so no tick blows the budget.
- **Many-positions suite `_solver_test_positions.py` (no ROS): 19/19 gates.**
  Simulates a large position set: 300 single-arm + 200 dual reachable points and a
  7³ workspace grid. Large-sample distribution: single ~99% <2 mm / 100% <5 mm
  (a few near-workspace-boundary FK-of-full-limit configs settle ~2 mm short),
  dual ~98% <5 mm with a rare shared-lift compromise to ~19 mm. Confirms the
  rotation component has **zero** effect on the solution.
- **Hard position-tracking suite `_solver_test_tracking.py` (no ROS): 20/20 gates.**
  Reproduces the operator's "gets stuck and can't keep tracking" report with large
  *continuous* Cartesian sweeps that cross internal singularities, a boundary
  excursion + re-acquire, cold-hard convergence, and dual-arm tracking. Typical
  tracking is p95 sub-mm; an aggressive full-amplitude sweep rides a singularity
  through with a bounded, recovering transient (mean sub-mm) — it never gets stuck.
  Boundary re-acquire takes the stuck-at-150-mm-forever case to settling home sub-mm.
- **Quest absolute-orientation + lock `_quest_orientation_test.py` (ROS import,
  no DDS): 8/8 gates.** Unchanged: drives the real `on_xr_frame` and verifies the
  quest node's orientation mapping (90° controller → 90° gripper *target*, no
  accumulation, lock freeze/resume, position clutch). NB the controller now ignores
  this streamed orientation (position-only reach), so it affects only the headset
  triad viz, not the arm.
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

- Reaching is **position-only**: a `target_pose`'s point is reached and its
  orientation is ignored (no gripper-rotation control). The 6-DOF path was removed
  from the solver — `solve_step` accepts a 3-vector or a dict with `"pos"` and
  discards any `"quat"`/`"R"`. If you want orientation control back, you'd re-add
  the weighted orientation-error rows (`_so3_log(R_target @ R_tip.T)` under the
  position rows, using `pose_jacobian`'s angular rows) and the per-arm ori gating.
  Validation: `_solver_test.py` (15/15) + `_solver_test_positions.py` (19/19,
  many positions + zero-orientation-effect guard) + `_solver_test_tracking.py`
  (20/20, hard position sweeps / singularity crossings / boundary re-acquire).
  NB the Quest node still computes/streams an orientation for its headset triad,
  but the controller ignores it — see the `quest_node.py` note above.
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
- Base `/m1/cmd_vel` is open-loop swerve; no odometry is published on a ROS topic
  yet. `swerve.SwerveOdometry` exists and the Quest viz dead-reckons cmd_vel with
  it to drive the headset model, but it is *commanded*-velocity dead reckoning
  (no encoder feedback / `/odom` publisher). Wiring `SwerveOdometry` into
  `controller_node` to publish `nav_msgs/Odometry` + a TF is the next step.
- The Isaac graph publishes `/joint_states` + `/clock` only; TF comes from
  robot_state_publisher on the ROS side. No camera/lidar/IMU bridged yet (the
  URDF has those frames if you want to add sensor graphs).
- Node/attribute names in `ros_sim.py` target the Isaac Sim 5.x
  `isaacsim.ros2.bridge`; if Isaac is upgraded, re-check them.
