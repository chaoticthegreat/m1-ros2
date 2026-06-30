# M1 ROS 2 — handoff notes for the next agent

Quick orientation for whoever picks this up next. Read this first, then
`ros2_ws/README.md` (full ROS interface) and `isaac/README.md` (Isaac details).

## What this repo is

An AgileX Ranger Air swerve base + prismatic lift + dual 7-DOF OpenArm arms
(27 DOF), simulated in Isaac Sim and controlled over ROS 2. Goal: give an arm a
Cartesian target pose and the arm joints + shared lift reach toward it; drive
the base with a velocity command. Same control code is meant to later run on the
real robot.

> **Arm mounting (changed 2026-06-24).** The OpenArm "body" extrusion
> (`openarm_body_link0`, the ~0.70 m riser) was **removed**: both arm bases now
> bolt **FLUSH directly onto the moving lift carriage** (`lift_link`). They are
> direct children of `lift_link` (left `xyz 0.0135 0.19 0.3492`, right
> `-0.0485 0.19 0.3492`, both yaw −90°) — the carriage interface minus the riser,
> then lowered a further 2 in (0.0508 m) by request. Net effect vs the old
> description: the two arms (and their whole reachable workspace) sit **~0.749 m
> lower** (arm base world z 0.629 at lift=0); the lift's 0.85 m travel more than
> covers the drop, inter-arm spacing unchanged (62 mm). Purely a structural FK
> change (verified exact to ~1e-18, no orientation leak). The Isaac **USD was
> regenerated** (`isaac/convert_urdf_to_usd.py`; verified `openarm_body_link0` gone,
> arm bases at world z 0.629), so the sim matches. **Startup posture:** all-zeros
> now folds the (low) arms into the drivebase, so the sim seeds an **arms-up HOME
> pose** (`HOME_CONFIG` in `isaac/ros_sim.py`, mirrored in the MoveIt SRDF `home`
> group_states). All ROS-side consumers (solver/FK, RSP/RViz, MoveIt SRDF,
> web/Quest viz) use the URDF and are updated. **If you edit the URDF geometry
> again, re-run the convert script** — Isaac loads the USD, not the URDF.

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
  geometric Jacobian, plus a **Drake-backed** Cartesian reach controller. The FK
  utilities (`UrdfModel`, `ArmChain`, `mat_to_quat`/`quat_to_mat`/`_so3_log`,
  `pose_jacobian`/`gripper_pose`) stay pure-numpy / dependency-free (the viz nodes
  use them for FK and to *report* gripper rotation). **The reach is POSITION-ONLY**:
  a target is a 3D point (or a dict carrying `"pos"`; any `"quat"`/`"R"` is ignored)
  and the gripper fingertip is driven to it — orientation is not constrained.

  **The inverse kinematics is solved by Drake** (`pydrake`). `ReachController`
  builds a `MultibodyPlant` from the URDF (visual/collision stripped so no meshes
  need resolving; **gripper finger joints+links and their `mimic` stripped** — see
  below; `base_link` welded to the world since targets are base-relative; a
  fingertip frame added per arm at `GRIPPER_TIP_OFFSET` — Drake FK verified equal
  to the custom FK to machine precision) and runs a position-**cost** least-squares
  IK (`InverseKinematics.AddPositionCost`; SNOPT with a major-iteration cap, IPOPT
  fallback) inside an **amortized multi-start**. This replaced a ~600-line bespoke
  multi-seed DLS / Gauss-Newton / refine / re-acquire solver that stalled in local
  minima (the operator-reported "stops solving early / doesn't reach the optimal
  solution"); the Drake NLP + multi-start drives reachable targets to **microns**
  and settles an unreachable one at the closest config the joints allow. The plant
  is built ONCE per URDF (cached process-wide in `_DIK_CACHE`, eagerly in
  `__init__`) so its ~70 ms build is never charged to a `solve_step` tick; `pydrake`
  is imported lazily, so FK-only users without Drake still work.

  `solve_step(q_meas, targets)` keeps the same public contract: a joint-name→command
  dict plus `"_dist"`; the **held sentinel** is a dict with no joint keys; the
  command *leads* the measured pose toward the solved goal by a bounded `IK_MAX_DQ`
  step, then a per-arm capped Cartesian **hold/polish** (`_IK_HOLD_ITERS`, total
  displacement capped at `_IK_ARM_HOLD_CAP`, using the retained `_stack`/`_dls`)
  lands each gripper ON its target. Two regimes:
  * **Tracking** (cache hit, target moved < `_IK_TRACK_JUMP`): one warm Drake solve
    seeded from the cached solution and regularized toward it — stays in-branch (no
    elbow/base snaps), ~1–3 ms.
  * **Cold** (first solve / arm-set change / big jump): an amortized multi-start
    (`_IK_SEEDS_PER_TICK` Drake seeds per tick, best-so-far carried in the cache
    while the command already leads toward it, so no tick blows the 60 Hz budget).
    Seed 0 is the measured/cached config (nearest branch, minimal slew); the search
    **early-stops** once a seed converges (`_IK_CONVERGED`), so an easy target
    finishes in one tick, while the diverse seeds (lift sweep + random postures,
    regularized toward mid-range for a stable posture) escape a local minimum for a
    hard target. Any arm whose target barely moved is **pinned** to its cached
    branch and the shared lift is **hard-pinned** to its cached height, so a big
    move on one arm never drags a held one. If the current pose ALREADY reaches
    every target, `solve_step` simply **holds** (no re-solve) — so re-tracking a
    planned path from a settled config never null-space-shifts and slews the tip.
  A tracking solve whose residual stays large for `_IK_REACQUIRE_TICKS` escalates
  to a cold multi-start that frees only the stuck arm(s). When both arms reach at
  once they share one stacked program, so the single lift is the least-squares
  compromise that best serves both grippers.

  **Validation (all gated suites green):** `_solver_test.py` 15/15
  (single/dual reach 100% sub-mm; worst-case `solve_step` ~21 ms, 0.5% over the
  60 Hz budget, median ~2 ms; the far-jump held-arm gate was split into a settled
  check + a bounded-transient check after the flush-mount drop — see below),
  `_accuracy_bench.py` 10/10 (M1 singularity max
  1.13 mm / p95 ~0; M2 smooth 0.0 mm; M3 dual max 0.19 mm / p99 0.14; M4 near-
  boundary max 0.13 mm, 100% <1 mm; LAT max ~20 ms, 0.1% over budget),
  `_solver_test_positions.py` 21/21 (300 single + 200 dual reachable points 100%
  sub-mm; rotation has zero effect on the solution — bit-identical to the bare
  point), `_solver_test_tracking.py` 21/21 (smooth/wide/full singularity sweeps
  sub-mm; boundary excursion saturates and re-acquires home sub-mm),
  `_solver_test_pathing.py` 23/23 (plan-and-track lands on B at ~0.16 mm, max tip
  step 16.7 mm; unreachable target gives up and drops back under budget).
  **Drake is now a runtime dependency** of `m1_control` — install it for the ROS
  interpreter with `/usr/bin/python3 -m pip install --user --break-system-packages drake`.

  **NB — two LIVE-only bugs the offline suites can't catch (they use perfect
  feedback + zeroed fingers), found driving the real sim+Quest loop:** (1) the
  gripper fingers are URDF `mimic`-coupled, and live the controller drives them,
  so the MEASURED finger pair violates the mimic relation; pinning those joints in
  the IK made the whole Drake program INFEASIBLE → SNOPT returned garbage → arms
  100+ mm off ("all targets red"). Hence the finger/mimic strip above — the IK
  plant has no fingers. (2) A stuck DUAL solve must re-acquire BOTH arms with the
  lift FREE; the lift-pin is only for the cold far-jump case. **Always validate the
  live closed loop** (compare the controller's command-fingertip, which should be
  ~0 mm, against the sim state). The sim arm gain `ARM_KP` in `isaac/ros_sim.py`
  was also raised 9000→30000 so the simulated arm actually holds a correct command
  at near-max reach (was sagging ~3-4 cm under gravity → amber); sim-fidelity only.
- `.../collision.py` — **dependency-free capsule self-collision model** (new).
  Approximates each link as a capsule (its FK centerline + a radius) and reports
  the signed clearance between checked pairs via segment-to-segment distance —
  pure numpy, no meshes/FCL. Scope is **self-collision only**: each arm's MOVING
  links (joint1→fingertip) vs the OTHER arm's links + its static riser, and vs the
  shared body column. The rigid mount risers and the two arms' shared lower chain
  are body, not arm, so they never read as a constant collision; intra-arm and
  shared-mount pairs are excluded. Radii are conservative (arm 5 cm, gripper 6 cm,
  column 11 cm, base 19 cm) — calibrated so ~99 % of random reachable dual poses
  read clear while genuine cross-overs collide. Also exposes a finite-difference
  clearance gradient and a witness-point separating direction (`clearance_detail`)
  the planner uses to avoid/route around collisions. Self-test (gated 9/9):
  `PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.collision`.
- `.../trajectory.py` — **collision-free Cartesian path planner** (new, *test +
  viz only*; the live controller stays reactive). `TrajectoryPlanner.plan(start_q,
  {arm: goal_point})` interpolates a straight task-space line A→B, solves each
  waypoint with the SAME DLS core the live solver uses (`ReachController._stack` /
  `_dls`) warm-started for branch continuity, and keeps it collision-free in two
  stages: (1) **null-space avoidance** — push joints up the clearance gradient
  *inside the fingertip task null space*, so the redundant DOFs open a gap without
  moving the tip (re-converging the task after each nudge, TASK PRIORITY — it never
  trades the goal away); (2) **path detour** — for an intermediate waypoint whose
  *target point itself* is task-coupled-colliding (a tip pinned near the body),
  bow the fingertip path off the line along the separating direction (endpoints
  A/B are never moved). Returns a `Trajectory` of waypoints (q, residual,
  clearance, colliding) + summaries (`collision_free`, `reached`, `min_clearance`)
  the tests gate on and the Quest viz draws. NB: in this robot's workspace self-
  collisions are rare and usually a tip pinned near static structure (null-space
  can't open those — honestly flagged `colliding`, drawn red). Smoke (4/4):
  `… /usr/bin/python3 -m m1_control.trajectory`. **NB (collision review fix):** when
  only one arm is planned (the Quest preview plans each arm separately), the
  collision FK is seeded with the OTHER arm's **measured** joints (`plan`'s
  `background`, passed into `CollisionModel.clearance_of_vec/_gradient/_detail`),
  so a path is checked against the other arm where it ACTUALLY is — not a phantom
  straight-out (q=0) arm that would let a path through the real arm read green.
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
 **Position-only** (orientation fully removed): hands drive the target POINT, the
 gripper rotation is not controlled, and the in-headset overlay shows a **REACH
 ERROR HUD** instead of orientation triads.
 Serves ONE self-contained WebXR page over **HTTPS** (self-signed cert auto-made
 with openssl into `~/.cache/m1_quest/`; WebXR needs a secure context over LAN
 IP). The Quest browser opens it, enters immersive-ar (passthrough) and POSTs
 both controllers' grip-space **positions** + buttons to `/api/xr` (orientation is
 no longer sent). **Two control schemes**, toggled live by an **A/X+B/Y chord**
 (both buttons together, edge-triggered; `self.control_mode`, default `relative`,
 surfaced in the viz + 2D page + in-VR HUD tag):
   * **relative** (default, unchanged): hold Grip to "grab"; the target follows the
     hand's **MOTION** (delta, heading-projected). Controllers **CROSS-mapped**
     (left hand → right arm). Standard ratcheted VR teleop.
   * **absolute** ("embodiment", new): the robot's chest/arms frame **rides the
     headset** (`bodyAnchor`, a chest-locked base_link frame). Its POSITION follows
     the head every frame (arms stay on you as you walk), but its FORWARD/yaw is a
     **STABLE captured direction** (grabbed on entering absolute + on B/Y recenter,
     robust to looking up/down via the head-up fallback) — continuously following the
     head GAZE re-rotated the whole mapping when you looked down/around (the "arm
     forward goes 90° to the right" bug). So the two robot arms overlay the operator's
     arms and **mirror them**. The PAGE maps each hand into `base_link` via that body
     frame (`pos_base` = `inv(bodyAnchor.matrixWorld)·hand_world`) and the node sets
     `target = engage + scale·(hand_base − hand_base@engage)` (scale 1.0, or
     `PRECISION_SCALE` fine). **CROSS-mapped** (left controller → right arm) — same
     as relative; verified live (the arm meshes render mirrored from the URDF's
     left/right labels, so cross-map is what puts each gripper under the right hand).
     Grip-gated (release frees). A SECOND set of the two **arm meshes** is
     drawn under `bodyAnchor` (live joints) as the on-body overlay; the existing full
     robot stays ~1.1 m in front, joystick-drivable. The chord drops any grab; the
     page suppresses B/Y recenter while A/X is down. **NB the body-frame mapping +
     overlay are JS-only (live-only) — validate handedness/alignment in the headset.**
 **Thumbstick click toggles a per-arm PRECISION mode** (`fine`, edge-triggered):
 while on, hand motion is scaled by `PRECISION_SCALE` (0.25) for fine sub-cm target
 placement (scales the delta in relative, the displacement-from-engage in absolute).
 Trigger→gripper, A/X→re-seed the target to the live fingertip ("home to here"),
 B/Y→recenter the hologram.
 Validated headless by `_quest_position_test.py` (18/18: relative clutch, precision
 toggle + edge-trigger, A/X reseed, base drive, the error-window data path, AND the
 absolute snap-on-engage / 1:1 tracking / fine-scale / chord toggle, driving the
 real `on_xr_frame`/`_viz_locked`/`snapshot`). **The client-side anchor-inverse
 (`pos_base`) is JS three.js matrix math — LIVE-only, not unit-testable; validate
 reach-to-hologram in the headset.**
 **Base drive (thumbstick push):** LEFT stick fwd/back→drive fwd/back (vx),
 left/right→strafe (vy); RIGHT stick left/right→turn (yaw); smooth rescaled
 deadzone; cmd_vel is set every frame so centring the stick stops at once
 (BASE_HOLD now only guards a lost connection). The headset **robot model
 actually drives through the room**: `SwerveOdometry` dead-reckons the commanded
 cmd_vel into a base pose streamed as `viz["base"]`, and the page hangs all link
 meshes/markers off a `robotBase` group (inside the
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
 are obvious), plus a floating **REACH ERROR HUD** (a canvas-textured panel that
 billboards to face the operator, showing each arm's target↔fingertip error in mm,
 color-coded) anchored above the robot;
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
 **Performance pass (the viz had become laggy):** the per-`/api/xr` work is no
 longer done under `_lock` — `on_xr_frame` takes a cheap snapshot under the lock
 then builds the viz (the ~36-link FK + serialization) OUTSIDE it, so the heavy
 work no longer serializes against the 60 Hz `_tick` and the `/joint_states`
 callback. The full-robot `links` + per-arm skeleton FK are **memoized on a
 joint-version counter** (bumped only when `q_meas` actually changes), so the many
 POSTs that land between two joint frames reuse the FK instead of recomputing it;
 only the cheap hand-dependent bits (target sphere, error line, base pose) are
 rebuilt each frame. (This rides on the kinematics FK caching above.) Client-side,
 the REACH-ERROR HUD canvas is redrawn/re-uploaded only when its numbers change,
 and the wireframe/trajectory buffers are preallocated and filled in place instead
 of reallocated every frame. **Trajectory preview:** a throttled (~3.5 Hz)
 background thread plans a collision-free path (`trajectory.py`) from each arm's
 live fingertip to its current target and streams it as `viz["traj"]`; the page
 draws it as a per-arm polyline + waypoint dots, green where clear and **red where
 a waypoint self-collides**, so the operator sees the planned motion (and any
 collision) before committing. Planning is off the request path (snapshot under
 lock, plan outside, store under lock) so it never stalls a frame.
 **Performance pass 2 — the in-headset model "trailed/stuttered" (smooth
 passthrough, jerky robot).** Root cause was NOT the GPU or the server (measured
 `on_xr_frame` is 0.009–0.33 ms/frame, 4.1 KB payload — see `_quest_perf_bench.py`):
 the robot pose `lastViz` only refreshed when an `/api/xr` POST round-tripped
 (gated one-in-flight) with **no interpolation**, and **Nagle's algorithm was on**
 (stdlib `disable_nagle_algorithm` defaults False) so each request ate a ~40 ms
 delayed-ACK stall on a real network (loopback never reproduces it, so the offline
 suites couldn't catch it). Fix, three parts: (1) **server transport** — the
 `Handler` sets `disable_nagle_algorithm = True` (TCP_NODELAY) and `_send` now
 writes the whole HTTP response (status+headers+body) in ONE `wfile.write` (one
 segment), keeping HTTP/1.1 keep-alive; (2) **client snapshot interpolation** —
 the page keeps the **two most recent** server frames (`vizPrev`/`vizCur` via
 `ingestViz`, with client receive-times + RTT/interval EMAs) and each render frame
 `interpState`/`updateRobot` **lerp/slerp** link poses + markers + base ~one
 inter-arrival interval IN THE PAST between them, **clamped** so a stall holds the
 last pose (no extrapolation) — turning a jittery ~30 Hz stream into smooth motion
 at the headset's 72–90 Hz, regardless of WiFi jitter; the `keepalive:true` fetch
 flag (beacon pool) was dropped and all per-frame `new Vector3/Quaternion`
 allocations hoisted to reused scratch. Cost: ~one interval (~20–40 ms) of VISUAL
 latency on the **preview only** — arm/base commands publish at 60 Hz server-side,
 unaffected. **NB (review-found, LIVE-only):** a **B/Y recenter** is a base
 discontinuity (old drifted pose → zeroed anchor); the fetch `.then` nulls
 `vizPrev` AFTER `ingestViz` on a `place` so `interpState` SNAPS to the anchor for
 one frame instead of swooping the model across the room. (3) **`/?perf` dev HUD**
 — opening the page with `?perf` shows a second billboarded panel (render FPS,
 data-update Hz, POST RTT, payload KB) for on-device before/after metrics; zero
 clutter on the normal URL. Server side validated by `_quest_perf_bench.py`
 (3/3: TCP_NODELAY set, end-to-end viz over real TLS correct, **1291 req/s** on one
 keep-alive connection); `_quest_position_test.py` stays 10/10 (the data path is
 unchanged). The interpolation/recenter paths are JS-only — **validate the smooth
 model + the B/Y snap in the live headset** (the offline suites can't drive the JS).
 **Performance pass 3 — the viz "lags majorly and freezes for 10s+ at times"
 (intermittent hard freeze, smooth passthrough).** Diagnosed with a parallel
 investigation + a live planner benchmark (which **refuted** the obvious
 "trajectory worker hogs the GIL" theory: even 4 s of back-to-back pathological
 plans kept the `/api/xr` handler's response build < 1 ms — numpy releases the GIL
 and CPython's 5 ms switch interval keeps it fair). **Real root cause: a missing
 request deadline.** The headset POSTs are gated **one-in-flight**, so a SINGLE
 stalled request (roaming/sleeping Quest WiFi → TCP RTO backoff stacking to
 seconds, or a half-open connection) suppresses ALL further POSTs; `ingestViz`
 never advances and the two-frame interpolation clamps to the **last pose** —
 frozen robot, smooth passthrough, for the full network stall, self-recovering
 when the request finally settles. The server compounded it: the `/api/xr`
 keep-alive handler had **no socket timeout**, so a wedged SSL write parked the
 sole POST thread for the OS TCP timeout (minutes). Loopback never stalls, so the
 offline suites couldn't catch it (same blind spot as the Nagle bug). Fix, three
 parts: (1) **client `AbortController`** on the `/api/xr` fetch (`POST_TIMEOUT_MS`
 700 ms) so a hung request self-aborts and the next frame re-POSTs — a 10 s freeze
 becomes a sub-second held-pose hiccup; a `POST_BACKOFF_S` (0.3 s) pause after an
 abort stops a flapping link from TLS-handshake-thrashing (abort drops the
 keep-alive connection). (2) **server** `Handler.timeout = 10.0` (reaps an
 idle/backgrounded connection) + the `_send` write-failure handler broadened to
 `except OSError: self.close_connection = True` (tears down a wedged write instead
 of parking the thread — the primary reaper). (3) **planner amplifier** —
 `trajectory.plan(deadline=...)` wall-clock budget (the worker passes
 `TRAJ_PLAN_BUDGET` 0.15 s/arm); the arms mount flush+low so descending/inward
 targets pin the tip near the body column where an unbudgeted plan ran **0.5–2 s+**
 and pinned a core (measured worst 1908 ms → 236 ms; common ~1 cm clutch step
 unchanged at ~19 ms). The preview is cosmetic, so once over budget the remaining
 waypoints drop avoid/detour and cap their task solve (still flagged colliding/red).
 Validated: `_solver_test_pathing.py` 23/23 (default `deadline=None` = full
 fidelity, gates unchanged), `_quest_position_test.py` 10/10, `_quest_perf_bench.py`
 3/3 (1819 req/s, TCP_NODELAY intact — the timeouts only fire on the failure path).
 Also hoisted `reachColor`'s `THREE.Color` to reused constants (no per-frame alloc)
 and added a **WebGL context-loss/restore** handler (a memory-pressure GPU context
 loss on the Quest otherwise blanks the model permanently — a vanish, not a freeze).
 **The freeze fix is failure-path-only (happy path byte-identical) — validate the
 live headset:** open `/?perf`, drag a clutched hand DOWN+INWARD (body-pinned
 colliding regime) and confirm the model keeps refreshing with at most sub-0.5 s
 hiccups (no multi-second freeze), and briefly walk out of WiFi range and confirm
 it holds then resumes within ~0.7 s of reconnect. The offline suites can't
 reproduce a WiFi stall. NB: a transport rewrite to a push/WebSocket library
 (aiohttp/`websockets`) WOULD structurally remove the request/RTT coupling, but it
 is far more new code + failure modes (asyncio loop alongside `rclpy.spin`, TLS
 re-plumbing) than this ~6-line deadline — a deliberate future architecture choice,
 not the freeze fix.
- `.../web_node.py` — `ros2 run m1_control m1_web`: browser control panel on
 http://localhost:8080. Same `/m1/*`-only bridge (sim + real). Stdlib HTTP
 server + embedded HTML/JS (no extra deps); base drive pad, per-arm Cartesian
 targets + gripper, live fingertip/dist readout, dead-man'd base. A status dot
 reflects whether `/joint_states` is live. UI is themed after anthropic.com
 (warm cream, serif headings, clay accent). An untouched arm's panel target
 re-syncs to its live fingertip so nudges stay relative to where the arm
 actually is (the lift is a shared compromise when both arms reach).
- `ros2_ws/src/m1_bringup/launch/bringup.launch.py` — RSP + controller + RViz
  (the **sim** path; consumes Isaac's `/joint_states`).
- `assets/{ranger_air,openarm}_description` — also ROS 2 packages (symlinked into
  `ros2_ws/src/`); hold URDF + meshes.

## Real-hardware deployment (Damiao CAN motors + AgileX base)

Full guide: **@ros2_ws/HARDWARE.md**. Design + the OpenArm-MoveIt-vs-Drake
comparison: `docs/superpowers/specs/2026-06-23-real-hardware-deployment-design.md`.

**Decision (hybrid):** reuse OpenArm's Damiao CAN/ros2_control hardware layer; keep
the Drake position-only IK as the reactive brain; MoveIt is optional/later for
*planned collision-aware* moves only (it structurally **cannot** express the
shared-lift-serving-both-arms coupling — KDL is one-chain-one-tip — nor
position-only reach nor 60 Hz reactive teleop; the Drake brain does all four). The
arms are customized OpenArm (Damiao DM motors); the base is (likely stock) AgileX
Ranger Air whose firmware takes only body `Twist` (mode-switched) — `swerve.py`'s
per-module output can't drive it (memory `agilex-ranger-no-per-module-cmd`).

**The `/m1/*` + `/joint_states` contract is invariant** (27-DOF name order, wheels=
velocity, rest=position, fingers mimic-coupled), so the brain + all operator nodes
+ all gated suites are UNCHANGED. The real-hardware seam replaces `isaac/ros_sim.py`
with a `ros2_control` stack + bridges:
- `m1_hardware/` (NEW, C++) — `M1SystemInterface : hardware_interface::SystemInterface`,
  forked from `openarm_hardware` + vendored `openarm_can` (Apache-2.0), **ported
  Humble→Jazzy**, generalized DOF (lift added), per-joint CAN id/model/gains from
  URDF params. Drives Damiao **MIT mode** over SocketCAN. Loads with NO bus present
  (logs "no bus / mock I/O"); live motor I/O is the on-hardware checkpoint.
- `m1_can_tools/` (NEW, Python) — a **hardware-free** byte-exact DM CAN codec
  (`dm_protocol`), a pluggable `Transport` (fake/socketcan/serial, lazy backends),
  a maintenance-mode `MotorBus`, and **`m1_hwconfig`** — a web page to scan/assign/
  zero/limit/jog/test motors (`ros2 run m1_can_tools m1_hwconfig` → :8090). Bus
  ownership is maintenance (config tool) XOR run (ros2_control) — never both.
- `m1_control/` adds three pure-function bridge nodes: `m1_joint_bridge`
  (`/m1/joint_command` → `arm_position_controller`, the 17 commanded upper-body
  joints; the 2 `finger_joint2` are state-only mimics), `m1_base_bridge`
  (`/m1/cmd_vel` → AgileX body Twist + motion-mode), `m1_ranger_shim` (AgileX wheel
  feedback → `/joint_states`).
- `m1_bringup/` adds `hardware.launch.py` (`use_mock:=true` mock_components default,
  `use_mock:=false` real plugin, `use_base:=`), `m1.ros2_control.xacro`,
  `m1_controllers.yaml` (forward_position + JSB + per-arm JTC; `enforce_command_limits`
  on), `m1_joint_limits.yaml`.

Run: `ros2 launch m1_bringup hardware.launch.py use_mock:=true` (offline) or
`use_mock:=false can_interface:=can0 motor_map:=…` (real). NB: clean up launches
with **SIGINT to `ros2 launch`** then a PID sweep — do NOT `pkill -f <node-name>`
(the pattern matches your own shell's command line and SIGKILLs your shell).

**Validated offline (no hardware):** `m1_can_tools` 34/34; `_bridge_test.py` 15/15;
mock ros2_control loop (controllers active, brain reach flows brain→bridge→
controller→mock); real plugin loads + activates with no bus; config page serves;
**all brain gated suites still green (113/113)**. Deferred to hardware: per-joint
sign/offset, the live closed loop (command-fingertip ≈ measured), gain tuning, the
base Twist path. See HARDWARE.md "Deferred live-validation checkpoints".

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
- **Solver is now Drake-backed** (see the `kinematics.py` key-file note). The
  numbers below are with the Drake `InverseKinematics` + amortized multi-start
  solver; they are equal-or-better than the bespoke DLS solver it replaced.
- **Full solver suite `_solver_test.py` (no ROS): 15/15 gates pass.** Covers
  reachability (single+dual), continuous tracking, hold-under-disturbance,
  latency distribution, and stress. Key results: **100% of reachable single-arm
  targets <1 mm *from a cold start*** (mean ~0.01 mm), dual-arm 100% <5 mm (max
  ~0.08 mm), **worst-case `solve_step` ~21 ms (median ~2 ms, 0.5% over the 60 Hz
  budget — a goal, not a cutoff)**. Held-arm far-jump disturbance: SETTLES at
  0 mm with a brief ~67 mm transient while the shared lift slews to serve the
  other arm's cold jump (was ~7 mm pre-flush-mount; the lower arms couple the
  lift's z-slew into the held fingertip more — fully recovers, nowhere near the
  old >130 mm "snap" bug). Gate split accordingly: settled <2 mm + transient
  bounded <80 mm.
  Steady-state `solve_step` ~1 ms; the cold multi-start search is amortized across
  ticks (one or two Drake seeds per tick) so no tick blows the budget.
- **Many-positions suite `_solver_test_positions.py` (no ROS): 21/21 gates** (the
  two hard latency gates became one worst-case-bounded gate under the 60 Hz-is-a-
  goal policy; this suite is ~100% cold solves so its median is not representative).
  Simulates a large position set: 300 single-arm + 200 dual reachable points and a
  7³ workspace grid (reachable single/dual 100% <1 mm, grid reached 100% <2 mm).
  Large-sample distribution: single ~99% <2 mm / 100% <5 mm
  (a few near-workspace-boundary FK-of-full-limit configs settle ~2 mm short),
  dual ~98% <5 mm with a rare shared-lift compromise to ~19 mm. Confirms the
  rotation component has **zero** effect on the solution.
- **Hard position-tracking suite `_solver_test_tracking.py` (no ROS): 21/21 gates.**
  Reproduces the operator's "gets stuck and can't keep tracking" report with large
  *continuous* Cartesian sweeps that cross internal singularities, a boundary
  excursion + re-acquire, cold-hard convergence, and dual-arm tracking. Typical
  tracking is p95 sub-mm; an aggressive full-amplitude sweep rides a singularity
  through with a bounded, recovering transient (mean sub-mm) — it never gets stuck.
  Boundary re-acquire takes the stuck-at-150-mm-forever case to settling home sub-mm.
  (Still 20/20 with the persistent-refinement change — the refine is steady-gated
  so it never perturbs a tracking sweep.)
- **Point-to-point trajectory suite `_solver_test_pathing.py` (no ROS): 23/23
  gates**. Tests "going BETWEEN two points and landing accurately", not just
  cold solves: (A) plan a collision-free Cartesian path A→B and drive the
  controller along it — lands on B at ~0.2 mm, follows the path p95 <0.2 mm, no
  fingertip jump; (B) **warm solve** — settle AT A, then command B and converge
  (100% <1 mm), the warm/in-motion case; (C) given collision-free endpoints the
  planner keeps the whole path self-collision-free (24/24) and honestly flags any
  it cannot; (D) avoidance never breaks a reach nor lowers clearance (20/20); (E)
  **persistence** — hard held dual targets driven sub-mm, and a genuinely
  unreachable target settles at the closest config with every tick under the 60 Hz
  budget (the refine gives up cleanly). Run: `/usr/bin/python3 _solver_test_pathing.py`.
- **Quest position teleop `_quest_position_test.py` (ROS import, no DDS): 10/10
  gates.** Drives the real `on_xr_frame`/`_viz_locked`/`snapshot`: position clutch,
  PRECISION mode (thumbstick-click, edge-triggered, scales motion by
  `PRECISION_SCALE`), A/X reseed to the live fingertip, thumbstick base drive, and
  the REACH-ERROR data path the in-VR HUD + 2D page render. (Replaces the old
  `_quest_orientation_test.py`, retired with the orientation removal.)
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
- `_e2e_check.py` was **removed** (maintenance pass): it had drifted to a broken,
  unrunnable state — it POSTed to a non-existent `/api/cmd` route, used stale
  command types (`target`/`cmd_vel`/`stop`) that `apply()` rejects, and `KeyError`'d
  on a removed `njoints` field before any check ran — and was not in the gated
  suite. The reach path is covered by `_ros_reach_check.py` instead.
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
  Validation: `_solver_test.py` (15/15) + `_solver_test_positions.py` (22/22,
  many positions + workspace coverage + zero-orientation-effect guard) +
  `_solver_test_tracking.py` (20/20, hard position sweeps / singularity crossings
  / boundary re-acquire) + `_accuracy_bench.py` (10/10 accuracy regression gates).
  The Quest node is also fully position-only now (orientation removed; thumbstick-
  click is a precision toggle) — see the `quest_node.py` note above.
- The LIVE controller has no collision avoidance / planning — it's a reactive
  Jacobian controller solved to convergence each tick. Collision-free *planning*
  now exists offline (`collision.py` + `trajectory.py`, used by the trajectory
  tests and the Quest path preview) but is intentionally NOT wired into the live
  control loop. The capsule model is **self-collision only** (arm↔arm, arm↔body)
  and a conservative approximation (capsules, no environment/world model); the
  planner reaches the goal first and avoids collisions best-effort (null-space +
  path detour), honestly flagging the rare task-coupled pose it cannot open. To
  make execution collision-aware you'd either gate the live command on
  `CollisionModel.clearance` or drive the controller along a pre-planned
  `Trajectory`. **A MoveIt 2 config now exists** for planned collision-aware moves
  (`m1_bringup/moveit/`, `m1_moveit.launch.py`, Phase 3) — `left_arm`/`right_arm`
  KDL groups + a `both_arms_lift` joint-space group (the lift can't go through KDL
  IK, so it's OMPL joint-space only); validated on mock (all groups plan; per-arm
  plan+execute lands). It's **additive/optional**, hot-swapped JTC↔forward
  controller for planned moves; the reactive Drake teleop + shared lift stay on the
  brain. Open gap: no combined JTC to *execute* a `both_arms_lift` trajectory yet.
- Shared-lift tradeoff: the lift is one prismatic joint feeding both arms, so
 when both arms have targets the stacked solve picks the single lift height that
 minimises the combined (equal-weight) fingertip error. Two arms with targets at
 very different heights therefore share the lift as a compromise; clear one
 arm's target if you want the lift to commit entirely to the other.
- Worst-case `solve_step` latency is ~20 ms (median ~1–2 ms, ≤0.5% over the
  16.7 ms 60 Hz budget — a goal, not a hard cutoff): the cold Drake multi-start is
  **amortized across ticks** (`_IK_SEEDS_PER_TICK` Drake solves + a resumable `job`
  in the cache), so no single tick runs the whole search. A single Drake solve is
  ~2–10 ms; the ~70 ms plant build is one-time (in `__init__`, cached), never on a
  tick. Continuous tracking / steady ticks are ~1 ms. The trade-off is that a hard
  cold target *converges over a handful of ticks* — the command leads toward the
  best-so-far meanwhile, so the arm starts moving immediately and the goal sharpens.
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
- **Isaac USD regenerated for the flush-mount change (2026-06-24).** The arms were
  re-mounted flush on the lift carriage (extrusion removed, ~0.70 m lower) in
  `assets/ranger_air_description/urdf/ranger_air_description.urdf`. Isaac loads the
  USD (`assets/usd/ranger_air.usd` + `configuration/*.usd`), not the URDF, so it was
  re-derived by re-running `/home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py`
  (headless; ROS-bridge OmniGraph is added at runtime by `ros_sim.py`, so nothing is
  lost). Verified: `openarm_body_link0` absent, arm bases at world z 0.68 (matches the
  URDF FK), gripper `mimic` honored. **Reminder: any future URDF geometry edit needs
  that convert script re-run** or the sim silently diverges. The ROS-side stack
  (solver, RSP/RViz, MoveIt SRDF, web/Quest viz, all gated suites) uses the URDF and
  is validated (10/10 suites green; browser-rendered the new model).
- **Base mass 9.63 -> 63 kg to stop the robot TIPPING OVER (2026-06-29; USD
  regenerated).** The operator reported "the solver fails to converge on some
  positions"; the arm sat ~80-135 mm short of high/forward targets. It was NOT the
  solver (reaches 0 mm offline), NOT self-collision (Isaac imports with
  `self_collision=False`; the real-mesh probe found only constant mount overlaps),
  and NOT gravity (Drake gravity torque ~7 N·m vs 40 N·m limits). The real cause:
  the sim `base_link` mass was a **9.63 kg placeholder**, leaving the free-standing
  base **top-heavy** (lift+arms ~15 kg up to ~1 m over a 0.39x0.34 m wheel
  footprint). A high/forward dual reach pushed the combined COM past the support
  edge and **the whole robot tipped over** -- the arm was then mashed on the ground
  (joint1 effort pinned at its -40 N·m limit), so reaches "failed". Fix: set
  `base_link` to a realistic **63 kg** (real AgileX Ranger Air) + a matching
  ~0.5x0.4x0.25 m box inertia, COM kept low (z~0.14), in the URDF; **re-ran the
  convert script** so the USD carries the new mass. Verified upright: the same
  previously-failing target now reaches **7.6 mm / 13.7 mm** with joint1 at its ~7
  N·m gravity load (was 134/80 mm, saturated). Mass-only is a *dynamics* change, so
  the kinematic ROS stack/gated suites are unaffected. **NB:** `self_collision` is
  off in the sim, so there is no arm self-collision protection there; the live
  controller is still collision-unaware (a known limitation) -- if you add
  collision-aware control, `tools/convert_collision_meshes.py` converts the URDF's
  `.dae/.stl` collision meshes to Drake-loadable `.obj` (Drake hulls only
  `.obj/.vtk/.gltf`). Diagnosis tooling left at repo root: `_solver_failure_logger.py`
  (live reach-error + joint-effort logger), `_diag_contact.py`, `_gravity_drake.py`.
