# M1 — ROS 2 control stack

A fully simulatable ROS 2 (Jazzy) setup for the M1 robot: an AgileX Ranger Air
swerve base + a prismatic lift + dual 7-DOF OpenArm arms. Give the arms a
Cartesian target pose and the arm joints + lift reach toward it; drive the base
with a velocity command.

## Architecture

The system is split into two processes that talk over DDS, so the same control
"brain" runs against the simulator today and the real robot later:

```
┌───────────────────────────┐   /joint_states, /clock     ┌──────────────────────────┐
│  Isaac Sim (python.sh)    │ ───────────────────────────▶│  ROS 2 control (Jazzy)   │
│  isaac/ros_sim.py         │                              │  m1_control / m1_bringup │
│  ROS 2 bridge OmniGraph   │ ◀─────────────────────────── │  IK + swerve brain       │
│  = robot driver stand-in  │        /m1/joint_command     │  + robot_state_publisher │
└───────────────────────────┘                              └──────────────────────────┘
```

Why split? Isaac Sim's bundled Python is 3.11 while ROS 2 Jazzy is 3.12, so
`rclpy` cannot be imported inside Isaac. The Isaac side therefore uses the
**ROS 2 bridge OmniGraph** (Isaac ships its own ROS 2 libraries) and only
exposes standard topics. To deploy on hardware you replace the Isaac process
with the real robot driver — the `m1_control` brain and launch files are
unchanged.

## Packages

| Package                  | Type          | What it does |
|--------------------------|---------------|--------------|
| `ranger_air_description` | ament_cmake   | URDF + meshes for the base/lift (also pulls in OpenArm meshes). |
| `openarm_description`    | ament_cmake   | OpenArm arm + pinch-gripper meshes. |
| `m1_control`             | ament_python  | The brain: `m1_controller` node (DLS Cartesian reach + swerve), `m1_send_pose` helper. |
| `m1_bringup`             | ament_python  | Launch files + RViz config. |

`ranger_air_description` and `openarm_description` are symlinks in `src/` back
to the repo's top-level `assets/` folders, so the meshes are not duplicated.

## ROS 2 interface

The controller (`m1_control/m1_controller`) is the only node you talk to:

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| in  | `/joint_states`              | `sensor_msgs/JointState`     | feedback from sim/robot |
| in  | `/m1/left_arm/target_pose`   | `geometry_msgs/PoseStamped`  | left gripper target (base_link frame) |
| in  | `/m1/right_arm/target_pose`  | `geometry_msgs/PoseStamped`  | right gripper target |
| in  | `/m1/cmd_vel`                | `geometry_msgs/Twist`        | swerve base (x fwd, y left, yaw) |
| in  | `/m1/left_arm/gripper`       | `std_msgs/Float64`           | 0 = closed … 1 = open |
| in  | `/m1/right_arm/gripper`      | `std_msgs/Float64`           | 0 = closed … 1 = open |
| out | `/m1/joint_command`          | `sensor_msgs/JointState`     | position → steer/lift/arms/fingers, velocity → wheels |
| out | `/m1/target_markers`         | `visualization_msgs/MarkerArray` | per-arm target sphere + label, fingertip, error line (RViz) |

Reaching is **position-only** today (the gripper fingertip is driven onto the
target point; orientation in the PoseStamped is currently ignored). The shared
lift is solved jointly with the arm(s) that are actively reaching, so high/low
targets become reachable by raising/lowering the torso. An arm that has reached
and gone idle is "parked": it holds its joints and rides the lift, so a
stationary arm no longer pins the lift in place — commanding a single arm to a
high/low pose now visibly recruits the lift. An out-of-reach target simply pulls
the arm + lift to their limits ("as close as possible") — the base is never
moved by the reach. When both arms reach at once they are solved together so the
shared lift is a compromise that gets **both** grippers as close as possible.
Target poses are published as RViz markers (`/m1/target_markers`).

> Operator targets are "sticky": once an arm's target is set (seeded from the
> live fingertip on connect, or commanded by the operator) it only changes when
> the operator changes it. An idle arm that rides the shared lift no longer has
> its stored target silently re-synced, so commanding one arm never moves the
> other arm's target.

## Build

```bash
cd /home/jerry/Downloads/m1-ros2-setup/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run (simulation)

Open two terminals.

**Terminal 1 — Isaac Sim (the simulated robot):**

```bash
source /opt/ros/jazzy/setup.bash          # so the bridge picks the right RMW
cd /home/jerry/Downloads/m1-ros2-setup
# Convert the URDF to USD once if assets/usd/ranger_air.usd is missing:
#   /home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py
/home/jerry/isaac-sim/python.sh isaac/ros_sim.py            # add --headless for no GUI
```

This publishes `/clock` + `/joint_states` and subscribes `/m1/joint_command`.
A status summary is written to `isaac/last_ros_sim_report.txt`.

**Terminal 2 — the ROS 2 brain + RViz:**

```bash
source /opt/ros/jazzy/setup.bash
source /home/jerry/Downloads/m1-ros2-setup/ros2_ws/install/setup.bash
ros2 launch m1_bringup bringup.launch.py            # use_rviz:=false to skip RViz
```

## Web control panel (sim and real robot)

`m1_web` serves a browser control panel and bridges it to the controller over
the same `/m1/*` topics, so it drives the simulator and the real robot
identically. It has **no extra dependencies** (Python stdlib HTTP server +
plain HTML/JS) and works headless.

```bash
source /opt/ros/jazzy/setup.bash
source /home/jerry/Downloads/m1-ros2-setup/ros2_ws/install/setup.bash
ros2 run m1_control m1_web        # then open http://localhost:8080
```

The panel has: a base drive pad (hold buttons or use `W/A/S/D`, `Q/E`, `Space`),
per-arm Cartesian target nudges + a "go to X Y Z" box, gripper sliders, and a
live readout (fingertip position, distance-to-target, lift height). A status dot
turns green only when `/joint_states` is arriving — if it is red, **nothing will
move**, because the controller waits for feedback before it acts (start the sim
or the robot first). Override the bind address/port with parameters:

```bash
ros2 run m1_control m1_web --ros-args -p host:=0.0.0.0 -p port:=9000
```

## Interactive teleop in a terminal (sim and real robot)

`m1_teleop` is the operator console. It only publishes to the controller's
`/m1/*` input topics, so the *same* keyboard interface drives the simulator and
the real robot — whatever is providing `/joint_states` + applying
`/m1/joint_command` underneath. It is plain text, so it works headless / over
SSH. Run it in its own terminal (it needs keyboard focus):

```bash
source /opt/ros/jazzy/setup.bash
source /home/jerry/Downloads/m1-ros2-setup/ros2_ws/install/setup.bash
ros2 run m1_control m1_teleop
```

Each arm keeps an internal Cartesian target (in `base_link`), seeded from the
live `/joint_states` so the arm holds still on connect; the keys nudge it.

```
Base:  w/s fwd/back   a/d turn L/R   q/e strafe   space stop
Arm:   i/k +x/-x   j/l +y/-y   u/o up/down   (active arm)
Sel:   t switch arm   [ / ] close/open gripper
Misc:  +/- step size   h re-seed & stop   ESC / Ctrl-C quit
```

Unlike `isaac/teleop.py` (which drives the Isaac articulation directly and is
sim-only), `m1_teleop` is hardware-agnostic.

## Send commands (scripted / one-shot)

```bash
# Reach the left arm to a point (metres, base_link frame: x fwd, y left, z up):
ros2 run m1_control m1_send_pose --arm left  --xyz 0.30 0.20 0.95

# Or publish the pose directly:
ros2 topic pub --once /m1/right_arm/target_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: base_link}, pose: {position: {x: 0.30, y: -0.20, z: 0.90}}}'

# Drive the swerve base (forward + turn):
ros2 topic pub -r 20 /m1/cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.4}, angular: {z: 0.3}}'

# Open / close a gripper:
ros2 topic pub --once /m1/left_arm/gripper std_msgs/msg/Float64 '{data: 1.0}'
```

## Test / verify

There is no automated test suite; verification is a few quick manual checks.

**1. Build is clean.** From `ros2_ws/`:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install        # should finish with 4 packages, no errors
source install/setup.bash
```

**2. Kinematics + reach (standalone, no GPU/sim/DDS needed).** This loads the
URDF, runs the same DLS reach the controller uses, and confirms two things: a
reachable target converges sub-centimetre, and an *unreachable* target (here a
3 m height) does **not** error — the arm + shared lift saturate at their joint
limits and the fingertip gets **as close as physically possible**:

```bash
source /opt/ros/jazzy/setup.bash
source /home/jerry/Downloads/m1-ros2-setup/ros2_ws/install/setup.bash
/usr/bin/python3 - <<'PY'
from m1_control.kinematics import ReachController, UrdfModel, LIFT_JOINT
import numpy as np

urdf = "/home/jerry/Downloads/m1-ros2-setup/assets/ranger_air_description/urdf/ranger_air_description.urdf"
reach = ReachController(UrdfModel.from_string(open(urdf).read()))
lift_lo, lift_hi = reach.model.joints[LIFT_JOINT].lower, reach.model.joints[LIFT_JOINT].upper

def run(target, iters=600):
    q = {}
    for _ in range(iters):
        out = reach.solve_step(q, {"left": np.array(target, float)})
        for k, v in out.items():
            if k != "_dist":
                q[k] = v
    tip = reach.fingertip("left", q)
    return tip, float(np.linalg.norm(np.array(target) - tip)), q[LIFT_JOINT]

for label, tgt in [("reachable", [0.30, 0.20, 0.95]), ("way too high", [0.30, 0.20, 3.00])]:
    tip, dist, lift = run(tgt)
    print(f"{label:13s} target={tgt}  tip={tip.round(3).tolist()}  "
          f"residual={dist*100:5.1f} cm  lift={lift:.3f} m (limits {lift_lo}..{lift_hi})")
PY
```

Expect the reachable target to settle to ~0 cm, and the 3 m target to leave a
large residual with the lift pinned at its upper limit (no crash, no NaNs) —
that is the "reach as close as the joints allow" behaviour.

**3. End-to-end over ROS (no sim required).** Launch the brain, publish a fake
`/joint_states`, send a target, and watch the command react.

```bash
# Terminal A — the brain (RViz optional):
source /opt/ros/jazzy/setup.bash
source /home/jerry/Downloads/m1-ros2-setup/ros2_ws/install/setup.bash
ros2 launch m1_bringup bringup.launch.py use_rviz:=false

# Terminal B — stand in for the robot driver (publish zeroed feedback):
source /opt/ros/jazzy/setup.bash && source .../ros2_ws/install/setup.bash
ros2 topic pub -r 30 /joint_states sensor_msgs/msg/JointState \
  '{name: [lift_joint,
           openarm_left_joint1, openarm_left_joint2, openarm_left_joint3,
           openarm_left_joint4, openarm_left_joint5, openarm_left_joint6, openarm_left_joint7,
           openarm_right_joint1, openarm_right_joint2, openarm_right_joint3,
           openarm_right_joint4, openarm_right_joint5, openarm_right_joint6, openarm_right_joint7,
           fl_steering_joint, fr_steering_joint, rr_steering_joint, rl_steering_joint,
           openarm_left_finger_joint1, openarm_left_finger_joint2,
           openarm_right_finger_joint1, openarm_right_finger_joint2],
    position: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'

# Terminal C — drive it and watch the output:
source /opt/ros/jazzy/setup.bash && source .../ros2_ws/install/setup.bash
ros2 run m1_control m1_send_pose --arm left --xyz 0.30 0.20 1.40   # high target
ros2 topic echo --once /m1/joint_command          # lift_joint position should rise
ros2 topic pub --once /m1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.4}}'
ros2 topic echo --once /m1/joint_command          # wheel velocities should be non-zero
```

A high target should drive `lift_joint` up in `/m1/joint_command`; a `/m1/cmd_vel`
should produce non-zero wheel velocities + steer angles. Because the target
height is unbounded, sending a target above the robot's reach is safe — the
command simply saturates the arm + lift instead of being clamped or rejected.

**4. Full simulation.** Run the two-terminal flow under
[Run (simulation)](#run-simulation), then use the web panel or `m1_teleop`. The
web panel's distance-to-target readout shrinking toward 0 (and the RViz error
line collapsing) confirms the reach is working; an out-of-reach height leaves a
steady residual with the lift parked at its limit.

> Note on physical limits: the height clamp on the *target* was removed so you
> can aim anywhere vertically, but every joint is still clamped to its URDF
> `[lower, upper]` each control step (see `kinematics.py` `solve_step`), so the
> robot never commands past its mechanical limits.

## Tuning

- IK behaviour (adaptive damping, convergence tolerance, per-tick command step,
  null-space posture, restart search): top of
  `m1_control/m1_control/kinematics.py`.
- Swerve geometry / wheel + steer direction sign fixups:
  `m1_control/m1_control/swerve.py`.
- Sim drive gains (arm stiffness, wheel velocity drive):
  top of `isaac/ros_sim.py` (kept in sync with `isaac/teleop.py`).
- Control rate / topic names: `m1_control/config/m1_control.yaml`.

## Deploying to the real robot

Replace Terminal 1 (Isaac Sim) with the AgileX/OpenArm hardware driver that
publishes `/joint_states` and consumes `/m1/joint_command` (position for
steer/lift/arms/fingers, velocity for wheels). Everything in Terminal 2 stays
the same. If the real arms expose a `FollowJointTrajectory` action instead of a
raw joint command, add a small adapter node (or migrate the controller to
ros2_control); the IK/swerve math in `m1_control` is reused as-is.
```
