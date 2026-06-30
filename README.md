# M1 — Isaac Sim

Import and simulate the M1 manipulator (mobile base + lift +
dual OpenArm arms) in NVIDIA Isaac Sim.

## Layout

```
assets/
  ranger_air_description/   # Ranger Air URDF and meshes (also a ROS 2 package)
  openarm_description/      # OpenArm arm and gripper meshes (also a ROS 2 package)
  usd/                      # generated USD (created by the converter)
isaac/
  convert_urdf_to_usd.py    # URDF -> USD importer
  run_sim.py                # load USD into a physics scene and simulate
  teleop.py                 # keyboard teleop (standalone, no ROS)
  ros_sim.py                # run the robot as a ROS 2 bridge node (sim "driver")
  README.md                 # detailed Isaac instructions
ros2_ws/
  src/m1_control/           # whole-body brain: Cartesian arm/lift reach + swerve
  src/m1_bringup/           # launch files + RViz
  README.md                 # ROS 2 stack instructions
```

## Two ways to drive the robot

**Standalone Isaac (no ROS):** quick keyboard teleop / demos.

```bash
/home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py   # once
/home/jerry/isaac-sim/python.sh isaac/teleop.py                # or run_sim.py --demo
```

**ROS 2 (Jazzy) — fully simulatable, deployable:** give the arms a target pose
and the arm joints + lift reach toward it; drive the base with `/m1/cmd_vel`.

```bash
# Terminal 1: simulated robot (publishes /joint_states, takes /m1/joint_command)
source /opt/ros/jazzy/setup.bash
/home/jerry/isaac-sim/python.sh isaac/ros_sim.py

# Terminal 2: build + launch the control brain
cd ros2_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install
source install/setup.bash
ros2 launch m1_bringup bringup.launch.py

# Send a reach target (metres, base_link frame):
ros2 run m1_control m1_send_pose --arm left --xyz 0.30 0.20 0.95

# Or drive interactively (same interfaces work on the real robot):
ros2 run m1_control m1_web        # browser control panel at http://localhost:8080
ros2 run m1_control m1_teleop     # or a terminal keyboard console
```

See [`ros2_ws/README.md`](ros2_ws/README.md) for the full ROS 2 interface,
architecture, and deployment notes, and [`isaac/README.md`](isaac/README.md)
for Isaac details.

## Deploying on real hardware (Jetson AGX Orin)

The same control code runs on the real robot — no Isaac. For a from-scratch
bring-up on an **NVIDIA Jetson AGX Orin**, follow
[`DEPLOY_AGX_ORIN.md`](DEPLOY_AGX_ORIN.md): a step-by-step agent runbook covering
the OS/ROS strategy (native Jazzy on JetPack 7.2 vs Jazzy-in-Docker on JetPack 6),
Drake, the Damiao CAN bus, and Jetson tuning, plus a bootstrap script and systemd
units in [`deploy/agx-orin/`](deploy/agx-orin/). The hardware control seam itself
is documented in [`ros2_ws/HARDWARE.md`](ros2_ws/HARDWARE.md).
