---
name: colcon-build
description: >-
  Build the ROS 2 workspace (colcon build --symlink-install) with ROS 2 Jazzy
  sourced, and report any build errors. Use when asked to build the workspace or
  after changing package code that must be installed.
disable-model-invocation: true
---

# Build the M1 ROS 2 workspace

Source ROS 2 Jazzy first, build with symlink-install, then source the overlay.
Run from the repo root:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

There are 4 packages (`ranger_air_description`, `openarm_description`,
`m1_control`, `m1_bringup`); a clean build finishes with no errors. If it fails,
report the failing package and the first real compiler/import error (not the
downstream noise). `m1_control` and `m1_bringup` are `ament_python`; the two
description packages are `ament_cmake`.

Note: `colcon build` regenerates `ros2_ws/{build,install,log}/`, which are
gitignored — don't commit them.
