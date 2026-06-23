"""Bring up MoveIt 2 move_group for M1 PLANNED, collision-aware moves.

This is the OPTIONAL Phase-3 planning path. It is NOT the reactive teleop path --
that stays on the Drake position-only IK brain (m1_control). MoveIt here adds
planned collision-aware point-to-point motion, executed via the per-arm
JointTrajectoryControllers (left_arm_jtc / right_arm_jtc), hot-swapped in over the
reactive arm_position_controller.

What this launch does:
  * assembles a MoveIt config from m1_bringup's files:
      - robot_description  := urdf/m1_hardware.urdf.xacro  (use_mock:=true by default)
      - SRDF               := moveit/m1.srdf
      - kinematics         := moveit/kinematics.yaml   (KDL per arm)
      - OMPL               := moveit/ompl_planning.yaml
      - controllers        := moveit/moveit_controllers.yaml (-> the JTCs)
  * starts move_group with that config.
  * (use_ros2_control:=true, default) ALSO includes hardware.launch.py so the
    mock ros2_control stack (controller_manager + RSP + JTCs + joint_state_broadcaster)
    is up to execute a plan. Set use_ros2_control:=false if you launch
    hardware.launch.py yourself / already have a controller_manager up.
  * (use_rviz:=true) starts RViz with the MoveIt MotionPlanning plugin.

Args:
  use_mock:=true|false         mock_components (default) vs the real Damiao plugin
                               (forwarded to the m1_hardware URDF xacro)
  use_ros2_control:=true|false also bring up hardware.launch.py (default true)
  use_rviz:=true|false         start RViz MotionPlanning (default false)

Interpreter note: this is a launch file (run via `ros2 launch`), so it uses the
sourced ROS 2 Jazzy Python -- not the standalone /usr/bin/python3 rule for
solver scripts.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def _setup(context, *args, **kwargs):
    bringup_share = get_package_share_directory("m1_bringup")
    use_mock = LaunchConfiguration("use_mock").perform(context)
    use_rviz = LaunchConfiguration("use_rviz")
    use_ros2_control = LaunchConfiguration("use_ros2_control")

    # Assemble the MoveIt config. The package root is m1_bringup; we point at our
    # files explicitly (URDF in urdf/, the rest under moveit/). The robot_description
    # is the SAME m1_hardware.urdf.xacro the reactive stack uses, expanded with the
    # use_mock mapping so MoveIt collision-checks the exact robot ros2_control drives.
    moveit_config = (
        MoveItConfigsBuilder("m1", package_name="m1_bringup")
        .robot_description(
            file_path="urdf/m1_hardware.urdf.xacro",
            mappings={"use_mock": use_mock},
        )
        .robot_description_semantic(file_path="moveit/m1.srdf")
        .robot_description_kinematics(file_path="moveit/kinematics.yaml")
        .joint_limits(file_path="moveit/joint_limits.yaml")
        .trajectory_execution(file_path="moveit/moveit_controllers.yaml")
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"],
        )
        .to_moveit_configs()
    )

    # The OMPL pipeline yaml lives under moveit/, not the default config/ dir, so
    # the builder's planning_pipelines() won't auto-find it; load it explicitly
    # onto move_group as the `ompl` pipeline params.
    ompl_yaml = os.path.join(bringup_share, "moveit", "ompl_planning.yaml")
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_robot_description_semantic": True},
            {"ompl": _load_yaml(ompl_yaml)},
        ],
    )

    rviz_config = os.path.join(bringup_share, "rviz", "m1.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Reuse the mock ros2_control stack (controller_manager + RSP + JTCs) so a plan
    # can actually execute. hardware.launch.py already loads left_arm_jtc /
    # right_arm_jtc (inactive) which MoveIt activates on demand.
    hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "hardware.launch.py")
        ),
        launch_arguments={
            "use_mock": use_mock,
            "use_rviz": "false",
        }.items(),
        condition=IfCondition(use_ros2_control),
    )

    return [move_group_node, rviz_node, hardware_launch]


def _load_yaml(path):
    import yaml

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_mock", default_value="true"),
            DeclareLaunchArgument("use_ros2_control", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            OpaqueFunction(function=_setup),
        ]
    )
