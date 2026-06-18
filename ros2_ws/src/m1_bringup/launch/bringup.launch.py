"""Bring up the ROS 2 side of the M1 robot.

This launches everything that runs under system ROS 2 (Jazzy):
  * robot_state_publisher  - publishes /tf from /joint_states + the URDF
  * m1_controller          - the whole-body brain (pose -> /m1/joint_command)
  * rviz2 (optional)       - visualization

The physics simulator runs separately as the Isaac Sim process
(`/home/jerry/isaac-sim/python.sh isaac/ros_sim.py`), which provides
/joint_states and /clock and consumes /m1/joint_command. On the real robot you
would replace that Isaac process with the hardware driver; this launch file is
unchanged.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory("ranger_air_description")
    default_urdf = os.path.join(desc_share, "urdf", "ranger_air_description.urdf")
    rviz_config = os.path.join(
        get_package_share_directory("m1_bringup"), "rviz", "m1.rviz")
    control_config = os.path.join(
        get_package_share_directory("m1_control"), "config", "m1_control.yaml")

    urdf_path = LaunchConfiguration("urdf_path")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    with open(default_urdf, "r") as fh:
        robot_description = fh.read()

    return LaunchDescription([
        DeclareLaunchArgument("urdf_path", default_value=default_urdf),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use /clock from Isaac Sim."),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),

        Node(
            package="m1_control",
            executable="m1_controller",
            name="m1_controller",
            output="screen",
            parameters=[
                control_config,
                {"urdf_path": urdf_path, "use_sim_time": use_sim_time},
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_rviz),
        ),
    ])
