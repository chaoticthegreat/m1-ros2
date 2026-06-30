"""Bring up the M1 robot on real hardware (or ros2_control mock).

This replaces the Isaac Sim process with the ros2_control stack:

  * robot_state_publisher  - URDF + /joint_states -> /tf
  * ros2_control_node      - controller_manager hosting the hardware (mock or the
                             real m1_hardware/M1SystemInterface Damiao plugin)
  * spawners               - joint_state_broadcaster (-> /joint_states),
                             arm_position_controller (forward position, the 17
                             commanded upper-body joints), left/right_arm_jtc (inactive,
                             for planned moves later)
  * m1_joint_bridge        - /m1/joint_command -> /arm_position_controller/commands
  * m1_controller          - the unchanged Drake brain (pose -> /m1/joint_command)
  * (use_base) base path   - the vendored AgileX driver (agx_bringup_node, the
                             Ranger-Air SocketCAN driver) + m1_base_bridge
                             (/m1/cmd_vel -> /cmd_vel Twist) + m1_ranger_shim
                             (AgileX /steering_angles + /wheel_speeds -> /joint_states)

Args:
  use_mock:=true|false   mock_components (default) vs the real Damiao plugin
  use_rviz:=true|false
  use_base:=true|false   start the AgileX base driver + bridges (off by default)
  can_interface:=can0    real-mode arm (Damiao) CAN device
  can_fd:=true|false     arm bus CAN-FD
  base_can_interface:=can1   AgileX base CAN device (separate bus from the arms,
                             classic CAN -- the base driver is not CAN-FD)
  motor_map:=<path>      real-mode motor-id -> joint map

The operator interfaces (m1_web, m1_quest, m1_teleop) and m1_hwconfig are run
separately and are unchanged -- they speak only /m1/* + /joint_states.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory("m1_bringup")
    controllers_yaml = os.path.join(bringup_share, "config", "m1_controllers.yaml")
    rviz_config = os.path.join(bringup_share, "rviz", "m1.rviz")

    use_mock = LaunchConfiguration("use_mock")
    use_rviz = LaunchConfiguration("use_rviz")
    use_base = LaunchConfiguration("use_base")
    can_interface = LaunchConfiguration("can_interface")
    can_fd = LaunchConfiguration("can_fd")
    base_can_interface = LaunchConfiguration("base_can_interface")
    motor_map = LaunchConfiguration("motor_map")

    # Expand the hardware URDF (ranger_air description + the <ros2_control> tag).
    robot_description = {
        "robot_description": ParameterValue(
            Command([
                "xacro ",
                PathJoinSubstitution([
                    FindPackageShare("m1_bringup"), "urdf", "m1_hardware.urdf.xacro"]),
                " use_mock:=", use_mock,
                " can_interface:=", can_interface,
                " can_fd:=", can_fd,
                " motor_map:=", motor_map,
            ]),
            value_type=str,
        ),
    }

    args = [
        DeclareLaunchArgument("use_mock", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("use_base", default_value="false"),
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("can_fd", default_value="true"),
        DeclareLaunchArgument("base_can_interface", default_value="can1"),
        DeclareLaunchArgument("motor_map", default_value=""),
    ]

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[robot_description, controllers_yaml],
    )

    def spawner(name, *extra):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager", *extra],
            output="screen",
        )

    jsb = spawner("joint_state_broadcaster")
    arm_pos = spawner("arm_position_controller")
    left_jtc = spawner("left_arm_jtc", "--inactive")
    right_jtc = spawner("right_arm_jtc", "--inactive")

    joint_bridge = Node(
        package="m1_control",
        executable="m1_joint_bridge",
        output="screen",
    )

    brain = Node(
        package="m1_control",
        executable="m1_controller",
        name="m1_controller",
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    # Base path (AgileX) -- off by default (use_base:=true needs the real Ranger
    # Air + its CAN bus). The vendored AgileX driver (agx_bringup, the Ranger-Air
    # SocketCAN driver) is launched here together with the two bridges. The driver
    # subscribes /sub_cmd_vel (remapped <- /cmd_vel, which m1_base_bridge feeds) and
    # publishes /steering_angles + /wheel_speeds (which m1_ranger_shim consumes).
    # Its `interface` param selects the base CAN bus (separate from the arm bus).
    # NB: get_package_share_directory is NOT used for agx_bringup so the default
    # mock launch (use_base:=false) does not require it to be built; FindPackageShare
    # is a lazy substitution resolved only when this conditioned node runs.
    agx_driver = Node(
        package="agx_bringup",
        executable="agx_bringup_node",
        name="agx_bringup_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("agx_bringup"), "config", "agx_bringup.yaml"]),
            {"interface": base_can_interface},
        ],
        remappings=[("/sub_cmd_vel", "/cmd_vel")],
        condition=IfCondition(use_base),
    )
    base_bridge = Node(
        package="m1_control",
        executable="m1_base_bridge",
        output="screen",
        condition=IfCondition(use_base),
    )
    ranger_shim = Node(
        package="m1_control",
        executable="m1_ranger_shim",
        output="screen",
        parameters=[{
            # --- AgileX base feedback calibration (HARDWARE CHECKPOINTS) ---
            # See ros2_ws/AGILEX_SETUP.md §6. Module 01..04 -> our fl/fr/rr/rl;
            # default = AgileX motor order RF,RR,LR,LF. Confirm by jogging one
            # module at a time and watching /joint_states, then edit these.
            "corner_order": [3, 0, 1, 2],
            # Ranger Air wheel rolling radius (m); converts /wheel_speeds m/s -> rad/s.
            "wheel_radius": 0.055,
        }],
        condition=IfCondition(use_base),
    )

    return LaunchDescription([
        *args,
        rsp,
        control_node,
        jsb,
        arm_pos,
        left_jtc,
        right_jtc,
        joint_bridge,
        brain,
        rviz,
        agx_driver,
        base_bridge,
        ranger_shim,
    ])
