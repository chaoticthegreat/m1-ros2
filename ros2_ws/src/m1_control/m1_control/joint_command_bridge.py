"""Bridge: ``/m1/joint_command`` (JointState) -> arm_position_controller.

On the real robot the upper body (shared lift + dual 7-DOF arms + the two gripper
motors, 17 commanded joints) is driven through ros2_control. The
``forward_command_controller/ForwardCommandController`` configured for the
``position`` interface takes a bare ``std_msgs/Float64MultiArray`` whose entries
are the position setpoints **in the controller's joint order** (the order listed
in ``m1_controllers.yaml``). The brain (`controller_node`) still publishes the
full 27-DOF ``/m1/joint_command`` JointState exactly as it does for Isaac; this
node simply *picks* the 17 commanded upper-body positions out of it by name and
republishes them as the Float64MultiArray the controller wants. The steer / wheel
entries of ``/m1/joint_command`` are ignored here (the base is driven over its own
Twist path, `base_bridge`), and so are the two ``finger_joint2`` entries -- those
are URDF mimics auto-propagated by ros2_control from ``finger_joint1``.

In sim this bridge is NOT launched: Isaac consumes ``/m1/joint_command`` directly.

The mapping itself is a pure function (:func:`map_command`) so it is unit-testable
with no ROS / no DDS (see ``_bridge_test.py``).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# The 17 COMMANDED upper-body joints, in EXACTLY the order the
# arm_position_controller is configured with in m1_controllers.yaml. Order is
# load-bearing: the Float64MultiArray is positional, so this must match the
# controller's `joints:` list bit-for-bit. (lift, then left arm 1..7, right arm
# 1..7, then the two gripper motors left/right finger_joint1.) The two
# finger_joint2 are URDF mimics -- ros2_control auto-propagates them from
# finger_joint1, so they are NOT commanded here.
UPPER_BODY = [
    "lift_joint",
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
    "openarm_left_finger_joint1",
    "openarm_right_finger_joint1",
]


def map_command(js_name, js_pos, order):
    """Pick positions for ``order`` out of a (name, position) pair, by name.

    Pure: returns a ``list[float]`` the same length as ``order``; entry ``i`` is
    the position of ``order[i]`` taken from ``js_pos`` by matching ``order[i]``
    in ``js_name``. A name in ``order`` that is absent from ``js_name`` maps to
    ``0.0`` (so a partial command never desyncs the array length, and steer /
    wheel names in the input are simply dropped because they are not in
    ``order``). If a name appears more than once in ``js_name`` the first match
    wins.

    No ROS, no DDS -- unit-testable in isolation.
    """
    lookup = {}
    for name, pos in zip(js_name, js_pos):
        if name not in lookup:           # first occurrence wins
            lookup[name] = float(pos)
    return [lookup.get(name, 0.0) for name in order]


class JointCommandBridge(Node):
    """``/m1/joint_command`` -> ``/arm_position_controller/commands``."""

    def __init__(self):
        super().__init__("m1_joint_bridge")

        self.declare_parameter("command_topic", "/m1/joint_command")
        self.declare_parameter(
            "controller_command_topic", "/arm_position_controller/commands")

        in_topic = self.get_parameter("command_topic").value
        out_topic = self.get_parameter("controller_command_topic").value

        self.pub = self.create_publisher(Float64MultiArray, out_topic, 10)
        self.create_subscription(JointState, in_topic, self._on_command, 10)

        self.get_logger().info(
            f"joint bridge up: {in_topic} (JointState) -> "
            f"{out_topic} (Float64MultiArray, {len(UPPER_BODY)} joints)")

    def _on_command(self, msg: JointState):
        out = Float64MultiArray()
        out.data = map_command(list(msg.name), list(msg.position), UPPER_BODY)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
