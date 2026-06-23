"""Bridge: AgileX per-wheel feedback -> ``/joint_states`` (8 base joints).

The stock AgileX Ranger driver publishes its per-module steering angles + wheel
speeds as feedback, but it does **not** publish a ``sensor_msgs/JointState`` for
the base joints, so robot_state_publisher can't animate the base in the URDF /
RViz / the Quest viz. This node converts the driver's 4 steering angles + 4 wheel
speeds into a ``/joint_states`` message for the 8 base joints, applying the SAME
sign conventions `swerve.py` uses on the command side (`STEER_DIR` / `WHEEL_DIR`),
so feedback rendered through the URDF matches commanded motion.

Scope: ONLY the 8 base joints (4 steering + 4 wheel). The upper body's
``/joint_states`` come from ros2_control's ``joint_state_broadcaster``; both
publish to ``/joint_states`` and the brain unions them by name (it already merges
partial ``/joint_states``), so the two together reproduce the full 27-DOF state.

The mapping is a pure function (:func:`steer_wheel_to_jointstate`) so it is
unit-testable with no ROS / no DDS (see ``_bridge_test.py``).

NB: the AgileX feedback topic names below (``/ranger/steering_angles`` /
``/ranger/wheel_speeds``) are PLACEHOLDERS -- confirm them against the real
ranger_ros2 driver and override via the node parameters if they differ.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from m1_control.swerve import (
    STEER_DIR,
    STEER_JOINTS,
    WHEEL_DIR,
    WHEEL_JOINTS,
)

# Canonical base-joint order: the 4 steering joints then the 4 wheel joints,
# matching swerve.STEER_JOINTS + swerve.WHEEL_JOINTS
# (fl/fr/rr/rl_steering_joint, fl/fr/rr/rl_wheel_joint). Steering -> position,
# wheel -> velocity, same as the rest of the stack.
BASE_JOINTS = list(STEER_JOINTS) + list(WHEEL_JOINTS)


def steer_wheel_to_jointstate(steer, wheel):
    """Map 4 steering angles + 4 wheel speeds to a base ``/joint_states`` triple.

    Pure: ``steer`` and ``wheel`` are length-4 sequences in corner order
    ``fl, fr, rr, rl`` (matching :data:`STEER_JOINTS` / :data:`WHEEL_JOINTS`).
    Returns ``(names, positions, velocities)`` where:

    * ``names`` is :data:`BASE_JOINTS` (the 4 steering joints then the 4 wheels),
    * ``positions`` carries the steering angles in the first 4 slots (wheels get
      ``0.0`` -- a free-spinning wheel has no meaningful position),
    * ``velocities`` carries the wheel speeds in the last 4 slots (steering joints
      get ``0.0``),

    with the SAME per-joint sign fixups `swerve.py` applies on the command side
    (`STEER_DIR` / `WHEEL_DIR`) so feedback rendered through the URDF matches
    commanded motion. Because those direction maps are involutions
    (``dir * dir == 1``), applying them here exactly inverts the command-side
    sign flip, recovering the URDF-frame joint value from the driver's
    body-frame feedback.

    Raises ``ValueError`` if either input is not length 4. No ROS, no DDS.
    """
    if len(steer) != 4 or len(wheel) != 4:
        raise ValueError(
            f"expected 4 steering + 4 wheel values, got "
            f"{len(steer)} + {len(wheel)}")

    positions = [0.0] * len(BASE_JOINTS)
    velocities = [0.0] * len(BASE_JOINTS)
    for k, jn in enumerate(STEER_JOINTS):
        positions[k] = STEER_DIR[jn] * float(steer[k])
    for k, jn in enumerate(WHEEL_JOINTS):
        velocities[len(STEER_JOINTS) + k] = WHEEL_DIR[jn] * float(wheel[k])
    return list(BASE_JOINTS), positions, velocities


class RangerStateShim(Node):
    """AgileX wheel feedback -> ``/joint_states`` for the 8 base joints."""

    def __init__(self):
        super().__init__("m1_ranger_shim")

        # TODO-confirm: these AgileX feedback topic names are placeholders --
        # verify against the real ranger_ros2 driver, override via params.
        self.declare_parameter("steer_topic", "/ranger/steering_angles")
        self.declare_parameter("wheel_topic", "/ranger/wheel_speeds")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("publish_rate", 50.0)

        steer_topic = self.get_parameter("steer_topic").value
        wheel_topic = self.get_parameter("wheel_topic").value
        js_topic = self.get_parameter("joint_states_topic").value
        rate = float(self.get_parameter("publish_rate").value)

        self._steer = [0.0, 0.0, 0.0, 0.0]
        self._wheel = [0.0, 0.0, 0.0, 0.0]
        self._have_steer = False
        self._have_wheel = False

        self.js_pub = self.create_publisher(JointState, js_topic, 10)
        self.create_subscription(
            Float64MultiArray, steer_topic, self._on_steer, 10)
        self.create_subscription(
            Float64MultiArray, wheel_topic, self._on_wheel, 10)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"ranger shim up: {steer_topic} + {wheel_topic} -> {js_topic} "
            f"({len(BASE_JOINTS)} base joints)")

    def _on_steer(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self._steer = [float(v) for v in msg.data[:4]]
            self._have_steer = True

    def _on_wheel(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self._wheel = [float(v) for v in msg.data[:4]]
            self._have_wheel = True

    def _tick(self):
        # Don't publish phantom zeros before the driver has spoken.
        if not (self._have_steer or self._have_wheel):
            return
        names, positions, velocities = steer_wheel_to_jointstate(
            self._steer, self._wheel)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        self.js_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RangerStateShim()
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
