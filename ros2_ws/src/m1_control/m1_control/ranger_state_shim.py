"""Bridge: AgileX Ranger-Air feedback -> ``/joint_states`` (8 base joints).

The vendored AgileX driver (`agx_bringup`, see ``ros2_ws/src/vendor/agx_bringup``)
publishes its per-module steering angles + wheel speeds as two custom messages:

* ``/steering_angles`` -- ``agx_bringup/msg/SteeringAngles`` (``steering_01..04``,
  **radians**), and
* ``/wheel_speeds`` -- ``agx_bringup/msg/WheelSpeeds`` (``wheel_01..04``, **m/s,
  linear ground speed**).

It does **not** publish a ``sensor_msgs/JointState`` for the base joints, so
robot_state_publisher can't animate the base in the URDF / RViz / the Quest viz.
This node converts those two messages into a ``/joint_states`` message for the 8
base joints, applying the SAME sign conventions `swerve.py` uses on the command
side (`STEER_DIR` / `WHEEL_DIR`) so feedback rendered through the URDF matches
commanded motion, and converting the linear wheel speed (m/s) to the angular wheel
velocity (rad/s) the URDF wheel joints expect.

Scope: ONLY the 8 base joints (4 steering + 4 wheel). The upper body's
``/joint_states`` come from ros2_control's ``joint_state_broadcaster``; both
publish to ``/joint_states`` and the brain unions them by name (it already merges
partial ``/joint_states``), so the two together reproduce the full 27-DOF state.

The mapping is split into pure functions (:func:`reorder_corners`,
:func:`steer_wheel_to_jointstate`) so it is unit-testable with no ROS / no DDS /
no built workspace (see ``_bridge_test.py``); the message types are imported
lazily inside the node so the pure functions stay dependency-free.

HARDWARE CHECKPOINTS (cannot be verified from the driver source -- the driver
labels the modules ``01..04`` with no FL/FR/RR/RL legend):

* **Corner order** -- the ``01..04`` -> ``fl/fr/rr/rl`` assignment is a parameter
  (``corner_order``, default ``[3, 0, 1, 2]`` from the AgileX motor-ID order
  RF/RR/LR/LF). Confirm on the real robot by jogging one module at a time and
  watching which ``/joint_states`` entry moves; override the param if wrong.
* **Sign / zero** -- whether the driver's +rad steering matches our URDF steering
  ``+`` direction and zero pose is unverified; the ``STEER_DIR``/``WHEEL_DIR``
  involutions are applied as a starting point. Calibrate like the arm dir/offset.
* **Wheel radius** -- ``wheel_radius`` (default ``swerve.WHEEL_RADIUS``) converts
  m/s -> rad/s; set it to the Ranger Air's real rolling radius.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from m1_control.swerve import (
    STEER_DIR,
    STEER_JOINTS,
    WHEEL_DIR,
    WHEEL_JOINTS,
    WHEEL_RADIUS,
)

# Canonical base-joint order: the 4 steering joints then the 4 wheel joints,
# matching swerve.STEER_JOINTS + swerve.WHEEL_JOINTS
# (fl/fr/rr/rl_steering_joint, fl/fr/rr/rl_wheel_joint). Steering -> position,
# wheel -> velocity, same as the rest of the stack.
BASE_JOINTS = list(STEER_JOINTS) + list(WHEEL_JOINTS)

# Default AgileX-feedback-index -> our-corner mapping. Our corners are ordered
# (fl, fr, rr, rl) (swerve.CORNERS). The AgileX motor IDs run RF, RR, LR, LF (the
# 0x271/0x281 frames fill steering_01..04 / wheel_01..04 in that motor order), so
# the AgileX index for our fl is 3, fr is 0, rr is 1, rl is 2. i.e. our value k
# reads AgileX field ``CORNER_ORDER_DEFAULT[k]``. HARDWARE CHECKPOINT: confirm /
# override via the ``corner_order`` param (see module docstring).
CORNER_ORDER_DEFAULT = [3, 0, 1, 2]


def reorder_corners(values, order):
    """Permute a length-4 AgileX-order sequence into our (fl, fr, rr, rl) order.

    ``order`` is a length-4 sequence of AgileX indices: ``out[k] = values[order[k]]``
    so ``out`` is in our corner order. Pure; raises ``ValueError`` on a bad length
    or out-of-range index. No ROS, no DDS.
    """
    if len(values) != 4 or len(order) != 4:
        raise ValueError(
            f"expected length-4 values + order, got {len(values)} + {len(order)}")
    out = []
    for idx in order:
        if not 0 <= int(idx) < 4:
            raise ValueError(f"corner index out of range: {idx}")
        out.append(float(values[int(idx)]))
    return out


def steer_wheel_to_jointstate(steer, wheel, wheel_radius=WHEEL_RADIUS):
    """Map 4 steering angles + 4 wheel speeds to a base ``/joint_states`` triple.

    Pure: ``steer`` and ``wheel`` are length-4 sequences already in our corner
    order ``fl, fr, rr, rl`` (matching :data:`STEER_JOINTS` / :data:`WHEEL_JOINTS`
    -- apply :func:`reorder_corners` to AgileX-order feedback first). ``steer`` is
    in radians; ``wheel`` is the linear ground speed in m/s. Returns
    ``(names, positions, velocities)`` where:

    * ``names`` is :data:`BASE_JOINTS` (the 4 steering joints then the 4 wheels),
    * ``positions`` carries the steering angles (rad) in the first 4 slots (wheels
      get ``0.0`` -- a free-spinning wheel has no meaningful position),
    * ``velocities`` carries the wheel **angular** velocity (rad/s = m/s /
      ``wheel_radius``) in the last 4 slots (steering joints get ``0.0``),

    with the SAME per-joint sign fixups `swerve.py` applies on the command side
    (`STEER_DIR` / `WHEEL_DIR`) so feedback rendered through the URDF matches
    commanded motion. Because those direction maps are involutions
    (``dir * dir == 1``), applying them here exactly inverts the command-side sign
    flip, recovering the URDF-frame joint value from the driver's body-frame
    feedback.

    Raises ``ValueError`` if either input is not length 4 or ``wheel_radius<=0``.
    No ROS, no DDS.
    """
    if len(steer) != 4 or len(wheel) != 4:
        raise ValueError(
            f"expected 4 steering + 4 wheel values, got "
            f"{len(steer)} + {len(wheel)}")
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be > 0, got {wheel_radius}")

    positions = [0.0] * len(BASE_JOINTS)
    velocities = [0.0] * len(BASE_JOINTS)
    for k, jn in enumerate(STEER_JOINTS):
        positions[k] = STEER_DIR[jn] * float(steer[k])
    for k, jn in enumerate(WHEEL_JOINTS):
        velocities[len(STEER_JOINTS) + k] = (
            WHEEL_DIR[jn] * float(wheel[k]) / float(wheel_radius))
    return list(BASE_JOINTS), positions, velocities


class RangerStateShim(Node):
    """AgileX steering/wheel feedback -> ``/joint_states`` for the 8 base joints."""

    def __init__(self):
        super().__init__("m1_ranger_shim")

        # AgileX driver feedback topics (absolute names from agx_bringup; no remap
        # needed). Override via params if a future driver renames them.
        self.declare_parameter("steer_topic", "/steering_angles")
        self.declare_parameter("wheel_topic", "/wheel_speeds")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("publish_rate", 50.0)
        # HARDWARE CHECKPOINTS (see module docstring): corner order + wheel radius.
        self.declare_parameter("corner_order", CORNER_ORDER_DEFAULT)
        self.declare_parameter("wheel_radius", float(WHEEL_RADIUS))

        steer_topic = self.get_parameter("steer_topic").value
        wheel_topic = self.get_parameter("wheel_topic").value
        js_topic = self.get_parameter("joint_states_topic").value
        rate = float(self.get_parameter("publish_rate").value)
        self.corner_order = [int(i) for i in
                             self.get_parameter("corner_order").value]
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)

        # Lazy import: the message types live in the vendored agx_bringup package,
        # so the pure functions above stay importable without a built workspace.
        from agx_bringup.msg import SteeringAngles, WheelSpeeds

        self._steer = [0.0, 0.0, 0.0, 0.0]
        self._wheel = [0.0, 0.0, 0.0, 0.0]
        self._have_steer = False
        self._have_wheel = False

        self.js_pub = self.create_publisher(JointState, js_topic, 10)
        self.create_subscription(SteeringAngles, steer_topic, self._on_steer, 10)
        self.create_subscription(WheelSpeeds, wheel_topic, self._on_wheel, 10)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"ranger shim up: {steer_topic} (SteeringAngles) + {wheel_topic} "
            f"(WheelSpeeds) -> {js_topic} ({len(BASE_JOINTS)} base joints); "
            f"corner_order={self.corner_order}, wheel_radius={self.wheel_radius:.4f}")

    def _on_steer(self, msg):
        self._steer = [
            float(msg.steering_01), float(msg.steering_02),
            float(msg.steering_03), float(msg.steering_04),
        ]
        self._have_steer = True

    def _on_wheel(self, msg):
        self._wheel = [
            float(msg.wheel_01), float(msg.wheel_02),
            float(msg.wheel_03), float(msg.wheel_04),
        ]
        self._have_wheel = True

    def _tick(self):
        # Don't publish phantom zeros before the driver has spoken.
        if not (self._have_steer or self._have_wheel):
            return
        steer = reorder_corners(self._steer, self.corner_order)
        wheel = reorder_corners(self._wheel, self.corner_order)
        names, positions, velocities = steer_wheel_to_jointstate(
            steer, wheel, self.wheel_radius)
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
