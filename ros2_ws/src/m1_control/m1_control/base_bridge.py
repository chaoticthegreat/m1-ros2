"""Bridge: ``/m1/cmd_vel`` (Twist) -> AgileX stock-Ranger body Twist + mode.

The brain (`controller_node`) and operators speak a single holonomic-ish base
command on ``/m1/cmd_vel`` (vx forward, vy left, yaw). In sim that is resolved
per-module by `swerve.py`. On the **stock AgileX Ranger / Ranger Air** chassis,
the firmware does NOT accept per-module swerve commands -- it accepts a body
``Twist`` plus a discrete *motion mode* (DualAckermann / Parallel / Spinning /
Park / SideSlip), and it is **mode-switched, not free-holonomic**: it never
services vx + vy + yaw at once (strafe ignores yaw; spin forces linear to 0).
See memory ``agilex-ranger-no-per-module-cmd``.

So this bridge collapses the holonomic ``/m1/cmd_vel`` intent into the one mode
the firmware can honour right now, and republishes it as a body ``Twist`` on
``/cmd_vel`` (the AgileX driver's topic) plus the selected mode on
``/m1/base/motion_mode`` (Int8) for the driver to apply.

The mode-selection logic is a pure function (:func:`select_motion_mode`) so it is
unit-testable with no ROS / no DDS (see ``_bridge_test.py``). Like the operator
nodes the base command is dead-man'd (`BASE_HOLD`): if ``/m1/cmd_vel`` stops
refreshing, the bridge zeroes the base so a lost upstream never leaves it
coasting.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Int8

# Below these magnitudes a component is treated as zero (noise / round-off).
LIN_EPS = 1e-3        # m/s -- linear "is moving" threshold
YAW_EPS = 1e-3        # rad/s -- angular "is turning" threshold

# Seconds without a fresh /m1/cmd_vel before the base is zeroed (dead-man).
# Mirrors web_node.BASE_HOLD so a dropped upstream stops the base.
BASE_HOLD = 0.5

# Stock AgileX SetMotionMode byte values (DualAckermann / Parallel / Spinning).
# Published on /m1/base/motion_mode as an Int8 for the driver to translate into a
# ugv_sdk SetMotionMode call. Values follow the AgileX Ranger convention
# (0 = Ackermann, 1 = Parallel/strafe, 2 = Spinning-in-place); the driver should
# map these explicitly -- documented here, TODO-confirm against ranger_ros2.
MODE_DUAL_ACKERMANN = 0
MODE_PARALLEL = 1
MODE_SPINNING = 2

_MODE_BYTE = {
    "DUAL_ACKERMANN": MODE_DUAL_ACKERMANN,
    "PARALLEL": MODE_PARALLEL,
    "SPINNING": MODE_SPINNING,
}


def select_motion_mode(vx, vy, yaw, lin_eps=LIN_EPS, yaw_eps=YAW_EPS):
    """Collapse a holonomic (vx, vy, yaw) intent to one stock-Ranger mode.

    Pure: returns ``(mode, lin_x, lin_y, ang_z)`` where ``mode`` is one of
    ``"PARALLEL"`` / ``"SPINNING"`` / ``"DUAL_ACKERMANN"`` and the three floats
    are the body Twist to actually send in that mode. The stock firmware is
    mode-switched, so each mode keeps only the components it can service:

    * **PARALLEL** -- any significant strafe (``|vy| > lin_eps``): the base
      translates along (vx, vy) with all modules parallel; yaw is **forced to 0**
      (the mode cannot turn while strafing).
    * **SPINNING** -- only yaw (linear is negligible and ``|yaw| > yaw_eps``):
      spin in place; **linear is forced to 0**.
    * **DUAL_ACKERMANN** -- otherwise (driving, optionally with a turn): pass
      ``vx`` and ``yaw``; ``vy`` (lateral) is **forced to 0** -- a car-like base
      cannot strafe in this mode.

    No ROS, no DDS -- unit-testable in isolation.
    """
    if abs(vy) > lin_eps:
        # Significant lateral motion -> strafe; firmware ignores yaw here.
        return "PARALLEL", float(vx), float(vy), 0.0
    if abs(vx) <= lin_eps and abs(yaw) > yaw_eps:
        # Pure rotation -> spin in place; firmware forces linear to 0.
        return "SPINNING", 0.0, 0.0, float(yaw)
    # Drive (possibly turning) -> Ackermann; no lateral component.
    return "DUAL_ACKERMANN", float(vx), 0.0, float(yaw)


class BaseBridge(Node):
    """``/m1/cmd_vel`` -> ``/cmd_vel`` (body Twist) + ``/m1/base/motion_mode``."""

    def __init__(self):
        super().__init__("m1_base_bridge")

        self.declare_parameter("in_topic", "/m1/cmd_vel")
        self.declare_parameter("out_topic", "/cmd_vel")
        self.declare_parameter("mode_topic", "/m1/base/motion_mode")
        self.declare_parameter("publish_rate", 60.0)
        self.declare_parameter("base_hold", BASE_HOLD)

        in_topic = self.get_parameter("in_topic").value
        out_topic = self.get_parameter("out_topic").value
        mode_topic = self.get_parameter("mode_topic").value
        rate = float(self.get_parameter("publish_rate").value)
        self.base_hold = float(self.get_parameter("base_hold").value)

        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self._last_cmd = 0.0

        self.twist_pub = self.create_publisher(Twist, out_topic, 10)
        self.mode_pub = self.create_publisher(Int8, mode_topic, 10)
        self.create_subscription(Twist, in_topic, self._on_cmd_vel, 10)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"base bridge up: {in_topic} -> {out_topic} (Twist) + "
            f"{mode_topic} (Int8 mode); BASE_HOLD={self.base_hold:.2f}s")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd_vel(self, msg: Twist):
        self.cmd_vel = {
            "vx": float(msg.linear.x),
            "vy": float(msg.linear.y),
            "yaw": float(msg.angular.z),
        }
        self._last_cmd = self._now()

    def _tick(self):
        # Dead-man: if the upstream stopped refreshing, stop the base.
        if self._now() - self._last_cmd > self.base_hold:
            self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}

        mode, lin_x, lin_y, ang_z = select_motion_mode(
            self.cmd_vel["vx"], self.cmd_vel["vy"], self.cmd_vel["yaw"])

        tw = Twist()
        tw.linear.x = lin_x
        tw.linear.y = lin_y
        tw.angular.z = ang_z
        self.twist_pub.publish(tw)
        self.mode_pub.publish(Int8(data=_MODE_BYTE[mode]))


def main(args=None):
    rclpy.init(args=args)
    node = BaseBridge()
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
