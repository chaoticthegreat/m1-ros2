"""Bridge: ``/m1/cmd_vel`` (Twist) -> AgileX Ranger-Air driver body ``Twist``.

The brain (`controller_node`) and operators speak a single holonomic-ish base
command on ``/m1/cmd_vel`` (vx forward, vy left, yaw). In sim that is resolved
per-module by `swerve.py`. On the **real AgileX Ranger Air** chassis the base is
driven by the vendored AgileX driver (`agx_bringup`, see
``ros2_ws/src/vendor/agx_bringup``), which subscribes a plain
``geometry_msgs/Twist`` (its ``/sub_cmd_vel``, remapped to ``/cmd_vel``) and
**auto-selects the motion mode itself**: inside its CAN handler it inspects the
Twist and emits the enable (``0x421``) + mode (``0x141``) + motion (``0x111``)
frames, choosing PARALLEL when ``linear.y != 0``, SPINNING when the turn radius
``|vx/yaw| < 0.5 m``, else DUAL_ACKERMANN. So there is **no separate motion-mode
topic/service** on this driver -- the mode is set purely by the *shape* of the
Twist (memory ``agilex-ranger-no-per-module-cmd``: the chassis is mode-switched,
never blending strafe + yaw; `swerve.py`'s per-module output cannot drive it).

This bridge therefore:

* collapses the holonomic ``/m1/cmd_vel`` intent to a single mode's components
  via :func:`select_motion_mode` (zeroing the cross-mode components so we never
  ask the firmware to blend strafe + yaw -- which it silently drops), and
* republishes the result as a body ``Twist`` on ``/cmd_vel`` (the driver's input)
  at a fixed rate so the driver keeps a live command stream.

The driver does the actual mode switching from that Twist; the chosen mode is
logged here (and is independently readable on the driver's ``/motion_mode_feedback``)
but is **not** published as a command -- nothing consumes it.

The mode-selection logic is a pure function (:func:`select_motion_mode`) so it is
unit-testable with no ROS / no DDS (see ``_bridge_test.py``). Like the operator
nodes the base command is dead-man'd (`BASE_HOLD`): if ``/m1/cmd_vel`` stops
refreshing, the bridge zeroes the base so a lost upstream never leaves it coasting
(the driver has no command timeout of its own).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

# Below these magnitudes a component is treated as zero (noise / round-off).
LIN_EPS = 1e-3        # m/s -- linear "is moving" threshold
YAW_EPS = 1e-3        # rad/s -- angular "is turning" threshold

# Seconds without a fresh /m1/cmd_vel before the base is zeroed (dead-man).
# Mirrors web_node.BASE_HOLD so a dropped upstream stops the base.
BASE_HOLD = 0.5


def select_motion_mode(vx, vy, yaw, lin_eps=LIN_EPS, yaw_eps=YAW_EPS):
    """Collapse a holonomic (vx, vy, yaw) intent to one stock-Ranger mode.

    Pure: returns ``(mode, lin_x, lin_y, ang_z)`` where ``mode`` is one of
    ``"PARALLEL"`` / ``"SPINNING"`` / ``"DUAL_ACKERMANN"`` and the three floats
    are the body Twist to actually send in that mode. The chassis is mode-switched,
    so each mode keeps only the components it can service and **zeroes** the rest,
    so we never hand the driver a blended command it will silently drop:

    * **PARALLEL** -- any significant strafe (``|vy| > lin_eps``): the base
      translates along (vx, vy) with all modules parallel; yaw is **forced to 0**
      (the mode cannot turn while strafing).
    * **SPINNING** -- only yaw (linear is negligible and ``|yaw| > yaw_eps``):
      spin in place; **linear is forced to 0**.
    * **DUAL_ACKERMANN** -- otherwise (driving, optionally with a turn): pass
      ``vx`` and ``yaw``; ``vy`` (lateral) is **forced to 0** -- a car-like base
      cannot strafe in this mode.

    The vendored AgileX driver re-derives the mode from this (single-intent) Twist
    and is the authority on what the chassis can physically do (e.g. it may switch
    a tight-radius DUAL_ACKERMANN to SPINNING). Sending a clean single-intent Twist
    makes that derivation unambiguous.

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
    """``/m1/cmd_vel`` -> ``/cmd_vel`` (single-intent body Twist for AgileX)."""

    def __init__(self):
        super().__init__("m1_base_bridge")

        self.declare_parameter("in_topic", "/m1/cmd_vel")
        self.declare_parameter("out_topic", "/cmd_vel")
        self.declare_parameter("publish_rate", 60.0)
        self.declare_parameter("base_hold", BASE_HOLD)

        in_topic = self.get_parameter("in_topic").value
        out_topic = self.get_parameter("out_topic").value
        rate = float(self.get_parameter("publish_rate").value)
        self.base_hold = float(self.get_parameter("base_hold").value)

        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self._last_cmd = 0.0
        self._last_mode = None

        self.twist_pub = self.create_publisher(Twist, out_topic, 10)
        self.create_subscription(Twist, in_topic, self._on_cmd_vel, 10)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"base bridge up: {in_topic} -> {out_topic} (Twist) for the AgileX "
            f"driver (mode auto-selected by the driver from the Twist); "
            f"BASE_HOLD={self.base_hold:.2f}s")

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

        # Log only on mode change so the operator can see the firmware mode the
        # driver will pick, without spamming at the publish rate.
        if mode != self._last_mode:
            self.get_logger().info(f"base mode -> {mode}")
            self._last_mode = mode


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
