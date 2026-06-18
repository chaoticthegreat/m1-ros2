"""Convenience CLI to send an arm reach target to the M1 controller.

Examples:
    ros2 run m1_control m1_send_pose --arm left  --xyz 0.4 0.25 0.7
    ros2 run m1_control m1_send_pose --arm right --xyz 0.4 -0.25 0.6

Targets are in the robot base_link frame (x forward, y left, z up).
"""

from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PoseSender(Node):
    def __init__(self, arm: str, xyz):
        super().__init__("m1_send_pose")
        topic = f"/m1/{arm}_arm/target_pose"
        self.pub = self.create_publisher(PoseStamped, topic, 10)
        self.msg = PoseStamped()
        self.msg.header.frame_id = "base_link"
        self.msg.pose.position.x = float(xyz[0])
        self.msg.pose.position.y = float(xyz[1])
        self.msg.pose.position.z = float(xyz[2])
        self.msg.pose.orientation.w = 1.0
        self.topic = topic
        self.timer = self.create_timer(0.2, self._tick)
        self._sent = 0

    def _tick(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)
        self._sent += 1
        if self._sent >= 5:  # publish a few times so it is not missed, then exit
            self.get_logger().info(f"sent target on {self.topic}")
            raise SystemExit


def main(args=None):
    parser = argparse.ArgumentParser(description="Send an M1 arm reach target.")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--xyz", nargs=3, type=float, required=True,
                        metavar=("X", "Y", "Z"))
    parsed, _ = parser.parse_known_args(args)

    rclpy.init(args=args)
    node = PoseSender(parsed.arm, parsed.xyz)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
