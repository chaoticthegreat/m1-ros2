"""Swerve-drive kinematics for the Ranger Air base.

Maps a body-velocity command (vx forward, vy left, yaw) to per-module steering
angle + wheel spin. Ported directly from isaac/teleop.py so the sim and the
real base share the same kinematics.
"""

from __future__ import annotations

import math

WHEEL_RADIUS = 0.055          # effective rolling radius (m): m/s -> rad/s
STEER_SMOOTH_TAU = 0.10       # low-pass time constant on module heading (s)

WHEEL_JOINTS = ["fl_wheel_joint", "fr_wheel_joint", "rr_wheel_joint", "rl_wheel_joint"]
STEER_JOINTS = ["fl_steering_joint", "fr_steering_joint", "rr_steering_joint", "rl_steering_joint"]

# Direction fixups so a positive command moves every module the same physical
# way (URDF axes are not all aligned). Flip a sign if a wheel/corner is wrong.
WHEEL_DIR = {"fl_wheel_joint": 1.0, "fr_wheel_joint": 1.0, "rr_wheel_joint": 1.0, "rl_wheel_joint": -1.0}
STEER_DIR = {"fl_steering_joint": 1.0, "fr_steering_joint": 1.0, "rr_steering_joint": -1.0, "rl_steering_joint": 1.0}

# Module positions in the base frame (x forward, y left), metres.
MODULE_XY = {
    "fl": (0.194, 0.169),
    "fr": (0.194, -0.169),
    "rr": (-0.194, -0.169),
    "rl": (-0.194, 0.169),
}


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SwerveSolver:
    """Holds the per-module heading state and turns (vx, vy, yaw) into commands."""

    def __init__(self):
        self.wheel_head = [0.0, 0.0, 0.0, 0.0]

    def solve(self, vx: float, vy: float, yaw: float, dt: float):
        """Return (steer_targets, wheel_velocities) keyed by joint name."""
        steer_blend = min(1.0, dt / STEER_SMOOTH_TAU) if dt > 0 else 1.0
        steer_targets = {}
        wheel_vel = {}
        for k, jn in enumerate(WHEEL_JOINTS):
            corner = jn.split("_")[0]
            mx, my = MODULE_XY[corner]
            vxi = vx - yaw * my
            vyi = vy + yaw * mx
            speed = math.hypot(vxi, vyi)
            applied = self.wheel_head[k]
            if speed < 1e-4:
                spin = 0.0
            else:
                heading = math.atan2(vyi, vxi)
                spin = speed / WHEEL_RADIUS
                diff = wrap_to_pi(heading - applied)
                if abs(diff) > math.pi / 2.0:
                    heading = wrap_to_pi(heading + math.pi)
                    diff = wrap_to_pi(heading - applied)
                    spin = -spin
                applied = applied + diff * steer_blend
            self.wheel_head[k] = applied
            steer_targets[STEER_JOINTS[k]] = STEER_DIR[STEER_JOINTS[k]] * applied
            wheel_vel[jn] = WHEEL_DIR[jn] * spin
        return steer_targets, wheel_vel
