"""Swerve-drive kinematics for the Ranger Air base.

Maps a body-velocity command (vx forward, vy left, yaw) to per-module steering
angle + wheel spin (inverse kinematics), recovers the body velocity from the
module states (forward kinematics), and dead-reckons a base pose by integrating
the body velocity (odometry). Ported from isaac/teleop.py so the sim and the
real base share the same kinematics.

Three pieces, all in the base frame (x forward, y left, +yaw = CCW):

* :func:`module_states` -- the pure inverse map (vx, vy, yaw) -> per-module
  (heading, speed). Stateless; this is the math, with no smoothing or sign
  fixups layered on.
* :class:`SwerveSolver` -- the stateful command generator the controller calls
  each tick: it adds heading low-pass smoothing, the <=90 deg "flip and reverse"
  optimisation, wheel-speed **desaturation** (so a command that would saturate a
  module is scaled down uniformly, preserving the *direction* of travel instead
  of distorting it), and the per-joint direction fixups.
* :func:`forward_kinematics` / :class:`SwerveOdometry` -- the inverse direction:
  recover (vx, vy, yaw) from the module states (least-squares over the four
  modules), and integrate a body-velocity command into an (x, y, theta) pose
  with the exact SE(2) arc update (so a sustained turn-while-driving traces the
  true arc, not a polygon). The Quest viz uses this to actually drive the robot
  model through the room as you command the base.
"""

from __future__ import annotations

import math

import numpy as np

WHEEL_RADIUS = 0.055          # effective rolling radius (m): m/s -> rad/s
STEER_SMOOTH_TAU = 0.10       # low-pass time constant on module heading (s)
# Cap on a single module's wheel spin (rad/s). A command whose fastest module
# would exceed this is scaled down uniformly (desaturation) so the robot still
# drives in the commanded direction, just slower -- the standard swerve fix for
# "translate + spin asks one corner for more speed than it has". ~1.65 m/s at
# the 0.055 m rolling radius, comfortably above normal teleop (<~0.9 m/s/module)
# so it only ever clamps a pathological command.
MAX_WHEEL_SPEED = 30.0

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

# Corner order matching WHEEL_JOINTS / STEER_JOINTS, for the array helpers.
CORNERS = [jn.split("_")[0] for jn in WHEEL_JOINTS]


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def module_states(vx: float, vy: float, yaw: float):
    """Pure inverse kinematics: (vx, vy, yaw) -> per-module (heading, speed).

    Returns a list in :data:`CORNERS` order of ``(heading_rad, speed_mps)`` where
    ``heading`` is the module's drive direction in the base frame and ``speed``
    is the contact-point ground speed. This is the textbook swerve map with no
    smoothing, flips, or sign fixups -- :class:`SwerveSolver` layers those on,
    and :func:`forward_kinematics` inverts exactly this relation.
    """
    out = []
    for corner in CORNERS:
        mx, my = MODULE_XY[corner]
        # v_module = v_body + omega x r  ->  (vx - yaw*my, vy + yaw*mx)
        vxi = vx - yaw * my
        vyi = vy + yaw * mx
        speed = math.hypot(vxi, vyi)
        heading = math.atan2(vyi, vxi) if speed > 1e-12 else 0.0
        out.append((heading, speed))
    return out


def forward_kinematics(headings, speeds):
    """Forward kinematics: per-module (heading, speed) -> (vx, vy, yaw).

    The inverse of :func:`module_states`. Each module contributes the two
    equations ``vx - yaw*my = speed*cos(h)`` and ``vy + yaw*mx = speed*sin(h)``;
    with four modules that is an over-determined 8x3 system, solved in the
    least-squares sense so a slightly inconsistent set of measured module states
    (encoder noise, a module mid-slew) still yields the best-fit body twist. The
    exact-IK states round-trip to the original command to machine precision.
    """
    rows = []
    rhs = []
    for k, corner in enumerate(CORNERS):
        mx, my = MODULE_XY[corner]
        h, s = float(headings[k]), float(speeds[k])
        vxi, vyi = s * math.cos(h), s * math.sin(h)
        rows.append([1.0, 0.0, -my]); rhs.append(vxi)   # vx - yaw*my = vxi
        rows.append([0.0, 1.0, mx]);  rhs.append(vyi)   # vy + yaw*mx = vyi
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return float(sol[0]), float(sol[1]), float(sol[2])


def desaturate(speeds, max_speed: float = MAX_WHEEL_SPEED):
    """Scale a set of wheel spins so the fastest is within ``max_speed``.

    Returns ``(scaled_speeds, scale)`` where ``scale<=1``. Scaling every module
    by the same factor keeps the steering angles (hence the direction of travel)
    untouched and just slows the whole base -- which is what you want when a
    combined translate+rotate command asks one corner for more than it can give.
    """
    peak = max((abs(s) for s in speeds), default=0.0)
    if max_speed and peak > max_speed:
        scale = max_speed / peak
        return [s * scale for s in speeds], scale
    return list(speeds), 1.0


class SwerveSolver:
    """Holds the per-module heading state and turns (vx, vy, yaw) into commands."""

    def __init__(self, max_wheel_speed: float = MAX_WHEEL_SPEED):
        self.wheel_head = [0.0, 0.0, 0.0, 0.0]
        self.max_wheel_speed = max_wheel_speed

    def solve(self, vx: float, vy: float, yaw: float, dt: float):
        """Return (steer_targets, wheel_velocities) keyed by joint name."""
        steer_blend = min(1.0, dt / STEER_SMOOTH_TAU) if dt > 0 else 1.0
        states = module_states(vx, vy, yaw)

        # First resolve each module's applied heading + signed spin (with the
        # <=90 deg flip optimisation), then desaturate the spins together so a
        # saturated command keeps its direction.
        spins = [0.0, 0.0, 0.0, 0.0]
        for k in range(len(WHEEL_JOINTS)):
            heading, speed = states[k]
            applied = self.wheel_head[k]
            if speed < 1e-4:
                spins[k] = 0.0
                continue
            spin = speed / WHEEL_RADIUS
            diff = wrap_to_pi(heading - applied)
            if abs(diff) > math.pi / 2.0:
                heading = wrap_to_pi(heading + math.pi)
                diff = wrap_to_pi(heading - applied)
                spin = -spin
            applied = applied + diff * steer_blend
            # Re-wrap on store: under continuous same-sense rotation ``applied``
            # would otherwise grow without bound (it is emitted as a steering
            # joint POSITION command). The flip-and-reverse logic recomputes
            # ``diff = wrap_to_pi(heading - applied)``, so a wrapped ``applied``
            # yields the same diff -- the wrap is behaviour-preserving.
            self.wheel_head[k] = wrap_to_pi(applied)
            spins[k] = spin

        spins, _scale = desaturate(spins, self.max_wheel_speed)

        steer_targets = {}
        wheel_vel = {}
        for k, jn in enumerate(WHEEL_JOINTS):
            steer_targets[STEER_JOINTS[k]] = STEER_DIR[STEER_JOINTS[k]] * self.wheel_head[k]
            wheel_vel[jn] = WHEEL_DIR[jn] * spins[k]
        return steer_targets, wheel_vel


class SwerveOdometry:
    """Dead-reckons a base pose by integrating the commanded body velocity.

    Holds an (x, y, theta) pose in a fixed "odom" frame (x fwd, y left, +theta =
    CCW about +z). Each :meth:`update` advances it by one tick using the **exact
    SE(2) arc integration** of a constant body twist over ``dt`` -- so driving
    forward while turning traces the real circular arc rather than the chord, and
    a pure turn-in-place leaves (x, y) exactly fixed. Open-loop (it integrates the
    *command*, not encoders), which is all the Quest viz needs to move the robot
    model as the operator drives.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x, self.y, self.theta = x, y, theta

    def update(self, vx: float, vy: float, yaw: float, dt: float):
        """Integrate one tick of body velocity; return the new (x, y, theta)."""
        if dt <= 0.0:
            return self.x, self.y, self.theta
        phi = yaw * dt
        if abs(yaw) < 1e-9:
            # Straight (or pure strafe) segment: no rotation over the interval.
            dx_b = vx * dt
            dy_b = vy * dt
        else:
            # SE(2) exponential: body-frame displacement of a constant twist.
            s, c = math.sin(phi), math.cos(phi)
            dx_b = (vx * s - vy * (1.0 - c)) / yaw
            dy_b = (vx * (1.0 - c) + vy * s) / yaw
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        self.x += cos_t * dx_b - sin_t * dy_b
        self.y += sin_t * dx_b + cos_t * dy_b
        self.theta = wrap_to_pi(self.theta + phi)
        return self.x, self.y, self.theta

    @property
    def pose(self):
        return self.x, self.y, self.theta

    def quaternion(self):
        """Heading as a quaternion [x, y, z, w] (rotation about +z)."""
        half = 0.5 * self.theta
        return [0.0, 0.0, math.sin(half), math.cos(half)]
