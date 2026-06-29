#!/usr/bin/env /usr/bin/python3
"""Non-invasive reach-failure logger for the LIVE stack (Isaac + brain + quest).

Subscribes to the brain's published markers + /joint_states (NO restart of any
node, no extra solving) and records where the arm settles short of its target so
we can find which positions the solver fails to converge on and why.

  /m1/target_markers : per arm, ns "{arm}_target" id0 = commanded target point,
                       ns "{arm}_fingertip" id2 = FK of the MEASURED joints
                       (what the arm actually achieved in sim). Their distance is
                       the real achieved reach error -- exactly what the operator
                       sees turn amber/red.
  /joint_states      : the measured joint config, logged at each failure so we can
                       see joint-limit saturation / shared-lift compromise.

Outputs (under /tmp/m1_ros_log/):
  reach_samples.csv         every ~12 Hz: t, arm, target xyz, tip xyz, err_mm,
                            dual (both arms active), lift, + all arm joints.
  reach_failures.jsonl      one record per SETTLED-failure episode: the target has
                            been held stable for >SETTLE_S and err stays >FAIL_MM.
                            Carries the full joint config + per-joint range
                            fraction (so saturated joints / a pinned lift are
                            obvious) + whether the other arm was also commanding.

Run (its own process, alongside the live stack):
  source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
  /usr/bin/python3 _solver_failure_logger.py
Stop with Ctrl-C (or SIGINT to its PID); it flushes on every write.
"""
import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray

from m1_control.kinematics import UrdfModel, ARM_JOINTS, LIFT_JOINT

OUT_DIR = os.environ.get("ROS_LOG_DIR", "/tmp/m1_ros_log")
SAMPLE_HZ = 12.0          # full-trace sampling rate (CSV)
FAIL_MM = 20.0            # error above this (mm) = "not reaching" (amber+ on the HUD)
SETTLE_S = 1.2           # target must be stable this long before we call it settled
MOVE_EPS = 0.02          # target move (m) that resets a settle episode
SAT_FRAC = 0.03          # |range fraction| within this of 0 or 1 = joint at its limit


def _find_urdf():
    for cand in (
        "ros2_ws/install/ranger_air_description/share/ranger_air_description/"
        "urdf/ranger_air_description.urdf",
        "assets/ranger_air_description/urdf/ranger_air_description.urdf",
    ):
        if os.path.isfile(cand):
            return cand
    return None


class ReachLogger(Node):
    def __init__(self):
        super().__init__("m1_reach_failure_logger")
        os.makedirs(OUT_DIR, exist_ok=True)
        self.csv_path = os.path.join(OUT_DIR, "reach_samples.csv")
        self.jsonl_path = os.path.join(OUT_DIR, "reach_failures.jsonl")

        urdf = _find_urdf()
        self.model = UrdfModel.from_string(open(urdf).read()) if urdf else None
        # Per-joint (lower, upper) for the commanded arm joints + lift.
        self.limits = {}
        if self.model is not None:
            for j in ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]:
                jt = self.model.joints.get(j)
                if jt is not None:
                    self.limits[j] = (float(jt.lower), float(jt.upper))
        self.get_logger().info(
            f"reach logger up (urdf={'yes' if self.model else 'NO'}); "
            f"samples -> {self.csv_path}, failures -> {self.jsonl_path}")

        # latest measured joints, and latest target/tip per arm (base_link frame)
        self.q = {}
        self.target = {"left": None, "right": None}
        self.tip = {"left": None, "right": None}
        # settle-episode bookkeeping per arm
        self._ep_target = {"left": None, "right": None}     # the stable target
        self._ep_since = {"left": 0.0, "right": 0.0}        # when it became stable
        self._ep_logged = {"left": False, "right": False}   # already recorded?
        self._last_sample = 0.0
        self._n_fail = 0

        # CSV header
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            cols = (["wall_t", "arm", "tx", "ty", "tz", "fx", "fy", "fz",
                     "err_mm", "dual", "lift"]
                    + [f"q[{j}]" for j in ARM_JOINTS["left"]]
                    + [f"q[{j}]" for j in ARM_JOINTS["right"]])
            with open(self.csv_path, "a") as fh:
                fh.write(",".join(cols) + "\n")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(JointState, "/joint_states", self._on_js, qos)
        self.create_subscription(MarkerArray, "/m1/target_markers",
                                 self._on_markers, qos)

    def _on_js(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.q[n] = float(p)

    def _on_markers(self, msg: MarkerArray):
        # Refresh target (id0 add) + measured fingertip (id2 add) per arm.
        for m in msg.markers:
            arm = "left" if m.ns.startswith("left") else (
                "right" if m.ns.startswith("right") else None)
            if arm is None:
                continue
            add = (m.action == 0)  # Marker.ADD
            p = m.pose.position
            if m.ns.endswith("_target") and m.id == 0:
                self.target[arm] = [p.x, p.y, p.z] if add else None
            elif m.ns.endswith("_fingertip") and m.id == 2:
                self.tip[arm] = [p.x, p.y, p.z] if add else None
        self._tick()

    # --- core logic --------------------------------------------------------
    def _tick(self):
        now = time.monotonic()
        dual = (self.target["left"] is not None and
                self.target["right"] is not None)
        do_sample = (now - self._last_sample) >= (1.0 / SAMPLE_HZ)
        if do_sample:
            self._last_sample = now
        for arm in ("left", "right"):
            tgt, tip = self.target[arm], self.tip[arm]
            if tgt is None or tip is None:
                self._ep_target[arm] = None
                self._ep_logged[arm] = False
                continue
            err_mm = float(np.linalg.norm(np.array(tgt) - np.array(tip))) * 1e3
            if do_sample:
                self._write_sample(now, arm, tgt, tip, err_mm, dual)

            # settle-episode tracking
            ep = self._ep_target[arm]
            moved = ep is None or float(np.linalg.norm(
                np.array(tgt) - np.array(ep))) > MOVE_EPS
            if moved:
                self._ep_target[arm] = list(tgt)
                self._ep_since[arm] = now
                self._ep_logged[arm] = False
            else:
                settled = (now - self._ep_since[arm]) >= SETTLE_S
                if settled and not self._ep_logged[arm] and err_mm > FAIL_MM:
                    self._log_failure(arm, tgt, tip, err_mm, dual)
                    self._ep_logged[arm] = True

    def _arm_q_row(self, arm):
        return [round(self.q.get(j, float("nan")), 5) for j in ARM_JOINTS[arm]]

    def _write_sample(self, now, arm, tgt, tip, err_mm, dual):
        row = ([f"{now:.3f}", arm] + [f"{v:.4f}" for v in tgt]
               + [f"{v:.4f}" for v in tip] + [f"{err_mm:.1f}", int(dual),
               f"{self.q.get(LIFT_JOINT, float('nan')):.4f}"]
               + self._arm_q_row("left") + self._arm_q_row("right"))
        with open(self.csv_path, "a") as fh:
            fh.write(",".join(str(c) for c in row) + "\n")

    def _joint_detail(self):
        """Per commanded joint: value + range fraction; flag saturated ones."""
        out, saturated = {}, []
        for j, (lo, hi) in self.limits.items():
            v = self.q.get(j)
            if v is None:
                continue
            frac = (v - lo) / (hi - lo) if hi > lo else 0.5
            out[j] = {"q": round(v, 5), "lo": round(lo, 4), "hi": round(hi, 4),
                      "frac": round(frac, 3)}
            if frac <= SAT_FRAC or frac >= 1.0 - SAT_FRAC:
                saturated.append(f"{j}={frac:.2f}")
        return out, saturated

    def _log_failure(self, arm, tgt, tip, err_mm, dual):
        detail, saturated = self._joint_detail()
        rec = {
            "wall_t": round(time.time(), 3),
            "arm": arm,
            "target": [round(v, 4) for v in tgt],
            "fingertip": [round(v, 4) for v in tip],
            "err_mm": round(err_mm, 1),
            "dual_active": dual,
            "other_target": ([round(v, 4) for v in self.target[
                "right" if arm == "left" else "left"]]
                if self.target["right" if arm == "left" else "left"] else None),
            "lift": round(self.q.get(LIFT_JOINT, float("nan")), 4),
            "saturated_joints": saturated,
            "joints": detail,
        }
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._n_fail += 1
        self.get_logger().warn(
            f"[FAIL #{self._n_fail}] {arm} target={rec['target']} "
            f"err={err_mm:.0f}mm dual={dual} lift={rec['lift']} "
            f"sat={saturated or 'none'}")


def main():
    rclpy.init()
    node = ReachLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"reach logger stopping; {node._n_fail} failure episodes recorded.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
