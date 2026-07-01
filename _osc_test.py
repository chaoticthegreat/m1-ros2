#!/usr/bin/env /usr/bin/python3
"""Post-reach steady-state oscillation gate.

The operator drives via Quest, so the reach target is STREAMED from a hand and is
never perfectly static -- it carries ~mm sensor/tremor noise. The gated solver
suites all use a perfectly static target and stop at the held sentinel, so they
never exercise this: under a slightly-noisy-but-stationary target the redundant
(7-DOF arm + shared prismatic lift) reach re-solves every tick against a noisy goal
and the null space -- above all the SHARED LIFT (direct 1:1 z leverage) -- random-
walks, so the operator sees the arm and lift "oscillate a bit after reaching" while
the gripper stays on target.

The fix is two layers, both gated here:
  * LAYER 1 (solver): a lift-specific tracking reg (_IK_REG_LIFT_TRACK) so the
    least-squares tracking solve routes small target motion through the arm instead
    of the shared lift -> the LIFT no longer amplifies target noise. Measured by
    feeding a RAW jittering target straight to solve_step.
  * LAYER 2 (operator/controller): a dwell-freeze that latches a stationary-but-
    jittering target once it has dwelled, so the solver sees a static goal and holds
    a still posture (arm too). A genuine move releases it instantly (no lag / dead
    zone). Mirrored here by _hold_condition (== M1Controller._hold_condition).

  /usr/bin/python3 _osc_test.py      # prints [PASS]/[FAIL] + N/N gates
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))
from m1_control.kinematics import (            # noqa: E402
    ARM_JOINTS, LIFT_JOINT, ReachController, UrdfModel,
)

URDF = "assets/ranger_air_description/urdf/ranger_air_description.urdf"
ARM_ALL = ARM_JOINTS["left"] + ARM_JOINTS["right"]
COMMANDED = [LIFT_JOINT] + ARM_ALL

# Layer-2 defaults (mirror controller_node's target_hold_band / target_hold_ticks).
HOLD_BAND = 0.006
HOLD_TICKS = 8


def load():
    with open(URDF) as fh:
        return ReachController(UrdfModel.from_string(fh.read()))


def q0():
    q = {j: 0.0 for j in ARM_ALL}
    q[LIFT_JOINT] = 0.35
    return q


def cfg(arm, lift, vals):
    q = {LIFT_JOINT: lift}
    for j, v in zip(ARM_JOINTS[arm], vals):
        q[j] = v
    return q


class HoldCond:
    """Mirror of M1Controller._hold_condition (Layer 2). Freezes a dwelled target."""
    def __init__(self, band=HOLD_BAND, ticks=HOLD_TICKS):
        self.band, self.ticks = band, ticks
        self.st = {"ref": None, "n": 0, "frozen": None}

    def __call__(self, raw):
        st = self.st
        if raw is None or self.band <= 0.0:
            st["ref"], st["n"], st["frozen"] = None, 0, None
            return raw
        if st["ref"] is None or float(np.linalg.norm(raw - st["ref"])) > self.band:
            st["ref"], st["n"], st["frozen"] = raw.copy(), 0, None
            return raw
        st["n"] += 1
        if st["n"] >= self.ticks:
            if st["frozen"] is None:
                st["frozen"] = raw.copy()
            return st["frozen"]
        return raw


def run(reach, base_tgt, nticks=300, tail=180, noise_mm=1.0, seed=7, layer2=False):
    arms = [a for a in ("left", "right") if base_tgt.get(a) is not None]
    rng = np.random.default_rng(seed)
    q = q0()
    cond = {a: HoldCond() for a in arms}
    rec = {j: [] for j in COMMANDED}
    ferr = {a: [] for a in arms}
    for i in range(nticks):
        tgt = {a: None for a in ("left", "right")}
        for a in arms:
            b = np.asarray(base_tgt[a], float)
            raw = b + (np.zeros(3) if i < 80 else rng.normal(0, noise_mm / 1000.0, 3))
            tgt[a] = cond[a](raw) if layer2 else raw
        res = reach.solve_step(q, tgt)
        for jn, val in res.items():
            if jn != "_dist":
                q[jn] = val                     # ideal feedback (isolates the controller)
        for j in COMMANDED:
            rec[j].append(q.get(j, 0.0))
        for a in arms:
            ferr[a].append(float(np.linalg.norm(np.asarray(base_tgt[a]) - reach.fingertip(a, q))))
    lift_p2p_mm = (max(rec[LIFT_JOINT][-tail:]) - min(rec[LIFT_JOINT][-tail:])) * 1e3
    arm_p2p = {j: (max(rec[j][-tail:]) - min(rec[j][-tail:])) for j in ARM_ALL}
    worst_arm_deg = np.degrees(max(arm_p2p.values()))
    fmean = np.mean([np.mean(ferr[a][-tail:]) for a in arms]) * 1e3
    return {"lift_mm": lift_p2p_mm, "arm_deg": worst_arm_deg, "ferr_mm": fmean}


def main():
    reach = load()
    single = {"left": reach.fingertip("left", cfg("left", 0.4, (0.0, 0.3, 0.0, 0.7, 0.0, 0.3, 0.0))),
              "right": None}
    dual = {"left":  reach.fingertip("left",  cfg("left",  0.4, (0.2, 0.5, 0.0, 0.9, 0.0, 0.4, 0.0))),
            "right": reach.fingertip("right", cfg("right", 0.4, (-0.2, 0.5, 0.0, 0.9, 0.0, 0.4, 0.0)))}

    # Gate thresholds. Under a 1mm-jitter hold: LAYER 1 must keep the shared LIFT from
    # amplifying the noise (the dominant, most visible term); LAYER 1+2 must make the
    # whole hold still (lift AND arm).
    LIFT_MAX_MM = 2.0
    ARM_MAX_DEG = 0.4

    gates = []
    print("LAYER 1 (solver lift-reg): raw jittering target straight into solve_step")
    for name, tgt in (("single", single), ("dual", dual)):
        r = run(reach, tgt, noise_mm=1.0, layer2=False)
        print(f"   {name:6s} hold: lift {r['lift_mm']:6.2f}mm | arm {r['arm_deg']:5.2f}deg "
              f"| reach {r['ferr_mm']:5.2f}mm (tracks jitter)")
        gates.append((f"L1 {name}: shared-lift travel < {LIFT_MAX_MM}mm", r["lift_mm"] < LIFT_MAX_MM))

    print("\nLAYER 1+2 (solver + controller dwell-freeze): realistic pipeline")
    for name, tgt in (("single", single), ("dual", dual)):
        r = run(reach, tgt, noise_mm=1.0, layer2=True)
        print(f"   {name:6s} hold: lift {r['lift_mm']:6.2f}mm | arm {r['arm_deg']:5.2f}deg "
              f"| reach {r['ferr_mm']:5.2f}mm")
        gates.append((f"L1+2 {name}: lift travel < {LIFT_MAX_MM}mm", r["lift_mm"] < LIFT_MAX_MM))
        gates.append((f"L1+2 {name}: worst arm swing < {ARM_MAX_DEG}deg", r["arm_deg"] < ARM_MAX_DEG))

    print("\n=== GATES ===")
    npass = 0
    for label, ok in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        npass += ok
    print(f"\n{npass}/{len(gates)} gates passed")
    sys.exit(0 if npass == len(gates) else 1)


if __name__ == "__main__":
    main()
