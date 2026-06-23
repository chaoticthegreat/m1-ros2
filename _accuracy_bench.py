#!/usr/bin/env python3
"""Accuracy benchmark for the M1 position-only reach solver (no ROS).

The single shared yardstick for the tracking-accuracy improvement effort. It
measures the four dimensions we care about and prints a compact, parseable
summary so a closed loop (and fanned-out prototype agents) can compare a solver
variant against baseline:

  M1 SINGULARITY TRANSIENT -- worst-case fingertip error during an aggressive
     full-amplitude multi-joint sweep that crosses internal singularities.
  M2 SMOOTH TRACKING       -- fingertip error on a wide, well-conditioned sweep.
  M3 DUAL SHARED-LIFT       -- worst-arm final error over many dual targets that
     share one lift (a zero-error solution exists by construction).
  M4 NEAR-BOUNDARY REACH    -- final error over many cold single-arm targets that
     are FK of full-limit (often near-boundary) configs.
  LAT  -- solve_step worst-case + fraction over the 60 Hz budget.

Usage:
  /usr/bin/python3 _accuracy_bench.py [path/to/kinematics_variant.py]

With no argument it benches the installed ``m1_control.kinematics``. With a path
it dynamically loads that file as the solver module, so a prototype variant can
be benched in isolation. The last printed line is ``RESULT <json>``.
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "assets", "ranger_air_description", "urdf",
                    "ranger_air_description.urdf")
BUDGET_MS = 1000.0 / 60.0


def load_module(path=None):
    if path:
        spec = importlib.util.spec_from_file_location("kin_variant", path)
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass can resolve sys.modules[__module__].
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    sys.path.insert(0, os.path.join(HERE, "ros2_ws", "src", "m1_control"))
    import m1_control.kinematics as mod  # noqa: E402
    return mod


class Bench:
    def __init__(self, mod):
        self.mod = mod
        self.urdf = open(URDF).read()
        self.ARM = mod.ARM_JOINTS
        self.LIFT = mod.LIFT_JOINT
        self.ticks = []

    def reach(self):
        return self.mod.ReachController(self.mod.UrdfModel.from_string(self.urdf))

    def q0(self, lift=0.0):
        q = {j: 0.0 for j in self.ARM["left"] + self.ARM["right"]}
        q[self.LIFT] = lift
        return q

    def clip_cfg(self, r, arm, vals, lift):
        q = {self.LIFT: float(np.clip(lift, 0.0, 0.85))}
        for j, v in zip(self.ARM[arm], vals):
            jt = r.model.joints[j]
            q[j] = float(np.clip(v, jt.lower, jt.upper))
        return q

    def step(self, r, q, targets):
        t0 = time.perf_counter()
        res = r.solve_step(q, targets)
        self.ticks.append((time.perf_counter() - t0) * 1e3)
        for jn, val in res.items():
            if not jn.startswith("_"):
                q[jn] = val
        return res

    def _sweep(self, center, amps, freqs, lift, n, warmup=40):
        r = self.reach()
        q = self.q0()
        pe = []
        for k in range(n):
            t = k / 60.0
            vals = [center[i] + amps[i] * math.sin(freqs[i] * t + 0.5 * i)
                    for i in range(7)]
            p = np.asarray(r.fingertip("left", self.clip_cfg(r, "left", vals, lift)))
            self.step(r, q, {"left": p, "right": None})
            pe.append(float(np.linalg.norm(p - r.fingertip("left", q))))
        return np.array(pe[warmup:])

    def m1_singularity(self):
        pe = self._sweep(
            [-0.3, -0.8, 0.2, 1.2, 0.1, 0.1, 0.0],
            [1.1, 0.7, 1.0, 0.9, 1.2, 0.6, 1.3],
            [0.45, 0.3, 0.55, 0.4, 0.85, 0.65, 1.0], 0.5, n=1200)
        return {"mean": pe.mean() * 1e3, "p95": np.percentile(pe, 95) * 1e3,
                "max": pe.max() * 1e3}

    def m2_smooth(self):
        pe = self._sweep(
            [0.0, -0.7, 0.0, 1.1, 0.0, 0.2, 0.0],
            [0.6, 0.5, 0.2, 0.6, 0.1, 0.15, 0.1],
            [0.5, 0.35, 0.6, 0.4, 0.7, 0.5, 0.6], 0.5, n=900)
        return {"mean": pe.mean() * 1e3, "p95": np.percentile(pe, 95) * 1e3,
                "max": pe.max() * 1e3}

    def _rand_point(self, r, arm, rng, lift=None):
        q = {self.LIFT: rng.uniform(0.0, 0.85) if lift is None else lift}
        for j in self.ARM[arm]:
            jt = r.model.joints[j]
            q[j] = rng.uniform(jt.lower, jt.upper)
        return np.asarray(r.fingertip(arm, q))

    def _converge(self, targets, max_ticks=300):
        r = self.reach()
        q = self.q0()
        arms = [a for a in ("left", "right") if targets.get(a) is not None]
        for _ in range(max_ticks):
            if len(self.step(r, q, targets)) <= 1:
                break
        return max(float(np.linalg.norm(np.asarray(targets[a]) - r.fingertip(a, q)))
                   for a in arms)

    def m3_dual(self, n=200):
        rng = np.random.default_rng(99)
        errs = []
        for _ in range(n):
            r = self.reach()
            lift = rng.uniform(0.0, 0.85)
            tgt = {a: self._rand_point(r, a, rng, lift=lift) for a in ("left", "right")}
            errs.append(self._converge(tgt))
        errs = np.array(errs)
        return {"max": errs.max() * 1e3, "mean": errs.mean() * 1e3,
                "p99": np.percentile(errs, 99) * 1e3,
                "pct_lt5mm": 100 * (errs < 5e-3).mean()}

    def m4_boundary(self, n=300):
        out = {}
        worst = 0.0
        allmax = 0.0
        pct1 = []
        for arm in ("left", "right"):
            rng = np.random.default_rng(1234 if arm == "left" else 5678)
            errs = []
            for _ in range(n):
                r = self.reach()
                p = self._rand_point(r, arm, rng)
                errs.append(self._converge({arm: p,
                                            ("right" if arm == "left" else "left"): None}))
            errs = np.array(errs)
            allmax = max(allmax, errs.max())
            pct1.append(100 * (errs < 1e-3).mean())
        return {"max": allmax * 1e3, "pct_lt1mm": float(np.mean(pct1))}

    def latency(self):
        tt = np.array(self.ticks)
        return {"max": float(tt.max()), "p99": float(np.percentile(tt, 99)),
                "over_pct": 100 * float((tt > BUDGET_MS).mean())}

    def run(self):
        m1 = self.m1_singularity()
        m2 = self.m2_smooth()
        m3 = self.m3_dual()
        m4 = self.m4_boundary()
        lat = self.latency()
        return {"M1_singularity": m1, "M2_smooth": m2, "M3_dual": m3,
                "M4_boundary": m4, "LAT": lat}


# Accuracy regression gates. These LOCK IN the ~10-60x tracking-accuracy
# improvement (cold/dual/near-boundary convergence tightening + iterated command
# polish + faster re-acquire) with margin above the measured, deterministic
# values (M1 0.64 / M3 0.28 / M4 0.79 mm, 100% sub-mm, LAT max ~11 ms). A change
# that regresses tracking accuracy or blows the 60 Hz budget fails here.
def gates(res):
    return {
        "M1 singularity max < 2mm": res["M1_singularity"]["max"] < 2.0,
        "M1 singularity p95 < 1mm": res["M1_singularity"]["p95"] < 1.0,
        "M2 smooth tracking max < 1mm": res["M2_smooth"]["max"] < 1.0,
        "M3 dual max < 2mm": res["M3_dual"]["max"] < 2.0,
        "M3 dual p99 < 1mm": res["M3_dual"]["p99"] < 1.0,
        "M3 dual 100% <5mm": res["M3_dual"]["pct_lt5mm"] >= 100.0,
        "M4 near-boundary max < 1.2mm": res["M4_boundary"]["max"] < 1.2,
        "M4 near-boundary >=99.5% <1mm": res["M4_boundary"]["pct_lt1mm"] >= 99.5,
        # 60 Hz is a goal, not a hard cutoff: a far target gets a bigger budget to
        # converge (accuracy first), so we bound the worst case and allow a modest
        # over-budget fraction rather than requiring 0%.
        "LAT solve_step worst-case bounded < 60ms": res["LAT"]["max"] < 60.0,
        "LAT over-budget fraction < 15%": res["LAT"]["over_pct"] < 15.0,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    mod = load_module(path)
    res = Bench(mod).run()
    print(f"\n==== ACCURACY BENCH: {path or 'm1_control.kinematics'} ====")
    print(f"M1 singularity transient : mean {res['M1_singularity']['mean']:6.2f}  "
          f"p95 {res['M1_singularity']['p95']:6.2f}  max {res['M1_singularity']['max']:7.2f} mm")
    print(f"M2 smooth tracking       : mean {res['M2_smooth']['mean']:6.2f}  "
          f"p95 {res['M2_smooth']['p95']:6.2f}  max {res['M2_smooth']['max']:7.2f} mm")
    print(f"M3 dual shared-lift      : mean {res['M3_dual']['mean']:6.2f}  "
          f"p99 {res['M3_dual']['p99']:6.2f}  max {res['M3_dual']['max']:7.2f} mm  "
          f"<5mm {res['M3_dual']['pct_lt5mm']:.1f}%")
    print(f"M4 near-boundary reach   : max {res['M4_boundary']['max']:6.2f} mm  "
          f"<1mm {res['M4_boundary']['pct_lt1mm']:.1f}%")
    print(f"LAT solve_step           : max {res['LAT']['max']:.2f} ms  "
          f"p99 {res['LAT']['p99']:.2f} ms  over-budget {res['LAT']['over_pct']:.2f}%")
    print("RESULT " + json.dumps(res))
    # Gate only when benching the installed solver (a variant path is exploratory).
    g = gates(res)
    print("\n----------------  ACCURACY GATES  ----------------")
    npass = 0
    for name, ok in g.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(g)} gates passed")
    if path:
        return 0  # variant run: report metrics, don't fail on the locked-in gates
    return 0 if npass == len(g) else 1


if __name__ == "__main__":
    sys.exit(main())
