"""Maintenance-mode CAN bus owner for the M1 arms + lift.

``MotorBus`` is the single owner of the CAN bus while the robot is in
*maintenance* mode (ros2_control is down). It is what the ``m1_hwconfig`` web
page drives: scan/enumerate motors, enable/disable, jog (clamped), set-zero,
read telemetry. It speaks the :mod:`dm_protocol` codec over a :mod:`transport`
backend, so it is fully exercisable on a :class:`~m1_can_tools.transport.FakeTransport`
with no hardware.

Bus-ownership is mutually exclusive with the live ros2_control stack (see the
deployment design's safety section): in *run* mode this owner is NOT
constructed; the config page reads ``/joint_states`` instead.

The motor map (ID -> logical joint) is persisted to YAML. Schema, per joint::

    joint_name:
      id:          <int>          # CAN slave id
      master_id:   <int>          # host/master id (feedback arb id; = id + 0x10)
      model:       <str>          # DM model -> per-model [P,V,T]MAX
      soft_limits: {pos: [lo, hi], vel: <float>, effort: <float>}
      dir:         +1 | -1        # joint-direction sign vs. motor
      offset:      <float>        # zero offset (rad)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from m1_can_tools import dm_protocol as dm
from m1_can_tools.transport import Transport

# Default jog gains (gentle impedance for a maintenance nudge).
DEFAULT_JOG_KP = 10.0
DEFAULT_JOG_KD = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class MotorBus:
    """Owns the CAN bus in maintenance mode; per-joint enable/jog/zero/telemetry."""

    def __init__(self, transport: Transport, motor_map: Dict[str, dict]) -> None:
        self.transport = transport
        self.motor_map = motor_map

    # --- map helpers -------------------------------------------------------
    def _info(self, joint: str) -> dict:
        try:
            return self.motor_map[joint]
        except KeyError as exc:  # noqa: BLE001
            raise KeyError(
                f"joint {joint!r} not in motor map; "
                f"known: {sorted(self.motor_map)}") from exc

    def joints(self) -> List[str]:
        """Logical joint names known to this bus, in map order."""
        return list(self.motor_map.keys())

    # --- enable / disable --------------------------------------------------
    def _special(self, joint: str, kind: str) -> None:
        info = self._info(joint)
        self.transport.send(dm.arb_id(info["id"], "mit"), dm.special_frame(kind))

    def enable(self, joint: str) -> None:
        """Enable (energize) one motor."""
        self._special(joint, "enable")

    def disable(self, joint: str) -> None:
        """Disable (de-energize) one motor."""
        self._special(joint, "disable")

    def set_zero(self, joint: str) -> None:
        """Calibrate the current position of one motor as its zero."""
        self._special(joint, "set_zero")

    def clear_error(self, joint: str) -> None:
        """Clear a latched error on one motor."""
        self._special(joint, "clear_error")

    def enable_all(self) -> None:
        """Enable every mapped motor."""
        for joint in self.motor_map:
            self.enable(joint)

    def disable_all(self) -> None:
        """Disable every mapped motor."""
        for joint in self.motor_map:
            self.disable(joint)

    # --- jog (clamped) -----------------------------------------------------
    def jog(
        self,
        joint: str,
        pos: float,
        vel: float = 0.0,
        kp: float = DEFAULT_JOG_KP,
        kd: float = DEFAULT_JOG_KD,
        tau: float = 0.0,
    ) -> None:
        """Send one MIT command, clamped to the model AND the soft limits.

        The position/velocity/torque are clamped first to the configured
        ``soft_limits`` and then to the per-model ``[P,V,T]MAX`` (the encoder
        clamps to the model range too, but we clamp explicitly so the commanded
        value is observable). A jog is a single frame; the web page's deadman
        decides whether to keep sending it.

        NB (frame convention): ``jog`` commands in the **motor** frame -- the
        ``dir``/``offset`` calibration is NOT applied here, whereas
        :meth:`telemetry` reports ``pos`` in the **joint** frame (dir/offset
        applied). This is deliberate for raw maintenance nudges; an operator near
        a limit should read the motor-frame setpoint accordingly. (The live
        ros2_control path applies dir/offset in C++.)
        """
        info = self._info(joint)
        p_max, v_max, t_max = dm.limits(info["model"])
        soft = info.get("soft_limits", {})

        pos_lo, pos_hi = soft.get("pos", [-p_max, p_max])
        v_soft = float(soft.get("vel", v_max))
        t_soft = float(soft.get("effort", t_max))

        p = _clamp(float(pos), max(pos_lo, -p_max), min(pos_hi, p_max))
        v = _clamp(float(vel), -min(v_soft, v_max), min(v_soft, v_max))
        tq = _clamp(float(tau), -min(t_soft, t_max), min(t_soft, t_max))

        data = dm.encode_mit(p, v, kp, kd, tq, info["model"])
        self.transport.send(dm.arb_id(info["id"], "mit"), data)

    # --- telemetry / scan --------------------------------------------------
    def telemetry(self, joint: str, timeout: float = 0.01,
                  poll: bool = True) -> Optional[dict]:
        """Read & decode the most recent feedback frame for *joint*.

        Returns the decoded ``{id, err, pos, vel, torque, t_mos, t_rotor}`` (with
        ``dir``/``offset`` applied to ``pos``), or ``None`` if no frame arrived.
        When ``poll`` (default), first sends a non-energizing **refresh** request
        (``0xCC`` -> ``0x7FF``) so a motor that isn't streaming still replies --
        the documented way to poll DM state, and what makes passive maintenance
        telemetry work without enabling the motor.
        """
        info = self._info(joint)
        want = info.get("master_id", dm.master_id(info["id"]))
        if poll:
            self.transport.send(dm.PARAM_ARB_ID, dm.refresh_frame(info["id"]))
        frame = self.transport.recv(timeout=timeout)
        while frame is not None:
            arb, data = frame
            if arb == want and len(data) >= 8:
                fb = dm.decode_feedback(data, info["model"])
                fb["pos"] = info.get("dir", 1) * fb["pos"] + info.get("offset", 0.0)
                fb["joint"] = joint
                return fb
            frame = self.transport.recv(timeout=timeout)
        return None

    def scan(self, ids: Iterable[int]) -> List[dict]:
        """Poll each id in *ids* (non-energizing) and list the motors that replied.

        Used by the config page's inventory: pings the candidate slave ids with a
        **state-refresh** frame (opcode ``0xCC`` to ``0x7FF``) -- the motor replies
        with a feedback frame on its master id WITHOUT being enabled/powered, so an
        inventory scan never energizes the arm. Returns one dict per responder
        ``{id, master_id, model, joint, ...fb}``.
        """
        # Index the known map by slave id so a responder can be named.
        by_id = {info["id"]: (joint, info) for joint, info in self.motor_map.items()}
        by_master = {
            info.get("master_id", dm.master_id(info["id"])): (joint, info)
            for joint, info in self.motor_map.items()
        }
        # Poll every candidate id with the non-energizing refresh request.
        for sid in ids:
            self.transport.send(dm.PARAM_ARB_ID, dm.refresh_frame(sid))

        found: List[dict] = []
        frame = self.transport.recv(timeout=0.01)
        while frame is not None:
            arb, data = frame
            named = by_master.get(arb)
            model = named[1]["model"] if named else "DM4310"
            if len(data) >= 8:
                fb = dm.decode_feedback(data, model)
                joint = named[0] if named else None
                entry = {
                    "id": fb["id"],
                    "master_id": arb,
                    "model": model,
                    "joint": joint,
                    "pos": fb["pos"],
                    "vel": fb["vel"],
                    "torque": fb["torque"],
                    "t_mos": fb["t_mos"],
                    "t_rotor": fb["t_rotor"],
                    "err": fb["err"],
                }
                found.append(entry)
            frame = self.transport.recv(timeout=0.01)
        # Sort by id for a stable inventory listing.
        found.sort(key=lambda m: m["id"])
        return found

    def close(self) -> None:
        """Disable everything and release the transport."""
        try:
            self.disable_all()
        finally:
            self.transport.close()


# --- YAML map persistence ---------------------------------------------------
def load_map(path: str) -> Dict[str, dict]:
    """Load an ID->joint motor map from a YAML file."""
    import yaml  # ament_python dep; available for the ROS interpreter
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def save_map(path: str, m: Dict[str, dict]) -> None:
    """Persist an ID->joint motor map to a YAML file."""
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(m, fh, sort_keys=False, default_flow_style=False)
