"""Damiao (DM-series) CAN protocol codec -- pure-python, dependency-free.

This module is the byte-exact heart of the M1 hardware bring-up. It encodes the
three Damiao command modes (MIT impedance, position-velocity, velocity), the
four special control frames (enable / disable / set-zero / clear-error), the
mode arbitration-ID offsets, and decodes a feedback frame (position / velocity /
torque + MOS & rotor temperature + the error nibble).

It imports NOTHING third-party (no ``can``, no ``rclpy``, not even numpy) so it
is importable and unit-testable on any interpreter, anywhere -- exactly like the
gated solver tests. The physical bus and the ROS node live in ``transport`` and
``hwconfig_node``; this layer is just the protocol.

Conventions (verbatim from the deployment design's Global Constraints):

* Per-model ``[P_MAX (rad), V_MAX (rad/s), T_MAX (Nm)]`` limit tables in
  :data:`LIMITS`. KP range ``0..500``, KD range ``0..5``.
* MIT command packing (8 bytes): ``q`` 16-bit, ``dq`` 12-bit, ``kp`` 12-bit
  (over ``0..500``), ``kd`` 12-bit (over ``0..5``), ``tau`` 12-bit -- each
  field a symmetric (or 0-based for kp/kd) linear quantization of the value
  over its range.
* Mode arbitration-ID offsets added to the slave id: MIT ``+0x000``,
  POS_VEL ``+0x100``, VEL ``+0x200``, FORCE_POS ``+0x300``.
* Special frames are 8 bytes of ``0xFF`` with the last byte the opcode
  (enable ``0xFC``, disable ``0xFD``, set-zero ``0xFE``, clear-error ``0xFB``),
  sent to the slave id.
"""
from __future__ import annotations

import struct
from typing import Dict, Tuple

# --- Per-model limit tables -------------------------------------------------
# model -> (P_MAX rad, V_MAX rad/s, T_MAX Nm). Verbatim from the design's
# Global Constraints; these scale every MIT command/feedback quantization.
LIMITS: Dict[str, Tuple[float, float, float]] = {
    "DM4310":     (12.5, 30.0, 10.0),
    "DM4310_48V": (12.5, 50.0, 10.0),
    "DM4340":     (12.5, 8.0, 28.0),
    "DM4340_48V": (12.5, 10.0, 28.0),
    "DM6006":     (12.5, 45.0, 20.0),
    "DM8006":     (12.5, 45.0, 40.0),
    "DM8009":     (12.5, 45.0, 54.0),
    "DM10010L":   (12.5, 25.0, 200.0),
    "DM10010":    (12.5, 20.0, 200.0),
    "DMH3510":    (12.5, 280.0, 1.0),
    "DMH6215":    (12.5, 45.0, 10.0),
    "DMG6220":    (12.5, 45.0, 10.0),
}

# Gain quantization ranges (shared across all models).
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

# Mode -> arbitration-ID offset added to the slave id.
MODE_OFFSET: Dict[str, int] = {
    "mit":       0x000,
    "pos_vel":   0x100,
    "vel":       0x200,
    "force_pos": 0x300,
}

# Special control frame opcodes (last byte; first 7 bytes are 0xFF).
_SPECIAL_OPCODE: Dict[str, int] = {
    "enable":      0xFC,
    "disable":     0xFD,
    "set_zero":    0xFE,
    "clear_error": 0xFB,
}

# Convention: master id = slave id + 0x10 (never 0).
MASTER_ID_OFFSET = 0x10


def master_id(slave_id: int) -> int:
    """Master (host) arbitration id for a given slave id (slave + 0x10)."""
    return slave_id + MASTER_ID_OFFSET


def limits(model: str) -> Tuple[float, float, float]:
    """Return ``(P_MAX, V_MAX, T_MAX)`` for *model*; raise on unknown model."""
    try:
        return LIMITS[model]
    except KeyError as exc:  # noqa: BLE001
        raise KeyError(
            f"unknown DM model {model!r}; known: {sorted(LIMITS)}") from exc


# --- Scalar quantization ----------------------------------------------------
def float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    """Quantize *x* in ``[lo, hi]`` to an unsigned ``bits``-bit integer.

    Clamps *x* into the range first, then maps linearly so ``lo -> 0`` and
    ``hi -> (1<<bits) - 1``. This matches the Damiao reference packing.
    """
    span = hi - lo
    if span <= 0.0:
        return 0
    if x < lo:
        x = lo
    elif x > hi:
        x = hi
    return int((x - lo) * ((1 << bits) - 1) / span)


def uint_to_float(u: int, lo: float, hi: float, bits: int) -> float:
    """Inverse of :func:`float_to_uint`: integer back to a float in ``[lo,hi]``."""
    span = hi - lo
    return float(u) * span / ((1 << bits) - 1) + lo


# --- Command encoders -------------------------------------------------------
def encode_mit(
    p: float, v: float, kp: float, kd: float, tau: float, model: str
) -> bytes:
    """Encode an 8-byte MIT-impedance command frame for *model*.

    Field packing (Damiao reference): ``q`` 16-bit over ``[-P_MAX, P_MAX]``,
    ``dq`` 12-bit over ``[-V_MAX, V_MAX]``, ``kp`` 12-bit over ``[0, 500]``,
    ``kd`` 12-bit over ``[0, 5]``, ``tau`` 12-bit over ``[-T_MAX, T_MAX]``.
    The 12-bit ``dq``/``kp`` share a byte, as do ``kd``/``tau``::

        data[0] = q >> 8
        data[1] = q & 0xFF
        data[2] = dq >> 4
        data[3] = ((dq & 0xF) << 4) | (kp >> 8)
        data[4] = kp & 0xFF
        data[5] = kd >> 4
        data[6] = ((kd & 0xF) << 4) | (tau >> 8)
        data[7] = tau & 0xFF
    """
    p_max, v_max, t_max = limits(model)
    q_u = float_to_uint(p, -p_max, p_max, 16)
    dq_u = float_to_uint(v, -v_max, v_max, 12)
    kp_u = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_u = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    tau_u = float_to_uint(tau, -t_max, t_max, 12)
    return bytes(
        (
            (q_u >> 8) & 0xFF,
            q_u & 0xFF,
            (dq_u >> 4) & 0xFF,
            ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
            kp_u & 0xFF,
            (kd_u >> 4) & 0xFF,
            ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF),
            tau_u & 0xFF,
        )
    )


def encode_pos_vel(pos: float, vel: float) -> bytes:
    """Encode a position-velocity command: two little-endian float32 (8 bytes)."""
    return struct.pack("<ff", float(pos), float(vel))


def encode_vel(vel: float) -> bytes:
    """Encode a velocity command: one little-endian float32 (4 bytes)."""
    return struct.pack("<f", float(vel))


def special_frame(kind: str) -> bytes:
    """Return the 8-byte special control frame for *kind*.

    *kind* in ``{"enable", "disable", "set_zero", "clear_error"}``. The frame is
    seven ``0xFF`` bytes followed by the opcode byte; it is sent to the slave id.
    """
    try:
        opcode = _SPECIAL_OPCODE[kind]
    except KeyError as exc:  # noqa: BLE001
        raise ValueError(
            f"unknown special frame {kind!r}; "
            f"expected one of {sorted(_SPECIAL_OPCODE)}") from exc
    return bytes([0xFF] * 7 + [opcode])


def arb_id(slave_id: int, mode: str) -> int:
    """Arbitration id for *slave_id* in command *mode*.

    *mode* in ``{"mit", "pos_vel", "vel", "force_pos"}``; the per-mode offset is
    added to the slave id (MIT ``+0``, POS_VEL ``+0x100``, VEL ``+0x200``,
    FORCE_POS ``+0x300``).
    """
    try:
        return slave_id + MODE_OFFSET[mode]
    except KeyError as exc:  # noqa: BLE001
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(MODE_OFFSET)}"
        ) from exc


# --- Feedback decode --------------------------------------------------------
def decode_feedback(data: bytes, model: str) -> dict:
    """Decode an 8-byte Damiao feedback frame for *model*.

    Layout::

        D[0]   low nibble = motor id, high nibble = error code
        D[1:3] position, 16-bit over [-P_MAX, P_MAX]
        D[3], D[4]>>4   velocity, 12-bit over [-V_MAX, V_MAX]
        D[4]&0xF, D[5]  torque, 12-bit over [-T_MAX, T_MAX]
        D[6]   MOS temperature (deg C, raw byte)
        D[7]   rotor temperature (deg C, raw byte)

    Returns ``{id, err, pos, vel, torque, t_mos, t_rotor}``.
    """
    if len(data) < 8:
        raise ValueError(f"feedback frame needs 8 bytes, got {len(data)}")
    p_max, v_max, t_max = limits(model)

    motor_id = data[0] & 0x0F
    err = (data[0] >> 4) & 0x0F

    pos_u = (data[1] << 8) | data[2]
    vel_u = (data[3] << 4) | (data[4] >> 4)
    tau_u = ((data[4] & 0x0F) << 8) | data[5]

    return {
        "id": motor_id,
        "err": err,
        "pos": uint_to_float(pos_u, -p_max, p_max, 16),
        "vel": uint_to_float(vel_u, -v_max, v_max, 12),
        "torque": uint_to_float(tau_u, -t_max, t_max, 12),
        "t_mos": int(data[6]),
        "t_rotor": int(data[7]),
    }
