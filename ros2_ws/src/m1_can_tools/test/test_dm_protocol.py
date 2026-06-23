"""Damiao (DM-series) CAN protocol codec tests.

These tests are the correctness contract for ``dm_protocol`` -- the per-model
limit tables, the float<->uint quantization used by every MIT frame, the
byte-exact MIT / pos-vel / vel encoders, the special (enable/disable/zero/clear)
frames, the arbitration-ID mode offsets, and the feedback decode.

The module is pure-python with NO third-party deps (no ``can``, no ``rclpy``),
so these run anywhere:

    PYTHONPATH=ros2_ws/src/m1_can_tools \
        /usr/bin/python3 -m pytest ros2_ws/src/m1_can_tools/test/test_dm_protocol.py -v
"""
import struct

from m1_can_tools import dm_protocol as dm


# --- Task 0.1: limit tables + quantization ---------------------------------
def test_limits_table_values():
    assert dm.LIMITS["DM4310"] == (12.5, 30.0, 10.0)
    assert dm.LIMITS["DM8009"] == (12.5, 45.0, 54.0)
    assert dm.LIMITS["DM4340"] == (12.5, 8.0, 28.0)


def test_quantization_roundtrip():
    for bits in (12, 16):
        hi = 12.5
        for x in (-hi, -1.0, 0.0, 3.3, hi):
            u = dm.float_to_uint(x, -hi, hi, bits)
            assert 0 <= u < (1 << bits)
            back = dm.uint_to_float(u, -hi, hi, bits)
            assert abs(back - x) <= (2 * hi) / (1 << bits) + 1e-9


def test_float_to_uint_endpoints():
    assert dm.float_to_uint(-12.5, -12.5, 12.5, 16) == 0
    assert dm.float_to_uint(12.5, -12.5, 12.5, 16) == (1 << 16) - 1


# --- Task 0.2: byte-exact codec --------------------------------------------
def test_special_frames():
    assert dm.special_frame("enable") == bytes([0xFF] * 7 + [0xFC])
    assert dm.special_frame("disable") == bytes([0xFF] * 7 + [0xFD])
    assert dm.special_frame("set_zero") == bytes([0xFF] * 7 + [0xFE])
    assert dm.special_frame("clear_error") == bytes([0xFF] * 7 + [0xFB])


def test_arb_ids():
    assert dm.arb_id(0x01, "mit") == 0x01
    assert dm.arb_id(0x01, "pos_vel") == 0x101
    assert dm.arb_id(0x05, "vel") == 0x205


def test_mit_zero_packing():
    # p=0,v=0,kp=0,kd=0,tau=0 over symmetric ranges -> midpoints
    b = dm.encode_mit(0, 0, 0, 0, 0, "DM4310")
    assert len(b) == 8
    # q midpoint of 16b = 0x7FFF -> data[0]=0x7F data[1]=0xFF
    assert b[0] == 0x7F and b[1] == 0xFF


def test_mit_kp_kd_ranges():
    # kp uses 0..500, kd 0..5 (NOT symmetric) -> kp=0 -> 0, kp=500 -> 0xFFF
    b_lo = dm.encode_mit(0, 0, 0, 0, 0, "DM4310")
    b_hi = dm.encode_mit(0, 0, 500, 5, 0, "DM4310")
    assert (((b_lo[3] & 0xf) << 8) | b_lo[4]) == 0
    assert (((b_hi[3] & 0xf) << 8) | b_hi[4]) == 0xFFF


def test_pos_vel_le_float():
    assert dm.encode_pos_vel(1.0, 2.0) == struct.pack("<ff", 1.0, 2.0)


def test_vel_le_float():
    assert dm.encode_vel(2.5) == struct.pack("<f", 2.5)


def test_decode_roundtrip_pos():
    # encode a feedback-style buffer and decode pos within quantization
    # build: id=1,err=0; q=0 -> 0x7FFF; vel=0,torque=0; t_mos=40,t_rotor=45
    data = bytes([0x01, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 40, 45])
    fb = dm.decode_feedback(data, "DM4310")
    assert fb["id"] == 1 and fb["err"] == 0
    assert abs(fb["pos"]) < 0.001 and fb["t_mos"] == 40 and fb["t_rotor"] == 45
