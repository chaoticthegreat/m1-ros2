"""MotorBus (maintenance-mode bus owner) tests, driven on FakeTransport.

The bus owner is what the config/test web page sits on. These tests check the
maintenance operations -- enable / disable, jog (clamped to BOTH the per-model
limits and the configured soft limits), set-zero, telemetry decode, and scan --
plus the ID->joint YAML map round-trip. All hardware-free via FakeTransport.
"""
import os

from m1_can_tools import dm_protocol as dm
from m1_can_tools.motor_bus import MotorBus, load_map, save_map
from m1_can_tools.transport import FakeTransport


MAP = {
    "openarm_left_joint1": {
        "id": 0x01,
        "master_id": 0x11,
        "model": "DM4310",
        "soft_limits": {"pos": [-1.0, 1.0], "vel": 5.0, "effort": 3.0},
        "dir": 1,
        "offset": 0.0,
    },
    "openarm_left_joint2": {
        "id": 0x02,
        "master_id": 0x12,
        "model": "DM4340",
        "soft_limits": {"pos": [-2.0, 2.0], "vel": 4.0, "effort": 6.0},
        "dir": -1,
        "offset": 0.0,
    },
}


def _decode_last_mit(bus, transport, joint):
    """Decode the position of the most recent MIT frame sent for *joint*."""
    info = bus.motor_map[joint]
    want_id = dm.arb_id(info["id"], "mit")
    for arb, data in reversed(transport.sent):
        if arb == want_id and len(data) == 8:
            return dm.decode_feedback(
                bytes([info["id"]]) + data[:7], info["model"]
            )["pos"]
    raise AssertionError("no MIT frame found")


def test_enable_sends_mit_id_and_special():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    bus.enable("openarm_left_joint1")
    assert t.sent[-1] == (dm.arb_id(0x01, "mit"), dm.special_frame("enable"))


def test_disable_sends_special():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    bus.disable("openarm_left_joint1")
    assert t.sent[-1] == (dm.arb_id(0x01, "mit"), dm.special_frame("disable"))


def test_set_zero_sends_fe():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    bus.set_zero("openarm_left_joint1")
    assert t.sent[-1] == (dm.arb_id(0x01, "mit"), dm.special_frame("set_zero"))


def test_jog_clamps_to_soft_pos_limit():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    # Command 5.0 rad on a joint whose soft pos limit is [-1, 1] -> clamp to 1.0.
    bus.jog("openarm_left_joint1", 5.0, vel=0.0, kp=10.0, kd=1.0)
    pos = _decode_last_mit(bus, t, "openarm_left_joint1")
    assert abs(pos - 1.0) < 0.01

    # And the low side.
    bus.jog("openarm_left_joint1", -5.0)
    pos = _decode_last_mit(bus, t, "openarm_left_joint1")
    assert abs(pos - (-1.0)) < 0.01


def test_jog_within_limits_passes_through():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    bus.jog("openarm_left_joint1", 0.5)
    pos = _decode_last_mit(bus, t, "openarm_left_joint1")
    assert abs(pos - 0.5) < 0.01


def test_telemetry_decodes_injected_feedback():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    # midpoints -> pos/vel/torque ~0; temps 40/45; id=1 err=0
    t.inject(0x11, bytes([0x01, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 40, 45]))
    fb = bus.telemetry("openarm_left_joint1")
    assert fb["id"] == 1 and fb["err"] == 0
    assert abs(fb["pos"]) < 0.01 and fb["t_mos"] == 40 and fb["t_rotor"] == 45


def test_scan_lists_responders():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    # Inject feedback for ids 1 and 2 (master ids 0x11, 0x12).
    t.inject(0x11, bytes([0x01, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 40, 45]))
    t.inject(0x12, bytes([0x02, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 41, 46]))
    found = bus.scan(range(1, 4))
    ids = sorted(m["id"] for m in found)
    assert ids == [1, 2]


def test_enable_all_disable_all():
    t = FakeTransport()
    bus = MotorBus(t, MAP)
    bus.enable_all()
    enabled = [s for s in t.sent if s[1] == dm.special_frame("enable")]
    assert len(enabled) == 2
    bus.disable_all()
    disabled = [s for s in t.sent if s[1] == dm.special_frame("disable")]
    assert len(disabled) == 2


def test_map_yaml_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "motor_map.yaml")
    save_map(path, MAP)
    again = load_map(path)
    assert again == MAP
