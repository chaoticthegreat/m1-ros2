"""m1_hwconfig web-node data-path test -- headless, no HTTP server, no hardware.

Like ``_quest_position_test.py`` drives the real ``on_xr_frame`` via ``__new__``
(no ROS init / DDS), this drives the real ``M1HwConfigNode`` request handlers
(``api_state`` / ``api_scan`` / ``api_jog`` / ``api_limits`` / ``api_mode`` /
``api_enable`` / ``api_zero``) against a ``MotorBus(FakeTransport)``. It asserts
the maintenance-mode guard, the clamped jog, the scan inventory, the limits YAML
write, and the jog deadman -- the whole data path the page renders -- with NO
bus and NO server.
"""
import os

import pytest

from m1_can_tools import dm_protocol as dm
from m1_can_tools.hwconfig_node import M1HwConfigNode
from m1_can_tools.motor_bus import MotorBus
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
        "dir": 1,
        "offset": 0.0,
    },
}


def make_node(tmp_path, mode="maintenance"):
    """A bare node (no ROS init / DDS) wired to a FakeTransport MotorBus."""
    n = M1HwConfigNode.__new__(M1HwConfigNode)
    t = FakeTransport()
    n._clock = [0.0]                       # monotonic stub the node reads via _now
    n._init_state(
        bus=MotorBus(t, MAP),
        motor_map=MAP,
        mode=mode,
        limits_path=os.path.join(str(tmp_path), "m1_joint_limits.yaml"),
        map_path=os.path.join(str(tmp_path), "motor_map.yaml"),
    )
    return n, t


def _decode_last_mit(t, info):
    want = dm.arb_id(info["id"], "mit")
    for arb, data in reversed(t.sent):
        if arb == want and len(data) == 8:
            return dm.decode_feedback(bytes([info["id"]]) + data[:7], info["model"])["pos"]
    raise AssertionError("no MIT frame sent")


def test_scan_lists_injected_motors(tmp_path):
    n, t = make_node(tmp_path)
    t.inject(0x11, bytes([0x01, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 40, 45]))
    t.inject(0x12, bytes([0x02, 0x7F, 0xFF, 0x7F, 0xF0, 0x00, 41, 46]))
    code, body = n.api_scan({"from": 1, "to": 3})
    assert code == 200
    ids = sorted(m["id"] for m in body["motors"])
    assert ids == [1, 2]


def test_jog_rejected_outside_maintenance(tmp_path):
    n, t = make_node(tmp_path, mode="run")
    code, body = n.api_jog({"joint": "openarm_left_joint1", "pos": 0.5, "hold": True})
    assert code == 403
    assert not any(len(d) == 8 for _, d in t.sent)   # nothing commanded


def test_jog_in_maintenance_sends_clamped_frame(tmp_path):
    n, t = make_node(tmp_path)
    # Command 5.0 rad on a joint clamped to [-1, 1] -> the sent MIT pos is 1.0.
    code, body = n.api_jog(
        {"joint": "openarm_left_joint1", "pos": 5.0, "kp": 10.0, "kd": 1.0, "hold": True}
    )
    assert code == 200
    pos = _decode_last_mit(t, MAP["openarm_left_joint1"])
    assert abs(pos - 1.0) < 0.01


def test_limits_writes_yaml(tmp_path):
    n, t = make_node(tmp_path)
    code, body = n.api_limits(
        {"joint": "openarm_left_joint1", "pos": [-0.5, 0.5], "vel": 2.0, "effort": 1.5}
    )
    assert code == 200
    import yaml
    with open(n.limits_path) as fh:
        d = yaml.safe_load(fh)
    jl = d["joint_limits"]["openarm_left_joint1"]
    assert jl["max_position"] == 0.5 and jl["min_position"] == -0.5
    assert jl["max_velocity"] == 2.0 and jl["max_effort"] == 1.5
    # the in-memory soft limit is updated too (so the next jog clamps to it)
    assert n.motor_map["openarm_left_joint1"]["soft_limits"]["pos"] == [-0.5, 0.5]


def test_limits_rejects_over_model_max(tmp_path):
    n, t = make_node(tmp_path)
    # DM4310 P_MAX is 12.5 rad; a 99 rad limit must be rejected.
    code, body = n.api_limits(
        {"joint": "openarm_left_joint1", "pos": [-99.0, 99.0], "vel": 2.0, "effort": 1.5}
    )
    assert code == 400


def test_mode_guard_blocks_writes(tmp_path):
    n, t = make_node(tmp_path)
    # switch to run mode -> enable/zero/jog all refused
    n.api_mode({"mode": "run"})
    assert n.api_enable({"joint": "openarm_left_joint1", "on": True})[0] == 403
    assert n.api_zero({"joint": "openarm_left_joint1"})[0] == 403
    # back to maintenance -> allowed
    n.api_mode({"mode": "maintenance"})
    assert n.api_enable({"joint": "openarm_left_joint1", "on": True})[0] == 200


def test_jog_deadman_zeroes_after_timeout(tmp_path):
    n, t = make_node(tmp_path)
    # A held jog at t=0.
    n._clock[0] = 0.0
    n.api_jog({"joint": "openarm_left_joint1", "pos": 0.5, "hold": True})
    assert n._jog_active["openarm_left_joint1"] is not None
    # Tick within the hold window -> still active, re-commands the held pos.
    n._clock[0] = 0.1
    n.tick_deadman()
    assert n._jog_active["openarm_left_joint1"] is not None
    # Tick past the hold window -> the jog is dropped (disabled), not re-sent.
    n._clock[0] = 5.0
    sent_before = len(t.sent)
    n.tick_deadman()
    assert n._jog_active["openarm_left_joint1"] is None
    # A disable frame was issued to halt the motor.
    assert t.sent[-1] == (dm.arb_id(0x01, "mit"), dm.special_frame("disable"))


def test_api_state_reports_map_and_mode(tmp_path):
    n, t = make_node(tmp_path)
    code, body = n.api_state({})
    assert code == 200
    assert body["mode"] == "maintenance"
    assert set(body["map"].keys()) == set(MAP.keys())
    assert "bus_ok" in body
