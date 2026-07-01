"""m1_hwconfig web-node data-path test -- headless, no HTTP server, no hardware.

Like ``_quest_position_test.py`` drives the real ``on_xr_frame`` via ``__new__``
(no ROS init / DDS), this drives the real ``M1HwConfigNode`` request handlers
(``api_state`` / ``api_scan`` / ``api_jog`` / ``api_limits`` / ``api_mode`` /
``api_enable`` / ``api_zero``) against a ``MotorBus(FakeTransport)``. It asserts
the maintenance-mode guard, the clamped jog, the scan inventory, the limits YAML
write, and the jog deadman -- the whole data path the page renders -- with NO
bus and NO server.
"""
import copy
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


# --- regression: map/assign contract + validation (BUGs 13/14/29/30) -------
def _fresh_node(tmp_path, mode="maintenance"):
    """A node with a PRIVATE deep-copied map so a mutate doesn't leak across
    tests (the shared module-level ``MAP`` is otherwise aliased into the node)."""
    n = M1HwConfigNode.__new__(M1HwConfigNode)
    t = FakeTransport()
    n._clock = [0.0]
    local_map = copy.deepcopy(MAP)
    n._init_state(
        bus=MotorBus(t, local_map),
        motor_map=local_map,
        mode=mode,
        limits_path=os.path.join(str(tmp_path), "m1_joint_limits.yaml"),
        map_path=os.path.join(str(tmp_path), "motor_map.yaml"),
    )
    return n, t


def test_map_persists_nonzero_kp_kd_default(tmp_path):
    # BUG 13: api_map must persist NON-ZERO kp/kd (the C++ plugin refuses the
    # bus if a commanded joint has kp == 0). A page-style call sends no kp/kd.
    n, _ = _fresh_node(tmp_path)
    code, _ = n.api_map({"joint": "openarm_left_joint1", "id": 0x01,
                         "master_id": 0x11, "model": "DM8009"})
    assert code == 200
    e = n.motor_map["openarm_left_joint1"]
    assert e["kp"] > 0 and e["kd"] > 0
    # DM8009 default mirrors control_gains.yaml (proximal/lift kp70/kd2.5).
    assert e["kp"] == 70.0 and e["kd"] == 2.5


def test_map_default_kp_kd_is_model_keyed(tmp_path):
    n, _ = _fresh_node(tmp_path)
    # DM4310 (wrist) and DMH3510 (gripper) get distinct non-zero defaults.
    n.api_map({"joint": "openarm_left_joint1", "id": 0x01, "model": "DM4310"})
    assert n.motor_map["openarm_left_joint1"]["kp"] == 10.0
    n.api_map({"joint": "openarm_left_joint2", "id": 0x02, "model": "DMH3510"})
    assert n.motor_map["openarm_left_joint2"]["kp"] == 5.0


def test_map_explicit_kp_kd_honored(tmp_path):
    n, _ = _fresh_node(tmp_path)
    n.api_map({"joint": "openarm_left_joint1", "id": 0x01, "model": "DM4310",
               "kp": 33.0, "kd": 1.25})
    e = n.motor_map["openarm_left_joint1"]
    assert e["kp"] == 33.0 and e["kd"] == 1.25


def test_map_master_id_zero_replaced_with_default(tmp_path):
    # BUG 14: the page sends +('') === 0 for a blank master field; 0 is
    # reserved/invalid and must be replaced with id + 0x10.
    n, _ = _fresh_node(tmp_path)
    code, _ = n.api_map({"joint": "openarm_left_joint1", "id": 0x05,
                         "master_id": 0, "model": "DM4310"})
    assert code == 200
    assert n.motor_map["openarm_left_joint1"]["master_id"] == dm.master_id(0x05) == 0x15


def test_assign_master_id_zero_replaced_with_default(tmp_path):
    # BUG 14: same defaulting in api_assign.
    n, _ = _fresh_node(tmp_path)
    code, _ = n.api_assign({"old_id": 0x01, "new_id": 0x07, "master_id": 0})
    assert code == 200
    e = n.motor_map["openarm_left_joint1"]
    assert e["id"] == 0x07 and e["master_id"] == dm.master_id(0x07) == 0x17


def test_map_remap_honors_new_dir_offset(tmp_path):
    # BUG 29: a re-map with a NEW dir/offset must overwrite (was setdefault,
    # which silently dropped the new value once the key existed).
    n, _ = _fresh_node(tmp_path)
    assert n.motor_map["openarm_left_joint1"]["dir"] == 1
    code, _ = n.api_map({"joint": "openarm_left_joint1", "id": 0x01,
                         "master_id": 0x11, "model": "DM4310",
                         "dir": -1, "offset": 0.25})
    assert code == 200
    e = n.motor_map["openarm_left_joint1"]
    assert e["dir"] == -1 and e["offset"] == 0.25


def test_map_omitted_dir_offset_preserves_existing(tmp_path):
    # BUG 29 corollary: omitting dir/offset keeps the existing value (does not
    # reset to 1 / 0.0). Seed a non-default dir/offset, then re-map without them.
    n, _ = _fresh_node(tmp_path)
    n.motor_map["openarm_left_joint1"]["dir"] = -1
    n.motor_map["openarm_left_joint1"]["offset"] = 0.5
    n.api_map({"joint": "openarm_left_joint1", "id": 0x01, "model": "DM4310"})
    e = n.motor_map["openarm_left_joint1"]
    assert e["dir"] == -1 and e["offset"] == 0.5


@pytest.mark.parametrize("bad_id", [-3, 0, 0x800, 99999])
def test_map_rejects_out_of_range_id(tmp_path, bad_id):
    # BUG 30: an out-of-range CAN id (must be [1, 0x7FF]) is rejected 400 and
    # the map is left unchanged.
    n, _ = _fresh_node(tmp_path)
    before = dict(n.motor_map["openarm_left_joint1"])
    code, body = n.api_map({"joint": "openarm_left_joint1", "id": bad_id,
                            "model": "DM4310"})
    assert code == 400 and body["ok"] is False
    assert n.motor_map["openarm_left_joint1"] == before


@pytest.mark.parametrize("bad_id", [-3, 0, 0x800, 99999])
def test_assign_rejects_out_of_range_id(tmp_path, bad_id):
    # BUG 30: same id validation in api_assign.
    n, _ = _fresh_node(tmp_path)
    code, body = n.api_assign({"old_id": 0x01, "new_id": bad_id})
    assert code == 400 and body["ok"] is False
    assert n.motor_map["openarm_left_joint1"]["id"] == 0x01   # unchanged


def test_map_rejects_out_of_range_explicit_master_id(tmp_path):
    # BUG 30: an explicitly-supplied master_id out of [1, 0x7FF] is rejected.
    n, _ = _fresh_node(tmp_path)
    code, body = n.api_map({"joint": "openarm_left_joint1", "id": 0x01,
                            "master_id": 0x900, "model": "DM4310"})
    assert code == 400 and body["ok"] is False


def test_map_valid_id_at_can_boundaries_accepted(tmp_path):
    n, _ = _fresh_node(tmp_path)
    # id 1 (auto master 0x11) is valid.
    assert n.api_map({"joint": "openarm_left_joint1", "id": 1,
                      "model": "DM4310"})[0] == 200
    # id 0x7EF auto-derives master 0x7FF (the last in-range value) -> accepted.
    assert n.api_map({"joint": "openarm_left_joint1", "id": 0x7EF,
                      "model": "DM4310"})[0] == 200


def test_map_id_with_overflowing_auto_master_rejected(tmp_path):
    # id 0x7FF is itself in range, but its AUTO-derived master (0x80F) overflows
    # the CAN range -> rejected unless an explicit in-range master_id is supplied.
    n, _ = _fresh_node(tmp_path)
    assert n.api_map({"joint": "openarm_left_joint1", "id": 0x7FF,
                      "model": "DM4310"})[0] == 400
    assert n.api_map({"joint": "openarm_left_joint1", "id": 0x7FF,
                      "master_id": 0x7FE, "model": "DM4310"})[0] == 200
