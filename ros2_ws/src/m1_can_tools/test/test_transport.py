"""Transport abstraction tests (fake / socketcan / serial).

``FakeTransport`` is the hardware-free backend the rest of the suite drives. The
real backends (``SocketCanTransport`` / ``SerialTransport``) import ``can`` /
``serial`` LAZILY -- only when actually used -- mirroring how ``kinematics.py``
imports ``pydrake``. These tests assert that laziness so they run on a machine
with neither library installed.
"""
import sys
import types

import pytest

from m1_can_tools import transport as tp


def test_fake_send_records():
    t = tp.FakeTransport()
    t.send(0x101, b"\x01\x02\x03")
    assert t.sent == [(0x101, b"\x01\x02\x03")]
    t.close()


def test_fake_inject_recv():
    t = tp.FakeTransport()
    assert t.recv(timeout=0.0) is None        # empty queue -> None
    t.inject(0x11, b"\xAA\xBB")
    assert t.recv(timeout=0.0) == (0x11, b"\xAA\xBB")
    assert t.recv(timeout=0.0) is None        # drained


def test_make_transport_fake():
    t = tp.make_transport({"kind": "fake"})
    assert isinstance(t, tp.FakeTransport)


def test_make_transport_unknown_kind():
    with pytest.raises(ValueError):
        tp.make_transport({"kind": "bogus"})


def test_socketcan_is_lazy(monkeypatch):
    # Constructing a SocketCanTransport must NOT import `can`. We poison the
    # import so that any attempt to load it would blow up; construction proceeds
    # only because the import is deferred to first use.
    assert "can" not in sys.modules
    poison = types.ModuleType("can")

    def _boom(*a, **k):  # pragma: no cover - only hit if laziness breaks
        raise AssertionError("`can` was imported at construction time")

    poison.interface = types.SimpleNamespace(Bus=_boom)
    monkeypatch.setitem(sys.modules, "can", poison)

    t = tp.SocketCanTransport(channel="can0")     # no import/use yet -> ok
    assert t.channel == "can0"
    # First real use triggers the (poisoned) import -> our sentinel fires.
    with pytest.raises(AssertionError):
        t.send(0x01, b"\x00")


def test_socketcan_missing_lib_clear_error(monkeypatch):
    # If python-can is genuinely absent, the error must be a clear, actionable
    # message -- not a bare ModuleNotFoundError deep in the stack.
    monkeypatch.setitem(sys.modules, "can", None)  # forces ImportError on import
    t = tp.SocketCanTransport(channel="can0")
    with pytest.raises(RuntimeError) as exc:
        t.send(0x01, b"\x00")
    assert "python-can" in str(exc.value)


def test_serial_is_lazy(monkeypatch):
    poison = types.ModuleType("serial")

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("`serial` was imported at construction time")

    poison.Serial = _boom
    monkeypatch.setitem(sys.modules, "serial", poison)
    t = tp.SerialTransport(dev="/dev/ttyACM0", baud=921600)
    assert t.dev == "/dev/ttyACM0" and t.baud == 921600
    with pytest.raises(AssertionError):
        t.send(0x01, b"\x00" * 8)


# --- SimTransport: virtual motors reply to refresh / MIT -------------------
def test_sim_transport_refresh_and_mit():
    import m1_can_tools.transport as tp
    from m1_can_tools import dm_protocol as dm
    sim = tp.make_transport({"kind": "sim", "motors": {
        0x02: {"master_id": 0x12, "model": "DM8009"},
    }})
    # Refresh poll -> a feedback frame on the master id.
    sim.send(dm.PARAM_ARB_ID, dm.refresh_frame(0x02))
    frame = sim.recv()
    assert frame is not None and frame[0] == 0x12
    # MIT command -> virtual motor moves to commanded pos; next refresh reflects it.
    sim.send(dm.arb_id(0x02, "mit"), dm.encode_mit(0.5, 0, 30, 1, 0, "DM8009"))
    sim.recv()  # drain the immediate MIT reply
    sim.send(dm.PARAM_ARB_ID, dm.refresh_frame(0x02))
    arb, data = sim.recv()
    fb = dm.decode_feedback(data, "DM8009")
    assert abs(fb["pos"] - 0.5) < 0.02
    # recv with empty queue returns None (no infinite streaming).
    while sim.recv() is not None:
        pass
    assert sim.recv() is None
