"""Pluggable CAN transport for the M1 bring-up driver.

One small interface, three backends:

* :class:`FakeTransport` -- hardware-free; records sent frames and replays
  injected ones. The whole offline test suite (and the headless config-page
  data-path test) runs on it.
* :class:`SocketCanTransport` -- a real ``can0`` SocketCAN bus via ``python-can``.
* :class:`SerialTransport` -- a vendor USB-serial CAN dongle (``/dev/ttyACM*``)
  using its ``0x55 0xAA ... 0x55`` framing via ``pyserial``.

The real backends import their third-party library **lazily** -- only on first
use, never at construction -- mirroring how ``m1_control.kinematics`` defers its
``pydrake`` import. That keeps this module (and everything built on it, including
``dm_protocol``-level tests) importable on a machine with neither ``can`` nor
``serial`` installed. If the library is missing, the first use raises a clear,
actionable :class:`RuntimeError` telling the operator what to ``pip install``.
"""
from __future__ import annotations

import abc
import struct
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from m1_can_tools import dm_protocol as dm

Frame = Tuple[int, bytes]


class Transport(abc.ABC):
    """A bidirectional CAN frame transport (arbitration id + up to 8 data bytes)."""

    @abc.abstractmethod
    def send(self, arb_id: int, data: bytes) -> None:
        """Send one frame (``arb_id`` + ``data``)."""

    @abc.abstractmethod
    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        """Receive one frame within *timeout* seconds, or ``None`` if none arrived."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying bus / device."""


class FakeTransport(Transport):
    """In-memory transport for tests: records sends, replays injected frames.

    * :attr:`sent` -- the list of ``(arb_id, data)`` tuples that were sent.
    * :meth:`inject` -- queue a frame that the next :meth:`recv` will return.
    """

    def __init__(self) -> None:
        self.sent: List[Frame] = []
        self._rx: Deque[Frame] = deque()
        self.closed = False

    def send(self, arb_id: int, data: bytes) -> None:
        self.sent.append((int(arb_id), bytes(data)))

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        if self._rx:
            return self._rx.popleft()
        return None

    def inject(self, arb_id: int, data: bytes) -> None:
        """Queue a frame to be returned by a subsequent :meth:`recv`."""
        self._rx.append((int(arb_id), bytes(data)))

    def close(self) -> None:
        self.closed = True


class SocketCanTransport(Transport):
    """A real SocketCAN bus (``can0``) via ``python-can`` (imported lazily)."""

    def __init__(self, channel: str = "can0", fd: bool = False) -> None:
        self.channel = channel
        self.fd = fd
        self._bus = None  # built on first use

    @staticmethod
    def _import_can():
        try:
            import can  # lazy: only needed for the real bus
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "python-can is required for the SocketCAN transport. "
                "Install it for the ROS interpreter with: "
                "/usr/bin/python3 -m pip install --user "
                "--break-system-packages python-can"
            ) from exc
        return can

    def _ensure_bus(self):
        if self._bus is None:
            can = self._import_can()
            self._bus = can.interface.Bus(
                channel=self.channel, interface="socketcan", fd=self.fd
            )
        return self._bus

    def send(self, arb_id: int, data: bytes) -> None:
        can = self._import_can()
        bus = self._ensure_bus()
        bus.send(
            can.Message(
                arbitration_id=int(arb_id),
                data=bytes(data),
                is_extended_id=False,
                is_fd=self.fd,
            )
        )

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        bus = self._ensure_bus()
        msg = bus.recv(timeout=timeout)
        if msg is None:
            return None
        return (int(msg.arbitration_id), bytes(msg.data))

    def close(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None


class SerialTransport(Transport):
    """A vendor USB-serial CAN dongle via ``pyserial`` (imported lazily).

    Framing is the vendor's fixed 16-byte frame::

        [0x55, 0xAA, dlc, id0, id1, id2, id3, d0..d7, 0x55]

    i.e. a ``0x55 0xAA`` header, a data-length byte, the 32-bit arbitration id
    little-endian, the 8 data bytes (zero-padded), and a ``0x55`` tail.
    """

    HEAD0 = 0x55
    HEAD1 = 0xAA
    TAIL = 0x55
    FRAME_LEN = 16

    def __init__(self, dev: str = "/dev/ttyACM0", baud: int = 921600) -> None:
        self.dev = dev
        self.baud = baud
        self._ser = None  # built on first use
        self._rxbuf = bytearray()  # persistent rx accumulator (framing/resync)

    def _ensure_port(self):
        if self._ser is None:
            try:
                import serial  # lazy: only needed for the real dongle
            except ImportError as exc:  # noqa: BLE001
                raise RuntimeError(
                    "pyserial is required for the serial CAN transport. "
                    "Install it for the ROS interpreter with: "
                    "/usr/bin/python3 -m pip install --user "
                    "--break-system-packages pyserial"
                ) from exc
            self._ser = serial.Serial(self.dev, self.baud, timeout=0.0)
        return self._ser

    @classmethod
    def _pack(cls, arb_id: int, data: bytes) -> bytes:
        payload = bytes(data[:8]).ljust(8, b"\x00")
        return bytes(
            [cls.HEAD0, cls.HEAD1, len(data[:8])]
        ) + struct.pack("<I", int(arb_id)) + payload + bytes([cls.TAIL])

    @classmethod
    def _unpack(cls, frame: bytes) -> Optional[Frame]:
        if len(frame) != cls.FRAME_LEN:
            return None
        if frame[0] != cls.HEAD0 or frame[1] != cls.HEAD1 or frame[-1] != cls.TAIL:
            return None
        dlc = frame[2]
        arb_id = struct.unpack("<I", frame[3:7])[0]
        return (int(arb_id), bytes(frame[7:7 + dlc]))

    def send(self, arb_id: int, data: bytes) -> None:
        ser = self._ensure_port()
        ser.write(self._pack(arb_id, data))

    @classmethod
    def _find_header(cls, buf: bytearray) -> Optional[int]:
        """Index of the first ``0x55 0xAA`` start-of-frame pair, else ``None``."""
        for i in range(len(buf) - 1):
            if buf[i] == cls.HEAD0 and buf[i + 1] == cls.HEAD1:
                return i
        return None

    def _extract_frame(self) -> Optional[Frame]:
        """Pull one validated frame from ``_rxbuf``, resyncing on misalignment.

        Discards leading bytes until the buffer starts with the ``0x55 0xAA``
        header; if the 16-byte window then fails validation (a stray/dropped byte
        shifted alignment, or a corrupt frame), it advances ONE byte and re-scans
        so the reader re-locks onto the next genuine start-of-frame -- instead of
        the old behaviour where a single bad byte desynced the stream permanently.
        """
        buf = self._rxbuf
        while True:
            start = self._find_header(buf)
            if start is None:
                # No header yet; keep a trailing lone 0x55 (it may begin a header
                # split across reads), drop everything else as garbage.
                if buf and buf[-1] == self.HEAD0:
                    del buf[:-1]
                else:
                    buf.clear()
                return None
            if start:
                del buf[:start]                 # drop pre-header garbage
            if len(buf) < self.FRAME_LEN:
                return None                     # await the rest of the frame
            frame = self._unpack(bytes(buf[:self.FRAME_LEN]))
            if frame is not None:
                del buf[:self.FRAME_LEN]         # consume the valid frame
                return frame
            del buf[:1]                          # header but bad frame: re-lock

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        ser = self._ensure_port()
        ser.timeout = timeout
        # Append whatever is available to the persistent buffer (a frame may be
        # split across reads, or several may arrive at once) before parsing, so no
        # bytes are discarded and byte alignment survives across calls.
        chunk = ser.read(self.FRAME_LEN)
        if chunk:
            self._rxbuf.extend(chunk)
        return self._extract_frame()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._rxbuf.clear()


class SimTransport(Transport):
    """In-memory simulation of a set of DM motors (no hardware, no third-party deps).

    Models each motor's virtual state (position/velocity/torque/temperature) and
    replies with a faithful feedback frame (:func:`dm_protocol.encode_feedback`) on
    the motor's master id, in response to:

    * a **refresh** poll (``0x7FF`` + ``0xCC`` -> reply for the addressed motor),
    * a **special** frame (enable ``0xFC`` / disable ``0xFD`` / set-zero ``0xFE`` /
      clear-error ``0xFB`` -> update + reply), and
    * a **MIT** command (decode the commanded position, move the virtual motor to
      it, reply).

    :meth:`recv` returns only queued replies (``None`` when empty), so a polling
    reader (:meth:`MotorBus.telemetry`, which now sends a refresh before reading)
    drives it correctly. Used for offline demos of ``m1_hwconfig`` and as a richer
    test double than :class:`FakeTransport`.
    """

    def __init__(self, motors: Dict[int, dict]) -> None:
        self._m: Dict[int, dict] = {}
        for sid, info in motors.items():
            sid = int(sid)
            self._m[sid] = {
                "master_id": int(info.get("master_id", dm.master_id(sid))),
                "model": info.get("model", "DM4310"),
                "pos": float(info.get("pos", 0.0)),
                "vel": 0.0,
                "torque": 0.0,
                "t_mos": 32,
                "t_rotor": 30,
                "enabled": False,
                "err": 0,
            }
        self._rx: Deque[Frame] = deque()
        self.sent: List[Frame] = []
        self.closed = False

    def _reply(self, sid: int) -> None:
        m = self._m.get(int(sid))
        if m is None:
            return
        data = dm.encode_feedback(
            int(sid) & 0x0F, m["pos"], m["vel"], m["torque"],
            m["t_mos"], m["t_rotor"], m["model"], err=m["err"],
        )
        self._rx.append((m["master_id"], data))

    def send(self, arb_id: int, data: bytes) -> None:
        arb_id = int(arb_id)
        data = bytes(data)
        self.sent.append((arb_id, data))

        # Refresh poll: 0x7FF, [id_lo, id_hi, 0xCC, ...].
        if arb_id == dm.PARAM_ARB_ID and len(data) >= 3 and data[2] == 0xCC:
            self._reply(data[0] | (data[1] << 8))
            return

        # Special control frame: [0xFF*7, opcode], sent to the slave (MIT) id.
        if len(data) == 8 and data[:7] == b"\xff" * 7:
            m = self._m.get(arb_id)
            if m is not None:
                op = data[7]
                if op == 0xFC:
                    m["enabled"] = True
                elif op == 0xFD:
                    m["enabled"] = False
                    m["vel"] = m["torque"] = 0.0
                elif op == 0xFE:           # set zero
                    m["pos"] = 0.0
                elif op == 0xFB:           # clear error
                    m["err"] = 0
                self._reply(arb_id)
            return

        # Otherwise treat it as a MIT command at arb_id == slave id: move there.
        m = self._m.get(arb_id)
        if m is not None and len(data) == 8:
            try:
                cmd = dm.decode_mit_command(data, m["model"])
                m["pos"] = cmd["p"]
                m["torque"] = round(0.1 + 0.02 * abs(cmd["p"]), 3)
                m["t_mos"] = 34
                m["t_rotor"] = 31
            except Exception:  # noqa: BLE001
                pass
            self._reply(arb_id)

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        if self._rx:
            return self._rx.popleft()
        return None

    def close(self) -> None:
        self.closed = True


def make_transport(spec: dict) -> Transport:
    """Build a :class:`Transport` from a spec dict.

    ``spec["kind"]`` selects the backend:

    * ``"fake"`` -> :class:`FakeTransport`
    * ``"sim"`` -> :class:`SimTransport` (``motors``: ``{slave_id: {master_id, model, pos}}``)
    * ``"socketcan"`` -> :class:`SocketCanTransport` (``channel``, ``fd``)
    * ``"serial"`` -> :class:`SerialTransport` (``dev``, ``baud``)
    """
    kind = spec.get("kind")
    if kind == "fake":
        return FakeTransport()
    if kind == "sim":
        return SimTransport(spec.get("motors", {}))
    if kind == "socketcan":
        return SocketCanTransport(
            channel=spec.get("channel", "can0"), fd=bool(spec.get("fd", False))
        )
    if kind == "serial":
        return SerialTransport(
            dev=spec.get("dev", "/dev/ttyACM0"), baud=int(spec.get("baud", 921600))
        )
    raise ValueError(
        f"unknown transport kind {kind!r}; expected 'fake', 'sim', 'socketcan', or 'serial'"
    )
