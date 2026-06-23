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
from typing import Deque, List, Optional, Tuple

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

    def recv(self, timeout: float = 0.0) -> Optional[Frame]:
        ser = self._ensure_port()
        ser.timeout = timeout
        frame = ser.read(self.FRAME_LEN)
        if not frame or len(frame) != self.FRAME_LEN:
            return None
        return self._unpack(frame)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None


def make_transport(spec: dict) -> Transport:
    """Build a :class:`Transport` from a spec dict.

    ``spec["kind"]`` selects the backend:

    * ``"fake"`` -> :class:`FakeTransport`
    * ``"socketcan"`` -> :class:`SocketCanTransport` (``channel``, ``fd``)
    * ``"serial"`` -> :class:`SerialTransport` (``dev``, ``baud``)
    """
    kind = spec.get("kind")
    if kind == "fake":
        return FakeTransport()
    if kind == "socketcan":
        return SocketCanTransport(
            channel=spec.get("channel", "can0"), fd=bool(spec.get("fd", False))
        )
    if kind == "serial":
        return SerialTransport(
            dev=spec.get("dev", "/dev/ttyACM0"), baud=int(spec.get("baud", 921600))
        )
    raise ValueError(
        f"unknown transport kind {kind!r}; expected 'fake', 'socketcan', or 'serial'"
    )
