"""USB serial transport to the real ESP32 node."""
from __future__ import annotations

import time

import serial
from serial.tools import list_ports

from .. import protocol as P
from .base import Transport

BAUD = 921600


def list_serial_ports() -> list[str]:
    return [p.device for p in list_ports.comports()]


def probe_port(port: str, timeout: float = 2.5) -> bool:
    """True if a SPINE node answers SERVICE_DISCOVERY on this port."""
    try:
        with serial.Serial(port, BAUD, timeout=0.1) as s:
            time.sleep(1.8)                       # ESP32 resets when the port opens
            s.reset_input_buffer()
            parser = P.FrameParser()
            end = time.time() + timeout
            s.write(P.encode_frame(P.MsgType.SERVICE_DISCOVERY, b"", P.BROADCAST, 0))
            while time.time() < end:
                for f in parser.feed(s.read(512)):
                    if f.msg_type == P.MsgType.SERVICE_ADVERTISEMENT:
                        return True
    except (serial.SerialException, OSError):
        pass
    return False


def autodetect() -> str | None:
    for port in list_serial_ports():
        if probe_port(port):
            return port
    return None


class SerialTransport(Transport):
    name = "SERIAL"

    def __init__(self, port: str = "auto", baud: int = BAUD):
        self.port, self.baud = port, baud
        self.ser: serial.Serial | None = None

    def open(self) -> None:
        if self.port in ("auto", "", None):
            found = autodetect()
            if not found:
                raise RuntimeError(
                    "No SPINE node found on any serial port. Ports seen: "
                    f"{list_serial_ports() or 'none'}. Check cable/flash, or pass --port COMx.")
            self.port = found
        self.ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=0.05)
        if not str(self.port).startswith("loop://"):
            time.sleep(1.8)                        # let the board finish its reset
        self.ser.reset_input_buffer()

    def close(self) -> None:
        if self.ser:
            self.ser.close()
            self.ser = None

    def write(self, data: bytes) -> None:
        if self.ser:
            self.ser.write(data)

    def read(self, timeout: float = 0.1) -> bytes:
        if not self.ser:
            time.sleep(timeout)
            return b""
        self.ser.timeout = timeout
        data = self.ser.read(max(1, self.ser.in_waiting))
        return data

    @property
    def description(self) -> str:
        return f"SERIAL {self.port} @ {self.baud}"
