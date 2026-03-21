from __future__ import annotations
import serial
import time


class SerialTransport:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.2):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=8,
            parity="N",
            stopbits=1,
        )

    def close(self):
        self.ser.close()

    def write(self, data: bytes):
        self.ser.write(data)
        self.ser.flush()

    def read_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"read timeout, expect {n} byte got {len(buf)}")
            buf.extend(chunk)
        return bytes(buf)

    def read_frame(self) -> bytes:
        # 对齐 55 55
        first = self.read_exactly(1)
        while first != b"\x55":
            first = self.read_exactly(1)

        second = self.read_exactly(1)
        while second != b"\x55":
            first = second
            if first != b"\x55":
                first = self.read_exactly(1)
            second = self.read_exactly(1)

        # 读 length
        length_b = self.read_exactly(1)
        length = length_b[0]

        # length 已经不包括两个帧头，所以后面还要读 length 个字节
        rest = self.read_exactly(length)

        return b"\x55\x55" + bytes([length]) + rest