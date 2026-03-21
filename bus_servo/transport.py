from __future__ import annotations
import time
import serial


class SerialTransport:
    """
    半双工串口传输层。
    如果你的硬件需要额外控制 TX/RX 方向，
    可以在 _before_write / _before_read 里加 GPIO 切换。
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.02):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=8,
            parity='N',
            stopbits=1,
        )

    def close(self):
        self.ser.close()

    def _before_write(self):
        # TODO: 若硬件需要切换到发送模式，在这里做
        pass

    def _before_read(self):
        # TODO: 若硬件需要切换到接收模式，在这里做
        pass

    def write(self, data: bytes):
        self._before_write()
        self.ser.reset_output_buffer()
        self.ser.write(data)
        self.ser.flush()

    def read_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"read timeout, expect {n} bytes, got {len(buf)}")
            buf.extend(chunk)
        return bytes(buf)

    def read_frame(self) -> bytes:
        self._before_read()

        # 对齐帧头 55 55
        first = self.read_exactly(1)
        while first != b'\x55':
            first = self.read_exactly(1)

        second = self.read_exactly(1)
        while second != b'\x55':
            first = second
            if first != b'\x55':
                first = self.read_exactly(1)
            second = self.read_exactly(1)

        # 读 ID + Length
        head_rest = self.read_exactly(2)
        servo_id = head_rest[0]
        length = head_rest[1]

        # 再读 cmd + params + checksum，共 length 字节
        body = self.read_exactly(length)

        return b'\x55\x55' + bytes([servo_id, length]) + body