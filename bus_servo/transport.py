try:
    import serial
except ModuleNotFoundError:
    serial = None


class SerialTransport:
    """
    半双工串口传输层。
    如果硬件需要额外控制 TX/RX 方向，
    可以在 _before_write / _before_read 里加 GPIO 切换。
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.5):
        if serial is None:
            raise ModuleNotFoundError("pyserial is required: pip install pyserial")
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
        pass

    def _before_read(self):
        pass

    def write(self, data: bytes):
        self._before_write()
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
        self._before_read()

        # 对齐帧头 0x55 0x55
        first = self.read_exactly(1)
        while first != b'\x55':
            first = self.read_exactly(1)

        second = self.read_exactly(1)
        while second != b'\x55':
            first = second
            if first != b'\x55':
                first = self.read_exactly(1)
            second = self.read_exactly(1)

        # 读取 ID 和 Length
        head_rest = self.read_exactly(2)
        servo_id = head_rest[0]
        length = head_rest[1]

        # 协议里 Length 包含 Length 自己这个字节
        # 但这个字节已经读过了，所以后续只需再读 length - 1 字节
        remain = length - 1
        if remain <= 0:
            raise ValueError(f"invalid frame length field: {length}")

        body = self.read_exactly(remain)

        return b'\x55\x55' + bytes([servo_id, length]) + body
