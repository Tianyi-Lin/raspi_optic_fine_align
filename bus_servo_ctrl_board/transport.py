from __future__ import annotations
import time
import serial


class SerialTransport:
    """
    适配总线舵机控制板协议：
        55 55 Length Cmd Params...

    特点：
    - 使用内部缓冲区拼帧
    - 自动对齐帧头 0x55 0x55
    - 避免 read_exactly() 那种一旦分片就容易超时的问题
    - 更适合“偶尔只到 1 字节 / 5 字节”的串口流场景
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.5, debug: bool = False):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=8,
            parity="N",
            stopbits=1,
        )
        self.timeout = timeout
        self.debug = debug
        self._rx_buffer = bytearray()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def write(self, data: bytes):
        if self.debug:
            print(f"[TX] {data.hex(' ')}")
        self.ser.write(data)
        self.ser.flush()

    def reset_input_buffer(self):
        self.ser.reset_input_buffer()
        self._rx_buffer.clear()

    def _read_into_buffer(self, max_chunk: int = 256):
        """
        尽量把当前串口里已有数据都读进内部缓冲区。
        """
        waiting = self.ser.in_waiting
        if waiting > 0:
            chunk = self.ser.read(min(waiting, max_chunk))
            if chunk:
                self._rx_buffer.extend(chunk)
                if self.debug:
                    print(f"[RX-CHUNK] {chunk.hex(' ')}")

    def _drop_until_header(self):
        """
        丢弃缓冲区中帧头 55 55 之前的无效字节。
        """
        while len(self._rx_buffer) >= 2:
            if self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0x55:
                return
            dropped = self._rx_buffer.pop(0)
            if self.debug:
                print(f"[DROP] {dropped:02x}")

        # 如果只剩 1 个字节，也可能是半个帧头，先保留
        if len(self._rx_buffer) == 1 and self._rx_buffer[0] != 0x55:
            dropped = self._rx_buffer.pop(0)
            if self.debug:
                print(f"[DROP] {dropped:02x}")

    def read_frame(self) -> bytes:
        """
        从串口中读取一帧完整控制板协议数据：
            55 55 Length Cmd Params...

        返回完整帧字节串。
        """
        start = time.time()

        while time.time() - start < self.timeout:
            # 先尽可能收数据
            self._read_into_buffer()

            # 对齐帧头
            self._drop_until_header()

            # 至少要有：55 55 Length
            if len(self._rx_buffer) < 3:
                time.sleep(0.005)
                continue

            # 现在 buffer[0:2] 应该是 55 55
            if not (self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0x55):
                time.sleep(0.005)
                continue

            length = self._rx_buffer[2]

            # 基本合法性检查
            if length < 2:
                # Length 至少应包含 Cmd
                if self.debug:
                    print(f"[BAD-LENGTH] {length}, buffer={self._rx_buffer.hex(' ')}")
                # 丢掉一个 0x55，重新找帧头
                self._rx_buffer.pop(0)
                continue

            full_len = length + 3  # 2字节帧头 + 1字节Length + 后面length字节

            # 帧还没收完整，继续等
            if len(self._rx_buffer) < full_len:
                time.sleep(0.005)
                continue

            # 截取完整帧
            frame = bytes(self._rx_buffer[:full_len])
            del self._rx_buffer[:full_len]

            if self.debug:
                print(f"[FRAME] {frame.hex(' ')}")

            return frame

        # 超时
        partial = bytes(self._rx_buffer)
        raise TimeoutError(
            f"read_frame timeout, buffered={len(partial)} byte, partial={partial.hex(' ')}"
        )