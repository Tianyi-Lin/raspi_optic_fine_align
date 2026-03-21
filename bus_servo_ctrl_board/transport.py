from __future__ import annotations
import time
import serial


class SerialTransport:
    """
    控制板协议：
        55 55 Length Cmd Params...

    其中：
    - Length = N + 2
    - 整帧总长度 = Length + 2
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0, debug: bool = False):
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
        waiting = self.ser.in_waiting
        if waiting > 0:
            chunk = self.ser.read(min(waiting, max_chunk))
            if chunk:
                self._rx_buffer.extend(chunk)
                if self.debug:
                    print(f"[RX-CHUNK] {chunk.hex(' ')}")

    def _drop_until_header(self):
        while len(self._rx_buffer) >= 2:
            if self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0x55:
                return
            dropped = self._rx_buffer.pop(0)
            if self.debug:
                print(f"[DROP] {dropped:02x}")

        if len(self._rx_buffer) == 1 and self._rx_buffer[0] != 0x55:
            dropped = self._rx_buffer.pop(0)
            if self.debug:
                print(f"[DROP] {dropped:02x}")

    def read_frame(self) -> bytes:
        start = time.time()

        while time.time() - start < self.timeout:
            self._read_into_buffer()
            self._drop_until_header()

            if len(self._rx_buffer) < 3:
                time.sleep(0.005)
                continue

            if not (self._rx_buffer[0] == 0x55 and self._rx_buffer[1] == 0x55):
                time.sleep(0.005)
                continue

            length = self._rx_buffer[2]

            if length < 2:
                if self.debug:
                    print(f"[BAD-LENGTH] {length}, buffer={self._rx_buffer.hex(' ')}")
                self._rx_buffer.pop(0)
                continue

            full_len = length + 2

            if len(self._rx_buffer) < full_len:
                time.sleep(0.005)
                continue

            frame = bytes(self._rx_buffer[:full_len])
            del self._rx_buffer[:full_len]

            if self.debug:
                print(f"[FRAME] {frame.hex(' ')}")

            return frame

        partial = bytes(self._rx_buffer)
        raise TimeoutError(
            f"read_frame timeout, buffered={len(partial)} byte, partial={partial.hex(' ')}"
        )